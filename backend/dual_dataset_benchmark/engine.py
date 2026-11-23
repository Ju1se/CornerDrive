"""Federated round loop of the dual-dataset rarity benchmark.

One *unit* is one (seed, scenario, method) trajectory. Within a seed every
method starts from the same warm-up checkpoint and meets the same sampled
clients, the same local mini-batches, the same attacker behaviour and the
same golden audit draw in every round; only the aggregation differs, and each
method evolves its own global model.

Every round the engine logs, for each of the sampled updates, its hidden
ground truth together with the two loss drifts on the golden sets and the
score every baseline would give it. The analysis reads these logs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import aggregators as agg
from . import backend as be
from .backend import CNN, Backend
from .cifar_golden import (
    NUM_CLASSES,
    RARE_LABELS,
    ROLES,
    Federation,
    FederationConfig,
    GoldenAnnotation,
    build_federation,
)
from .detectors import (
    HONEST,
    RARITY,
    REASONS,
    UNAUDITED,
    VERDICTS,
    RoutingPolicy,
    VerdictPolicy,
    cosine_to,
    dual_verdict,
    l1_signals,
    prototype_readout,
    route,
)

TRUTHS = ("HONEST", "RARITY", "NOISE", "FRAUD")
BASELINES = ("fedavg", "geomed", "krum", "fltrust", "zeno", "zenopp", "fedval")
CORNERDRIVE = ("cornerdrive", "cd_exhaustive", "cd_main_only", "cd_corner_only", "cd_embedding")
METHODS = ("oracle", *BASELINES, *CORNERDRIVE)
METHOD_LABELS = {
    "oracle": "Oracle (ground-truth filter)",
    "fedavg": "FedAvg",
    "geomed": "GeoMed",
    "krum": "Multi-Krum",
    "fltrust": "FLTrust",
    "zeno": "Zeno",
    "zenopp": "Zeno++",
    "fedval": "FedVal",
    "cornerdrive": "CornerDrive (L1 + L2 dual)",
    "cd_exhaustive": "CornerDrive (exhaustive L2 dual)",
    "cd_main_only": "Main-only audit (exhaustive)",
    "cd_corner_only": "Corner-only audit (exhaustive)",
    "cd_embedding": "CornerDrive (exhaustive L2 dual, embedding space)",
}


@dataclass(frozen=True)
class EngineConfig:
    rounds: int = 100
    warmup_rounds: int = 20
    clients_per_round: int = 20
    local_epochs: int = 2
    batch_size: int = 32
    local_lr: float = 0.05
    audit_per_class: int = 100          # golden images drawn per class every round
    byzantine_budget: int = 5           # round(20 * (0.20 + 0.05)), as the repository estimates it
    sign_flip_scale: float = 3.0
    noise_ratio: float = 2.0            # |noise| / |update| of a noisy client
    zeno_rho: float = 1e-4
    fedval_s1: float = 3.0
    fedval_s1_avg: float = 5.0
    fedval_c: float = 3.0
    fedval_s2_init: float = 3.0
    extra_audit_steps: tuple[float, ...] = ()   # extra replica step sizes to log
    # What the server uses as its rare golden set (proxy stress tests):
    #   "rare"         the annotated rare classes                    (default)
    #   "partial"      only the first two rare classes; the third is absent
    #   "random_main"  a random slice of the normal golden set (negative control)
    corner_proxy: str = "rare"


@dataclass(frozen=True)
class CornerDrivePolicy:
    """Frozen CornerDrive operating point.

    audit_step   eta of the replica  W' = W + eta * u_i
    drift_mode   "absolute": dL = L(W') - L(W)
                 "relative": dL / L(W), a phase-invariant version of the same drift
    """

    routing: RoutingPolicy
    verdict: VerdictPolicy
    audit_step: float = 1.0
    drift_mode: str = "relative"
    # "loss": L is the replica's cross-entropy on the golden set.
    # "embedding": L is the prototype read-out of the replica's own embedding
    # on the golden set (the replica step is 1 there).
    space: str = "loss"

    def to_json(self) -> dict[str, Any]:
        return {"routing": asdict(self.routing), "verdict": asdict(self.verdict),
                "audit_step": self.audit_step, "drift_mode": self.drift_mode,
                "space": self.space}

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "CornerDrivePolicy":
        return CornerDrivePolicy(
            routing=RoutingPolicy(**payload["routing"]),
            verdict=VerdictPolicy(**payload["verdict"]),
            audit_step=float(payload.get("audit_step", 1.0)),
            drift_mode=str(payload.get("drift_mode", "relative")),
            space=str(payload.get("space", "loss")),
        )


@dataclass
class RoundDraw:
    clients: np.ndarray
    positions: torch.Tensor
    flips: torch.Tensor
    audit_index: torch.Tensor
    root_index: torch.Tensor
    noise_seed: int
    recheck_seed: int


class SeedContext:
    """Everything the methods of one seed share."""

    def __init__(
        self,
        data: dict[str, Any],
        ann: GoldenAnnotation,
        seed: int,
        cfg: EngineConfig,
        fed_cfg: FederationConfig | None = None,
        device: torch.device | str = "cuda",
        x_device: torch.Tensor | None = None,
        y_device: torch.Tensor | None = None,
        backend: Backend = CNN,
    ):
        """``x_device`` holds one row per pool image in the form the backend
        consumes: uint8 images for the CNN, standardised embeddings for the head."""
        self.seed = int(seed)
        self.cfg = cfg
        self.backend = backend
        self.fed_cfg = fed_cfg or FederationConfig()
        self.device = torch.device(device)
        self._y_np = data["y"]
        self._pool = ann.client_pool
        self.x = x_device if x_device is not None else torch.from_numpy(data["x"]).to(self.device)
        self.y = y_device if y_device is not None else torch.from_numpy(data["y"]).to(self.device)
        self.golden_by_class = [
            torch.from_numpy(ann.golden[data["y"][ann.golden] == k]) for k in range(NUM_CLASSES)
        ]
        self.golden_all = torch.from_numpy(ann.golden)
        self.eval_index = torch.from_numpy(ann.eval).to(self.device)
        self.eval_y = self.y[self.eval_index]
        self.rare_labels = torch.tensor(RARE_LABELS, device=self.device)
        self.eval_rare = torch.isin(self.eval_y, self.rare_labels)
        self._federations: dict[str, Federation] = {}
        self._fed_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.w0: torch.Tensor | None = None

    # -- data ---------------------------------------------------------------
    def federation(self, scenario: str) -> Federation:
        if scenario not in self._federations:
            cfg = FederationConfig(**{**asdict(self.fed_cfg), "scenario": scenario})
            fed = build_federation(self._y_np, self._pool, seed=self.seed, config=cfg)
            self._federations[scenario] = fed
            self._fed_tensors[scenario] = (
                torch.from_numpy(fed.client_index).to(self.device),
                torch.from_numpy(fed.train_labels).to(self.device),
            )
        return self._federations[scenario]

    def round_draw(self, round_index: int, candidates: np.ndarray | None = None) -> RoundDraw:
        """Deterministic in (seed, round): identical for every method."""
        cfg = self.cfg
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + round_index + 100_000)
        n = self.fed_cfg.samples_per_client
        k = cfg.clients_per_round
        population = candidates if candidates is not None else np.arange(self.fed_cfg.num_clients)
        clients = population[torch.randperm(len(population), generator=gen)[:k].numpy()]
        rows = k + 1                                    # + the server's own shard
        per_epoch = n // cfg.batch_size
        positions = torch.cat([
            torch.argsort(torch.rand(rows, n, generator=gen), dim=1)[:, :per_epoch * cfg.batch_size]
            .view(rows, per_epoch, cfg.batch_size).permute(1, 0, 2)
            for _ in range(cfg.local_epochs)
        ])
        flips = torch.rand(positions.shape, generator=gen) < 0.5
        audit = torch.cat([
            bucket[torch.randperm(len(bucket), generator=gen)[:cfg.audit_per_class]]
            for bucket in self.golden_by_class
        ])
        root = self.golden_all[torch.randperm(len(self.golden_all), generator=gen)[:n]]
        seeds = torch.randint(0, 2**31 - 1, (2,), generator=gen)
        return RoundDraw(
            clients=clients,
            positions=positions.to(self.device),
            flips=flips.to(self.device),
            audit_index=audit.to(self.device),
            root_index=root.to(self.device),
            noise_seed=int(seeds[0]),
            recheck_seed=int(seeds[1]),
        )

    # -- training pieces ----------------------------------------------------
    def train_round(self, w: torch.Tensor, scenario: str, draw: RoundDraw
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Local training of the sampled clients and of the server on its own
        golden shard. Returns (client updates [K, P], server update [P])."""
        index, labels = self._fed_tensors[scenario]
        clients = torch.from_numpy(draw.clients).to(self.device)
        shard_index = torch.cat([index[clients], draw.root_index[None]])
        shard_labels = torch.cat([labels[clients], self.y[draw.root_index][None]])
        updates = be.local_train(self.backend, w, self.x, shard_index, shard_labels,
                                 draw.positions, draw.flips, self.cfg.local_lr)
        return updates[:-1].clone(), updates[-1].clone()

    def apply_update_attacks(self, updates: torch.Tensor, roles: list[str], draw: RoundDraw) -> None:
        for i, role in enumerate(roles):
            if role == "sign_flip":
                updates[i] = -self.cfg.sign_flip_scale * updates[i]
            elif role == "noise":
                gen = torch.Generator().manual_seed(draw.noise_seed + i)
                noise = torch.randn(updates.shape[1], generator=gen).to(self.device)
                updates[i] = updates[i] + noise * (
                    self.cfg.noise_ratio * updates[i].norm() / noise.norm()
                )

    def evaluate(self, w: torch.Tensor) -> dict[str, Any]:
        loss, correct = be.example_losses(self.backend, w[None], self.x, self.eval_index,
                                          self.eval_y, self.backend.eval_chunk)
        loss, correct = loss[0], correct[0].float()
        per_class = torch.stack([correct[self.eval_y == k].mean() for k in range(NUM_CLASSES)])
        return {
            "acc_normal": float(correct[~self.eval_rare].mean()),
            "acc_rare": float(correct[self.eval_rare].mean()),
            "acc_all": float(correct.mean()),
            "loss_normal": float(loss[~self.eval_rare].mean()),
            "loss_rare": float(loss[self.eval_rare].mean()),
            "acc_class": per_class.cpu().numpy(),
        }

    def warmup(self, cache_dir: Path | None = None) -> torch.Tensor:
        """Shared initial checkpoint: FedAvg over clients with clean data
        (honest and rare-rich honest roles of the mixed assignment)."""
        if self.w0 is not None:
            return self.w0
        cache = None
        if cache_dir is not None:
            cache = Path(cache_dir) / (
                f"warmup_{self.backend.name}_s{self.seed}_r{self.cfg.warmup_rounds}.pt")
            if cache.exists():
                self.w0 = torch.load(cache, map_location=self.device)
                return self.w0
        fed = self.federation("mixed")
        benign = np.array([i for i, r in enumerate(fed.roles) if r in ("honest", "rarity")])
        w = self.backend.init_params(self.seed, self.device)
        for step in range(self.cfg.warmup_rounds):
            draw = self.round_draw(step - self.cfg.warmup_rounds, candidates=benign)
            updates, _root = self.train_round(w, "mixed", draw)
            w = w + updates.mean(dim=0)
        self.w0 = w
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(w.cpu(), cache)
        return self.w0


def _drift(client_loss: torch.Tensor, base_loss: torch.Tensor, mask: torch.Tensor
           ) -> tuple[np.ndarray, np.ndarray]:
    """Mean paired loss drift on the masked golden images and its standard error."""
    diff = client_loss[:, mask] - base_loss[mask][None]
    mean = diff.mean(dim=1)
    se = diff.std(dim=1) / math.sqrt(diff.shape[1])
    return mean.double().cpu().numpy(), se.double().cpu().numpy()


def run_unit(
    ctx: SeedContext,
    scenario: str,
    method: str,
    policies: dict[str, CornerDrivePolicy] | None = None,
    *,
    log_all_scores: bool | None = None,
) -> dict[str, Any]:
    """Run one trajectory; returns arrays (round level and update level) + meta.

    ``policies`` maps "loss" and "embedding" to the frozen CornerDrive
    operating points; ``cd_embedding`` uses the second, every other
    CornerDrive method the first.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    policy = None
    if method in CORNERDRIVE:
        policy = (policies or {}).get("embedding" if method == "cd_embedding" else "loss")
        if policy is None:
            raise ValueError(f"{method} needs a CornerDrivePolicy")
    cfg = ctx.cfg
    backend = ctx.backend
    fed = ctx.federation(scenario)
    w = ctx.warmup().clone()
    k = cfg.clients_per_round
    s2 = cfg.fedval_s2_init
    full = (method in ("oracle", "fedavg")) if log_all_scores is None else log_all_scores
    started = time.time()

    upd: dict[str, list[np.ndarray]] = {}
    rnd: dict[str, list[Any]] = {}

    def log_u(name: str, value) -> None:
        upd.setdefault(name, []).append(np.asarray(value))

    def log_r(name: str, value) -> None:
        rnd.setdefault(name, []).append(value)

    for t in range(cfg.rounds):
        draw = ctx.round_draw(t)
        roles = [fed.roles[c] for c in draw.clients]
        truth = [fed.truth[c] for c in draw.clients]
        updates, root = ctx.train_round(w, scenario, draw)
        ctx.apply_update_attacks(updates, roles, draw)

        # ---- golden audit draw: 100 images per class, ordered by class -----
        audit_y = ctx.y[draw.audit_index]
        position = torch.arange(audit_y.numel(), device=ctx.device) % cfg.audit_per_class
        is_rare = torch.isin(audit_y, ctx.rare_labels)
        if cfg.corner_proxy == "rare":
            rare_mask, main_mask = is_rare, ~is_rare
        elif cfg.corner_proxy == "partial":
            rare_mask = torch.isin(audit_y, ctx.rare_labels[:2])
            main_mask = ~is_rare
        elif cfg.corner_proxy == "random_main":
            slice_size = round(cfg.audit_per_class * len(RARE_LABELS)
                               / (NUM_CLASSES - len(RARE_LABELS)))
            rare_mask = ~is_rare & (position < slice_size)
            main_mask = ~is_rare & (position >= slice_size)
        else:
            raise ValueError(f"unknown corner proxy {cfg.corner_proxy!r}")
        replicas = torch.cat([w[None], w[None] + updates])
        loss, _, feats = be.example_losses(backend, replicas, ctx.x, draw.audit_index, audit_y,
                                           backend.audit_chunk, with_features=True)
        base, client = loss[0], loss[1:]
        base_main = float(base[main_mask].mean())
        base_corner = float(base[rare_mask].mean())
        d_main, se_main = _drift(client, base, main_mask)
        d_corner, se_corner = _drift(client, base, rare_mask)
        log_u("d_main", d_main); log_u("d_corner", d_corner)
        log_u("se_main", se_main); log_u("se_corner", se_corner)
        log_r("base_main", base_main); log_r("base_corner", base_corner)

        # ---- the same test in embedding space (prototype read-out) ---------
        support = position < cfg.audit_per_class // 2
        proto_loss, _ = prototype_readout(feats, audit_y, support, NUM_CLASSES)
        query_rare, query_main = rare_mask[~support], main_mask[~support]
        e_main, _ = _drift(proto_loss[1:], proto_loss[0], query_main)
        e_corner, _ = _drift(proto_loss[1:], proto_loss[0], query_rare)
        base_e_main = float(proto_loss[0][query_main].mean())
        base_e_corner = float(proto_loss[0][query_rare].mean())
        log_u("e_main", e_main); log_u("e_corner", e_corner)
        log_r("base_e_main", base_e_main); log_r("base_e_corner", base_e_corner)

        # Reference runs also log the drift at the frozen policy's replica step,
        # so the frozen dual test can be replayed on their update stream.
        logged_steps = set(cfg.extra_audit_steps)
        if full and policies and "loss" in policies:
            logged_steps.add(policies["loss"].audit_step)
        steps_needed = set(logged_steps)
        if policy is not None and policy.audit_step != 1.0:
            steps_needed.add(policy.audit_step)
        stepped: dict[float, tuple[np.ndarray, np.ndarray]] = {1.0: (d_main, d_corner)}
        for eta in sorted(steps_needed):
            if eta == 1.0:
                continue
            loss_eta, _ = be.example_losses(backend, w[None] + eta * updates, ctx.x,
                                            draw.audit_index, audit_y, backend.audit_chunk)
            stepped[eta] = (_drift(loss_eta, base, main_mask)[0],
                            _drift(loss_eta, base, rare_mask)[0])
            if eta in logged_steps:
                log_u(f"d_main_eta{eta:g}", stepped[eta][0])
                log_u(f"d_corner_eta{eta:g}", stepped[eta][1])

        mean_loss = client.mean(dim=1)
        class_loss = client.view(k, NUM_CLASSES, cfg.audit_per_class).mean(dim=2)
        norms = updates.norm(dim=1)
        log_u("norm", norms.cpu().numpy())

        # ---- scores of the baselines (all of them on the reference runs) ---
        scores: dict[str, torch.Tensor] = {}
        if full or method == "zeno":
            scores["zeno"] = agg.zeno_scores(base.mean(), mean_loss, updates, cfg.zeno_rho)
        if full or method == "zenopp":
            val_grad = be.loss_gradient(backend, w, ctx.x, draw.audit_index, audit_y)
            scores["zenopp"] = agg.zenopp_scores(updates, val_grad, cfg.zeno_rho)
        if full or method == "fltrust":
            scores["fltrust"] = cosine_to(updates, root)
        if full or method == "krum":
            scores["krum"] = -agg.krum_scores(updates, cfg.byzantine_budget)
        if full or method == "fedval":
            scores["fedval"] = agg.fedval_scores(class_loss, mean_loss, s2, cfg.fedval_s1,
                                                 cfg.fedval_s1_avg, cfg.fedval_c)
        for name, value in scores.items():
            log_u(f"score_{name}", value.double().cpu().numpy())
        signals = None
        if full or method in CORNERDRIVE:
            signals = l1_signals(updates)
            log_u("cos_dev", signals.cosine_deviation)
            log_u("norm_mad", signals.norm_mad)
            log_u("sign_dis", signals.sign_disagreement)

        # ---- aggregation ---------------------------------------------------
        verdict_final = [UNAUDITED] * k
        routed = np.zeros(k, dtype=bool)
        reasons = ["bypass"] * k
        if method == "oracle":
            mask = torch.tensor([t_ in (HONEST, RARITY) for t_ in truth], device=ctx.device)
            weight = mask.float() / mask.float().sum().clamp_min(1.0)
            delta = (updates * weight[:, None]).sum(dim=0)
        elif method == "fedavg":
            delta, weight = agg.fedavg(updates)
        elif method == "geomed":
            delta, weight = agg.geomed(updates)
        elif method == "krum":
            delta, weight = agg.multi_krum(updates, cfg.byzantine_budget)
        elif method == "fltrust":
            delta, weight = agg.fltrust(updates, root)
        elif method == "zeno":
            delta, weight = agg.zeno(updates, scores["zeno"], cfg.byzantine_budget)
        elif method == "zenopp":
            delta, weight = agg.zenopp(updates, scores["zenopp"])
        elif method == "fedval":
            cands = agg.fedval_s2_candidates(s2)
            weights = torch.stack([
                agg.fedval_weights(agg.fedval_scores(class_loss, mean_loss, c, cfg.fedval_s1,
                                                     cfg.fedval_s1_avg, cfg.fedval_c))
                for c in cands
            ])
            deltas = weights @ updates
            cand_loss, _ = be.example_losses(backend, w[None] + deltas, ctx.x,
                                             draw.audit_index, audit_y, backend.audit_chunk)
            best = int(cand_loss.mean(dim=1).argmin())
            s2, weight, delta = cands[best], weights[best], deltas[best]
        else:
            if policy.space == "embedding":
                dm, dc, scale_main, scale_corner = e_main, e_corner, base_e_main, base_e_corner
            else:
                dm, dc = stepped[policy.audit_step]
                scale_main, scale_corner = base_main, base_corner
            if policy.drift_mode == "relative":
                dm, dc = dm / max(scale_main, 1e-8), dc / max(scale_corner, 1e-8)
            elif policy.drift_mode != "absolute":
                raise ValueError(f"unknown drift mode {policy.drift_mode!r}")
            routing = policy.routing
            verdict_policy = policy.verdict
            if method != "cornerdrive":
                routing = RoutingPolicy(**{**asdict(routing), "exhaustive": True})
            log_u("verdict_drift_main", dm); log_u("verdict_drift_corner", dc)
            if method == "cd_main_only":
                verdict_policy = VerdictPolicy(**{**asdict(verdict_policy), "mode": "main_only"})
            elif method == "cd_corner_only":
                verdict_policy = VerdictPolicy(**{**asdict(verdict_policy), "mode": "corner_only"})
            routed, nominated, reasons = route(signals, routing,
                                               np.random.default_rng(draw.recheck_seed))
            verdict_all = dual_verdict(dm, dc, nominated, verdict_policy)
            accepted = ~routed | np.isin(verdict_all, (RARITY, HONEST))
            verdict_final = [str(v) if r else UNAUDITED for v, r in zip(verdict_all, routed)]
            mask = torch.from_numpy(accepted).to(ctx.device)
            weight = mask.float() / mask.float().sum().clamp_min(1.0)
            delta = (updates * weight[:, None]).sum(dim=0)
            log_u("nominated", nominated)

        w = w + delta

        # ---- logs ----------------------------------------------------------
        log_u("client", draw.clients)
        log_u("role", [ROLES.index(r) for r in roles])
        log_u("truth", [TRUTHS.index(t_) for t_ in truth])
        log_u("weight", weight.double().cpu().numpy())
        log_u("routed", routed)
        log_u("reason", [REASONS.index(r) for r in reasons])
        log_u("verdict", [VERDICTS.index(v) for v in verdict_final])
        metrics = ctx.evaluate(w)
        for name, value in metrics.items():
            log_r(name, value)
        log_r("fedval_s2", s2)
        log_r("delta_norm", float(delta.norm()))

    out: dict[str, Any] = {f"u_{name}": np.stack(values) for name, values in upd.items()}
    out.update({f"r_{name}": np.asarray(values) for name, values in rnd.items()})
    out["meta"] = json.dumps({
        "seed": ctx.seed,
        "scenario": scenario,
        "method": method,
        "model": backend.name,
        "label": METHOD_LABELS[method],
        "engine": asdict(cfg),
        "federation": fed.describe(),
        "policy": policy.to_json() if (policy is not None and method in CORNERDRIVE) else None,
        "roles": list(ROLES),
        "truths": list(TRUTHS),
        "verdicts": list(VERDICTS),
        "reasons": list(REASONS),
        "seconds": time.time() - started,
    })
    return out
