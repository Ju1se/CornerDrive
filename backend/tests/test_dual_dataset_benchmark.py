"""Checks for the CIFAR-10 dual-dataset rarity benchmark.

CPU only and independent of the CIFAR files: the data pieces are exercised on
synthetic labels and the engine on synthetic embeddings. Runs under pytest, or
directly with ``python backend/tests/test_dual_dataset_benchmark.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dual_dataset_benchmark import aggregators as agg  # noqa: E402
from dual_dataset_benchmark import batched_cnn, batched_head  # noqa: E402
from dual_dataset_benchmark.analysis import auroc, rarity_discovery  # noqa: E402
from dual_dataset_benchmark.backend import CNN, head_backend, local_train  # noqa: E402
from dual_dataset_benchmark.cifar_golden import (  # noqa: E402
    CORNER_HARM_MAP,
    NORMAL_LABELS,
    RARE_LABELS,
    FederationConfig,
    annotate_golden,
    build_federation,
)
from dual_dataset_benchmark.detectors import (  # noqa: E402
    FRAUD,
    HONEST,
    NOISE,
    RARITY,
    RoutingPolicy,
    VerdictPolicy,
    dual_verdict,
    geometric_median,
    l1_signals,
    prototype_readout,
    route,
)
from dual_dataset_benchmark.engine import (  # noqa: E402
    METHODS,
    CornerDrivePolicy,
    EngineConfig,
    SeedContext,
    run_unit,
)


def _labels(per_class: int = 400) -> np.ndarray:
    return np.repeat(np.arange(10), per_class)


# -- golden annotation -------------------------------------------------------
def test_golden_annotation_is_a_stratified_quarter():
    y = _labels(400)
    ann = annotate_golden(y, eval_per_class=20)
    everything = np.concatenate([ann.golden, ann.client_pool, ann.eval])
    assert len(np.unique(everything)) == len(y)                 # disjoint and complete
    assert len(ann.golden) == len(y) // 4
    assert np.all(np.bincount(y[ann.golden], minlength=10) == 100)
    assert set(y[ann.golden_rare]) == set(RARE_LABELS)
    assert set(y[ann.golden_normal]) == set(NORMAL_LABELS)
    assert np.all(np.bincount(y[ann.eval], minlength=10) == 20)
    again = annotate_golden(y, eval_per_class=20)
    assert np.array_equal(ann.golden_rare, again.golden_rare)   # deterministic


def test_federation_roles_shards_and_poisoning():
    y = _labels(400)
    ann = annotate_golden(y, eval_per_class=20)
    cfg = FederationConfig(num_clients=30, samples_per_client=32, n_rarity=3, n_corner_harm=2,
                           n_sign_flip=3, n_label_flip=3, n_noise=2)
    fed = build_federation(y, ann.client_pool, seed=7, config=cfg)
    flat = fed.client_index.ravel()
    assert len(np.unique(flat)) == flat.size and np.isin(flat, ann.client_pool).all()
    roles = np.array(fed.roles)
    assert {r: int((roles == r).sum()) for r in set(roles)} == {
        "honest": 17, "rarity": 3, "corner_harm": 2, "sign_flip": 3, "label_flip": 3, "noise": 2}
    assert fed.rare_share[np.isin(roles, ("rarity", "corner_harm"))].min() > 0.5
    assert fed.rare_share[roles == "honest"].max() < 0.1
    # label-flip poisons every label, corner-harm only the rare ones, the rest none
    for c in range(cfg.num_clients):
        changed = fed.train_labels[c] != fed.true_labels[c]
        if roles[c] == "label_flip":
            assert np.array_equal(fed.train_labels[c], 9 - fed.true_labels[c])
        elif roles[c] == "corner_harm":
            assert np.array_equal(changed, np.isin(fed.true_labels[c], RARE_LABELS))
            assert set(fed.train_labels[c][changed]) <= set(CORNER_HARM_MAP.values())
        else:
            assert not changed.any()
    assert [fed.truth[c] for c in np.flatnonzero(roles == "rarity")] == ["RARITY"] * 3

    # the no-attack scenario of the same seed has the same shards and no attacker
    clean = build_federation(y, ann.client_pool, seed=7,
                             config=FederationConfig(**{**cfg.__dict__, "scenario": "no_attack"}))
    assert np.array_equal(clean.client_index, fed.client_index)
    assert np.array_equal(clean.train_labels, clean.true_labels)
    assert set(clean.truth) == {"HONEST", "RARITY"} and clean.truth.count("RARITY") == 5


# -- models ------------------------------------------------------------------
class _RefCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.gn1 = nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32)
        self.conv2, self.gn2 = nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64)
        self.conv3, self.gn3 = nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)
        x = F.relu(self.gn3(self.conv3(x)))
        return self.fc(x.mean((2, 3)))


def test_batched_cnn_matches_a_plain_module():
    torch.manual_seed(0)
    flat = torch.stack([batched_cnn.init_params(s) for s in range(3)]).double()
    flat = flat + 0.05 * torch.randn_like(flat)
    x = torch.randn(3, 4, 3, 32, 32).double()
    own = batched_cnn.forward(flat, x)
    shared = batched_cnn.forward(flat, x[0])
    for c in range(3):
        ref = _RefCNN().double()
        ref.load_state_dict({n: v[c] for n, v in batched_cnn.unflatten(flat).items()})
        assert torch.allclose(ref(x[c]), own[c], atol=1e-9)
        assert torch.allclose(ref(x[0]), shared[c], atol=1e-9)


def test_batched_head_matches_a_plain_module():
    dim = 12
    backend = head_backend(dim)
    flat = torch.stack([batched_head.init_params(s, "cpu", dim) for s in range(3)]).double()
    assert flat.shape[1] == backend.num_params
    x = torch.randn(3, 5, dim).double()
    own, features = backend.forward(flat, x, True)
    shared = backend.forward(flat, x[0])
    a = batched_head.HIDDEN * dim
    for c in range(3):
        ref = nn.Sequential(nn.Linear(dim, batched_head.HIDDEN), nn.ReLU(),
                            nn.Linear(batched_head.HIDDEN, 10)).double()
        ref[0].weight.data = flat[c, :a].view(batched_head.HIDDEN, dim)
        ref[0].bias.data = flat[c, a:a + batched_head.HIDDEN]
        ref[2].weight.data = flat[c, a + batched_head.HIDDEN:-10].view(10, batched_head.HIDDEN)
        ref[2].bias.data = flat[c, -10:]
        assert torch.allclose(ref(x[c]), own[c], atol=1e-9)
        assert torch.allclose(ref(x[0]), shared[c], atol=1e-9)
    assert features.shape == (3, 5, batched_head.HIDDEN)


def test_local_training_of_a_client_ignores_the_other_clients():
    dim, n, clients = 8, 16, 4
    backend = head_backend(dim)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(200, dim, generator=gen).double()
    shard = torch.randint(0, 200, (clients, n), generator=gen)
    labels = torch.randint(0, 10, (clients, n), generator=gen)
    positions = torch.argsort(torch.rand(clients, n, generator=gen), dim=1).view(clients, 2, 8)
    positions = positions.permute(1, 0, 2)
    flips = torch.zeros(positions.shape, dtype=torch.bool)
    start = batched_head.init_params(0, "cpu", dim).double()
    together = local_train(backend, start, x, shard, labels, positions, flips, 0.1)
    alone = local_train(backend, start, x, shard[2:3], labels[2:3], positions[:, 2:3],
                        flips[:, 2:3], 0.1)
    assert torch.allclose(together[2], alone[0], atol=1e-10)
    assert CNN.num_params == batched_cnn.NUM_PARAMS


# -- L1 and L2 ---------------------------------------------------------------
def test_dual_verdict_follows_thesis_table_4_2():
    policy = VerdictPolicy(theta_tol=0.05, theta_rare=-0.03, theta_corner_harm=0.0, theta_util=0.01)
    cases = [
        ((0.06, -0.50, True), FRAUD),     # main-task harm wins over any corner gain
        ((0.00, 0.01, False), FRAUD),     # main-safe but corner-harmful
        ((0.04, -0.03, True), RARITY),    # corner gain inside the main tolerance
        ((0.00, -0.001, True), NOISE),    # nominated, neutral, no measurable utility
        ((0.00, -0.001, False), HONEST),  # same drift, not nominated
        ((-0.02, -0.001, True), HONEST),  # nominated but useful
    ]
    for (dm, dc, nominated), expected in cases:
        assert dual_verdict([dm], [dc], [nominated], policy)[0] == expected
    # the thesis conflict probe: helps the corner, harms the main task
    probe = ([0.0647], [-0.176], [True])
    main_only = VerdictPolicy(0.05, -0.03, 0.0, 0.01, "main_only")
    corner_only = VerdictPolicy(0.05, -0.03, 0.0, 0.01, "corner_only")
    assert dual_verdict(*probe, policy)[0] == FRAUD
    assert dual_verdict(*probe, main_only)[0] == FRAUD
    assert dual_verdict(*probe, corner_only)[0] == RARITY
    assert dual_verdict([0.0], [0.5], [False], main_only)[0] == HONEST   # blind to corner harm


def test_l1_signals_and_routing():
    torch.manual_seed(0)
    direction = torch.randn(500)
    updates = direction[None] + 0.3 * torch.randn(12, 500)
    updates[0] = -updates[0]            # reversed
    updates[1] = 8.0 * updates[1]       # scaled
    signals = l1_signals(updates)
    assert signals.cosine_deviation[0] > 1.5 and signals.cosine_deviation[2:].max() < 0.5
    assert signals.norm_mad[1] > 5 * np.median(signals.norm_mad[2:])
    assert signals.sign_disagreement[0] > 0.9 and signals.sign_disagreement[2:].max() < 0.3

    policy = RoutingPolicy(tau_cos=0.6, tau_norm_mad=20.0, tau_sign=0.55, p_recheck=0.0)
    routed, nominated, reasons = route(signals, policy, np.random.default_rng(0))
    assert reasons[0] == "cosine_screening" and reasons[1] == "norm_mad_screening"
    assert routed[:2].all() and nominated[:2].all() and not routed[2:].any()
    everything, nominated, reasons = route(
        signals, RoutingPolicy(0.6, 20.0, 0.55, 0.0, exhaustive=True), np.random.default_rng(0))
    assert everything.all() and nominated.sum() == 2 and reasons[5] == "exhaustive"
    rechecked, nominated, _ = route(signals, RoutingPolicy(0.6, 20.0, 0.55, 1.0),
                                    np.random.default_rng(0))
    assert rechecked.all() and nominated.sum() == 2      # a recheck does not nominate


def test_geometric_median_resists_an_outlier():
    torch.manual_seed(0)
    points = torch.randn(11, 50) * 0.1
    points[0] += 100.0
    median, weights = geometric_median(points)
    assert median.norm() < 1.0 and points.mean(dim=0).norm() > 5.0
    assert abs(float(weights.sum()) - 1.0) < 1e-5 and weights[0] < weights[1:].min()


def test_prototype_readout_ignores_the_head():
    torch.manual_seed(0)
    labels = torch.arange(10).repeat_interleave(8)
    support = (torch.arange(80) % 8) < 4
    means = torch.randn(10, 16) * 3
    separable = means[labels] + 0.1 * torch.randn(80, 16)
    features = torch.stack([separable, torch.randn(80, 16)])       # a good and a useless embedding
    loss, correct = prototype_readout(features, labels, support, 10)
    assert correct[0].float().mean() == 1.0 and loss[0].mean() < loss[1].mean()
    assert loss.shape == (2, 40)


# -- aggregation rules -------------------------------------------------------
def test_fedval_scores_match_formula_6():
    torch.manual_seed(1)
    class_loss = torch.rand(6, 10) * 2 + 0.2
    mean_loss = class_loss.mean(dim=1)
    s1, s1_avg, c_const, s2 = 3.0, 5.0, 3.0, 3.0
    expected = []
    for d in range(6):
        score = 0.0
        for k in range(10):
            label_mean = class_loss[:, k].mean()
            mad = (class_loss[:, k] - label_mean).abs().mean()
            boost = max(1.0, float((label_mean / mean_loss.mean()) ** s2))
            score += boost * s1 * float(label_mean - class_loss[d, k]) / float(mad) + c_const * s1
        mad_avg = (mean_loss - mean_loss.mean()).abs().mean()
        score += s1_avg * float(mean_loss.mean() - mean_loss[d]) / float(mad_avg) + c_const * s1_avg
        expected.append(score)
    got = agg.fedval_scores(class_loss, mean_loss, s2)
    assert torch.allclose(got, torch.tensor(expected), atol=1e-3)
    # an average client gets the baseline C*s1*K + C*s1_avg
    assert abs(float(got.mean()) - (c_const * s1 * 10 + c_const * s1_avg)) < 1e-3
    assert agg.fedval_s2_candidates(3.0) == [0.0, 2.5, 3.0, 3.5, 8.0]
    weights = agg.fedval_weights(torch.tensor([-5.0, 1.0, 3.0]))
    assert weights[0] == 0 and abs(float(weights.sum()) - 1.0) < 1e-6


def test_baseline_rules_reject_what_they_should():
    torch.manual_seed(0)
    direction = torch.randn(200)
    updates = direction[None] + 0.2 * torch.randn(10, 200)
    updates[0] = -3.0 * updates[0]
    for rule in (lambda: agg.multi_krum(updates, 2), lambda: agg.fltrust(updates, direction)):
        delta, weight = rule()
        assert weight[0] == 0 and abs(float(weight.sum()) - 1.0) < 1e-5
        assert F.cosine_similarity(delta, direction, dim=0) > 0.9
    # Zeno drops the b lowest scores, Zeno++ everything with a non-positive score
    scores = torch.tensor([-1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, -0.2])
    _, weight = agg.zeno(updates, scores, byzantine=2)
    assert weight[0] == 0 and weight[9] == 0 and int((weight > 0).sum()) == 8
    _, weight = agg.zenopp(updates, scores)
    assert int((weight > 0).sum()) == 8
    delta, weight = agg.zenopp(updates, -torch.ones(10))
    assert float(delta.abs().sum()) == 0 and float(weight.sum()) == 0
    _, weight = agg.fedavg(updates)
    assert torch.allclose(weight, torch.full((10,), 0.1))


def test_auroc_and_rarity_discovery():
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    types = np.array(["rarity", "rarity", "honest", "corner_harm", "honest"], dtype=object)
    verdicts = np.array(["RARITY", "HONEST", "RARITY", "RARITY", "HONEST"], dtype=object)
    found = rarity_discovery(types, verdicts)
    assert found["recognition"] == 0.5 and abs(found["precision"] - 1 / 3) < 1e-9
    assert found["corner_harm_named_rarity"] == 1 and found["honest_named_rarity"] == 1


# -- engine ------------------------------------------------------------------
def test_every_method_runs_and_methods_of_a_seed_are_paired():
    rng = np.random.default_rng(0)
    y = _labels(400)
    means = rng.normal(size=(10, 16)) * 2.0
    x = torch.from_numpy(means[y] + rng.normal(size=(len(y), 16))).float()
    ann = annotate_golden(y, eval_per_class=20)
    fed_cfg = FederationConfig(num_clients=30, samples_per_client=32, n_rarity=3, n_corner_harm=2,
                               n_sign_flip=3, n_label_flip=3, n_noise=2)
    cfg = EngineConfig(rounds=3, warmup_rounds=1, clients_per_round=10, batch_size=16,
                       audit_per_class=10, byzantine_budget=2, local_lr=0.05)
    verdict = VerdictPolicy(theta_tol=0.2, theta_rare=-0.02, theta_corner_harm=0.2, theta_util=1e-6)
    routing = RoutingPolicy(0.9, 2.0, 0.45, 0.25)
    policies = {"loss": CornerDrivePolicy(routing, verdict, audit_step=0.5),
                "embedding": CornerDrivePolicy(routing, verdict, space="embedding")}
    ctx = SeedContext({"y": y}, ann, 11, cfg, fed_cfg, "cpu", x_device=x,
                      y_device=torch.from_numpy(y), backend=head_backend(16))
    runs = {method: run_unit(ctx, "mixed", method, policies) for method in METHODS}
    reference = runs["fedavg"]
    for method, result in runs.items():
        assert result["u_weight"].shape == (3, 10)
        assert np.all(result["u_weight"] >= 0)
        totals = result["u_weight"].sum(axis=1)
        assert np.all((np.abs(totals - 1.0) < 1e-4) | (totals == 0)), method
        # same clients, same roles in every method: the comparison is paired
        assert np.array_equal(result["u_client"], reference["u_client"])
        assert np.array_equal(result["u_role"], reference["u_role"])
        assert np.isfinite(result["r_acc_all"]).all()
    # all methods start from one checkpoint: the first round's updates and
    # their golden-set drifts are identical across methods
    assert np.allclose(runs["zeno"]["u_d_main"][0], reference["u_d_main"][0], atol=1e-5)
    assert np.allclose(runs["cornerdrive"]["u_d_corner"][0], reference["u_d_corner"][0], atol=1e-5)
    oracle = runs["oracle"]
    fraud = oracle["u_truth"] == 3
    assert np.all(oracle["u_weight"][fraud] == 0)
    exhaustive = runs["cd_exhaustive"]
    assert exhaustive["u_routed"].all() and (exhaustive["u_verdict"] != 4).all()
    assert (runs["fedavg"]["u_verdict"] == 4).all()          # baselines audit nothing
    no_attack = run_unit(ctx, "no_attack", "fedavg")
    assert set(np.unique(no_attack["u_truth"])) <= {0, 1}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
