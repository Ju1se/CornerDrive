# Thesis Revision Plan After Real-Gradient Benchmark

This note maps the current dissertation PDF to the new real-gradient benchmark
evidence in the repository. It focuses on the experiment data and experiment
setup changes needed to make the thesis data-driven rather than synthetic-only.

## Current Thesis Position

The current PDF frames the evaluation as a controlled ALG synthetic benchmark:

- Chapter 5 states that "full client-side federated optimisation is outside the
  current benchmark".
- Table 5.1 lists only the ALG setup.
- Section 5.2 compares FedAvg, GeoMed, Multi-Krum, and CornerDrive, with
  Zeno-style validation only as an audit-signal ablation.
- Section 5.11 and Chapter 6 explicitly say real client-SGD validation remains
  future work.

This was correct for the submitted PDF, but it is no longer complete. The repo
now contains real-gradient calibration and benchmark evidence from MNIST,
FashionMNIST, and LEAF/FEMNIST, with Multi-Krum, FLTrust, Zeno, Zeno++, and
CornerDrive compared on the same round schedule.

## New Evidence To Add

### Completed Small Real-Gradient Benchmark

Source files:

- `results/real_gradient_full_method_comparison.csv`
- `results/real_gradient_adaptive_method_comparison.csv`

Default cosine-only CornerDrive performed poorly on real gradients because many
fraud updates bypassed L1. After real-data adaptation, CornerDrive improved as
follows across MNIST, FashionMNIST, and LEAF/FEMNIST:

| CornerDrive profile | Main acc | Corner acc | Fraud survival | Rarity retention | Selected updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default V2.5 cosine-only | 0.4475 | 0.4723 | 0.6250 | 0.8545 | 11.83 |
| Real-data adaptive L1V3 | 0.4783 | 0.5918 | 0.0521 | 0.8402 | 7.29 |

Interpretation:

- The original `p_recheck=0.10` synthetic setting does not transfer directly to
  real non-IID gradients.
- The failure is mainly L1 exposure, not L2 verdict quality.
- Real gradients require multi-signal L1 routing using cosine, norm, sign, and
  recheck evidence.

### Expanded Multi-Seed Reliability Benchmark

Source files:

- `results/real_gradient_reliability_medium/real_gradient_reliability_runs.csv`
- `results/real_gradient_reliability_medium/real_gradient_reliability_summary.csv`
- `results/real_gradient_reliability_medium/real_gradient_reliability_summary.json`

Completed expanded setup:

| Setting | Value |
| --- | --- |
| Datasets | MNIST, FashionMNIST, LEAF/FEMNIST |
| Seeds | 20260507, 20260508, 20260509 |
| Clients per dataset | 120 |
| Max samples per client | 48 |
| Clients per round | 20 |
| Rounds per seed | 10 |
| Observations per dataset | 600 client-round observations |
| Total observations | 1,800 client-round observations |
| Total fraud observations | 450 |
| Total rarity observations | 581 |
| Policy profile | `real_data_adaptive` |
| L1 mode | `v3_m2_norm_sign_fixed` |

CornerDrive per-dataset results:

| Dataset | Main acc | Corner acc | Fraud survival | Rarity retention | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 0.7214 +/- 0.0237 | 0.8607 +/- 0.0371 | 0.0267 +/- 0.0523 | 0.7131 +/- 0.0803 | 0.8083 +/- 0.0229 |
| FashionMNIST | 0.6249 +/- 0.0119 | 0.9189 +/- 0.0073 | 0.1000 +/- 0.0987 | 0.6568 +/- 0.0201 | 0.8183 +/- 0.0255 |
| LEAF/FEMNIST | 0.0659 +/- 0.0090 | 0.3352 +/- 0.0172 | 0.0200 +/- 0.0392 | 0.9667 +/- 0.0653 | 0.8267 +/- 0.0118 |

Macro average across the three datasets:

| Method | Main acc | Corner acc | Fraud survival | Rarity retention |
| --- | ---: | ---: | ---: | ---: |
| Multi-Krum | 0.4891 | 0.6878 | 0.0533 | 0.8858 |
| FLTrust | 0.4691 | 0.6489 | 0.1178 | 0.6308 |
| Zeno | 0.4970 | 0.6824 | 0.1244 | 0.9788 |
| Zeno++ | 0.4946 | 0.6936 | 0.0000 | 0.4865 |
| CornerDrive | 0.4707 | 0.7050 | 0.0489 | 0.7789 |

Interpretation:

- CornerDrive no longer has the lowest fraud survival in every metric:
  Zeno++ reaches 0.0000 macro fraud survival but retains only 0.4865 rarity.
- CornerDrive offers a stronger balance: low fraud survival, highest macro
  corner accuracy, and substantially higher rarity retention than Zeno++.
- Multi-Krum is competitive on fraud and rarity retention, but it does not
  provide the explicit Fraud/Rarity/Honest/Noise verdict semantics that the
  thesis claims as the main contribution.
- Zeno has very high rarity retention but higher fraud survival, supporting the
  thesis argument that single-objective validation is insufficient for the
  dual-objective discrimination problem.

## Required Thesis Changes

### Abstract

Current abstract says real client-SGD validation is still required. Update it
to say that the thesis now includes a real-gradient validation bridge, while
full IoV deployment remains future work.

Suggested replacement emphasis:

> In addition to the controlled ALG benchmark, a real-gradient validation study
> derives client updates from MNIST, FashionMNIST, and LEAF/FEMNIST. The
> expanded real-gradient benchmark covers 1,800 client-round observations across
> three seeds, including 450 fraud and 581 rarity observations. It shows that
> the original cosine-only L1 policy does not transfer directly to real non-IID
> gradients, but an adaptive L1V3 norm/sign/recheck profile reduces
> CornerDrive fraud survival to 0.0489 while retaining 0.7789 rarity and
> achieving the highest macro corner accuracy among the compared real-gradient
> methods.

Keep a limitation sentence:

> This remains benchmark-level evidence; full vehicular deployment with real
> vehicle identities, wall-clock latency, and adaptive attackers remains future
> work.

### Chapter 1: Contributions

Add a new contribution after the ALG benchmark contribution:

> A real-gradient reliability benchmark is added to test whether the
> discrimination rule survives outside synthetic archetype geometry. The
> benchmark derives gradients from public real datasets, compares Multi-Krum,
> FLTrust, Zeno, Zeno++, and CornerDrive, and reports multi-seed confidence
> intervals over 1,800 client-round observations.

Also soften any wording that says the evaluation is "only" synthetic. The
better framing is:

- ALG remains the primary mechanism-isolation benchmark.
- Real-gradient reliability is the external-validity bridge.
- Full IoV deployment remains out of scope.

### Chapter 2: Literature Review

Update Section 2.3.1 and Section 2.5:

- Zeno and Zeno++ should no longer be described only as "signal-level" reference
  methods. They are now included as real-gradient benchmark baselines.
- FLTrust should be described as both a literature comparison and an implemented
  baseline in the real-gradient benchmark.

Suggested sentence:

> In the real-gradient benchmark, FLTrust, Zeno, and Zeno++ are implemented as
> direct baselines. This makes the comparison stronger than the original ALG
> ablation, where Zeno-style scoring was used only as a single-objective audit
> signal.

### Chapter 4: Methodology

Update Sections 4.3 and 4.4.

Current method describes L1 mainly as cosine deviation plus probabilistic
recheck. Keep that as V2.5/ALG mode, but add a real-data adaptive L1V3 mode:

| L1 profile | Signals | Purpose |
| --- | --- | --- |
| V2.5 cosine-only | geometric median cosine deviation + probabilistic recheck | Controlled ALG mechanism benchmark |
| Real-data adaptive L1V3 | cosine deviation, norm MAD, sign disagreement, probabilistic recheck | Real non-IID gradient routing |

Add the data-driven reason:

> Real-gradient diagnostics show that many fraud updates, especially
> sign-flip-proxy updates, can remain below the cosine-only threshold and enter
> aggregation without L2 review. Therefore, the real-gradient profile routes
> updates using norm and sign evidence in addition to cosine deviation.

Policy parameters to document:

| Parameter | ALG setting | Real-gradient adaptive setting |
| --- | ---: | ---: |
| `theta_tol` | 0.05 | 0.02 |
| `theta_rare` | -0.03 | -0.005 |
| `cosine_filter_threshold` | 0.70 | 0.60 |
| `recheck_probability` | 0.10 in selected ALG row | 0.25 |
| `cornerdrive_l1_mode` | `v25_cosine_fixed` | `v3_m2_norm_sign_fixed` |
| `norm_mad_threshold` | not used | 2.5 |
| `sign_threshold` | not used | 0.55 |

### Chapter 5: Structure

Chapter 5 should be reorganised into two experimental blocks.

Recommended structure:

1. `5.1 Experimental Design`
   - Explain two-tier evaluation:
     - ALG for internal mechanism control.
     - Real-gradient reliability benchmark for external validity.
2. `5.1.1 Synthetic ALG Protocol`
   - Keep most of current content.
3. `5.1.2 Real-Gradient Reliability Protocol`
   - New subsection.
4. `5.2 Compared Methods and Metrics`
   - Add FLTrust, Zeno, Zeno++.
5. Existing ALG results sections.
6. New `5.x Real-Gradient Benchmark Results`.
7. New `5.x Data-Driven Policy Adaptation`.
8. Discussion updated to compare synthetic and real evidence.

### New Table: Real-Gradient Dataset and Setup

Insert after Table 5.1 or as Table 5.2 before the synthetic archetype table.

| Setting | Value |
| --- | --- |
| Data sources | MNIST, FashionMNIST, LEAF/FEMNIST |
| Client construction | Torchvision non-IID two-label shards for MNIST/FashionMNIST; real writer/client partitions for LEAF/FEMNIST |
| Clients per dataset | 120 |
| Max samples per client | 48 |
| Clients per round | 20 |
| Rounds per seed | 10 |
| Seeds | 20260507, 20260508, 20260509 |
| Compared methods | Multi-Krum, FLTrust, Zeno, Zeno++, CornerDrive |
| Fraud families | sign-flip proxy, corner-harm |
| Rarity definition | client is rarity-heavy if >= 30% samples belong to corner labels |
| Total observations | 1,800 client-round observations |
| Fraud observations | 450 |
| Rarity observations | 581 |

### New Table: Real-Gradient Reliability Results

Use this table in the new real-gradient results subsection:

| Dataset | CornerDrive main acc | Corner acc | Fraud survival | Rarity retention | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 0.7214 +/- 0.0237 | 0.8607 +/- 0.0371 | 0.0267 +/- 0.0523 | 0.7131 +/- 0.0803 | 0.8083 +/- 0.0229 |
| FashionMNIST | 0.6249 +/- 0.0119 | 0.9189 +/- 0.0073 | 0.1000 +/- 0.0987 | 0.6568 +/- 0.0201 | 0.8183 +/- 0.0255 |
| LEAF/FEMNIST | 0.0659 +/- 0.0090 | 0.3352 +/- 0.0172 | 0.0200 +/- 0.0392 | 0.9667 +/- 0.0653 | 0.8267 +/- 0.0118 |

### New Table: Real-Gradient Method Comparison

Use macro averages across the three datasets:

| Method | Main acc | Corner acc | Fraud survival | Rarity retention |
| --- | ---: | ---: | ---: | ---: |
| Multi-Krum | 0.4891 | 0.6878 | 0.0533 | 0.8858 |
| FLTrust | 0.4691 | 0.6489 | 0.1178 | 0.6308 |
| Zeno | 0.4970 | 0.6824 | 0.1244 | 0.9788 |
| Zeno++ | 0.4946 | 0.6936 | 0.0000 | 0.4865 |
| CornerDrive | 0.4707 | 0.7050 | 0.0489 | 0.7789 |

Suggested interpretation:

> The real-gradient benchmark changes the conclusion from "CornerDrive simply
> dominates robust suppression" to a more nuanced trade-off. Zeno++ eliminates
> fraud most aggressively but keeps much less rarity. Zeno keeps rarity but
> allows more fraud. CornerDrive provides the best corner accuracy and a middle
> path between suppression and rarity preservation, while also producing
> explicit update-level verdicts.

### New Subsection: Data-Driven Policy Adaptation

This subsection should explain why the real-data policy changed.

Core evidence:

- Default real-gradient CornerDrive had fraud survival 0.6250.
- Adaptive L1V3 reduced it to 0.0521 in the small benchmark.
- In the expanded reliability benchmark, CornerDrive fraud survival is 0.0489
  macro average across three datasets.

Suggested text:

> The real-gradient diagnostic run revealed that the ALG-selected cosine-only
> L1 policy did not transfer directly. With the default V2.5 profile,
> CornerDrive reached 0.6250 fraud survival on the real-gradient benchmark. The
> reason was exposure: many fraud updates were not routed to L2 and therefore
> entered aggregation as clean updates. The adaptive profile lowers the cosine
> threshold, increases probabilistic recheck, and enables norm/sign L1V3
> routing. This reduces fraud survival to 0.0521 in the small real-gradient
> benchmark and 0.0489 in the expanded multi-seed benchmark, at an L1 review
> cost of about 0.8178.

### Chapter 5 Discussion

Current discussion says:

> real client-SGD deployment and adaptive attacks remain future work.

Replace with:

> Real-gradient validation partly closes the external-validity gap, but does not
> eliminate it. The new benchmark uses real public client data to derive
> gradients, and LEAF/FEMNIST preserves real writer partitions, but MNIST and
> FashionMNIST use deterministic non-IID shards and BDD100K image data is not yet
> fully included. Therefore, the evidence supports transfer from synthetic ALG
> to real-gradient calibration, not full IoV deployment.

Add a caveat:

> The adaptive profile improves fraud suppression by raising L2 review coverage
> to about 80-83%. This is acceptable for an evidence benchmark, but a deployed
> system must calibrate the review rate against latency and compute budget.

### Chapter 6 Conclusion

The conclusion should no longer end with "real client-SGD validation is still
required" as the main limitation. It should say:

- Synthetic ALG established the mechanism under controlled labels.
- Real-gradient benchmark tested the same rule on real data-derived updates.
- The original cosine-only policy failed on real data, revealing an exposure
  bottleneck.
- The adaptive L1V3 profile improved fraud suppression while preserving
  rarity.
- Full IoV deployment, BDD100K-scale validation, latency, and adaptive attackers
  remain future work.

Suggested paragraph:

> The added real-gradient benchmark shows that the synthetic conclusion
> transfers only after calibration. The original cosine-only L1 profile allowed
> high fraud survival on real non-IID gradients, demonstrating that ALG
> visibility assumptions were too favourable. After enabling adaptive L1V3
> routing and stricter L2 thresholds, CornerDrive achieved 0.0489 macro fraud
> survival over 1,800 real-gradient client-round observations, while retaining
> 0.7789 rarity and producing the highest macro corner accuracy among the
> compared real-gradient methods.

### Chapter 6 Future Work

Move "Real client-SGD validation" from future work into completed work, but keep
these future items:

1. Full IoV/BDD100K validation with actual driving images and pseudo/real vehicle
   partitions.
2. Larger seed count and longer training horizon.
3. Wall-clock L2 audit cost and GPU/CPU forward-pass budget.
4. Adaptive attackers that know the L1V3 routing profile.
5. Dynamic corner taxonomy and larger corner proxy calibration.

Suggested future-work replacement:

> The present real-gradient benchmark is a validation bridge, not a deployment
> prototype. Future work should extend it to BDD100K-scale driving images,
> real-time object-detection models, longer FL horizons, and adaptive attackers
> that shape gradients against both L1V3 routing and L2 dual-loss thresholds.

## Claims To Avoid After The New Results

Do not claim:

- CornerDrive always beats Multi-Krum on fraud survival.
- CornerDrive always retains more rarity than Zeno.
- The real-gradient benchmark proves IoV deployment readiness.
- L1V3 is free. It costs about 0.8178 review coverage in the expanded run.

Stronger, data-supported claim:

> CornerDrive provides a data-driven trade-off: it maintains low fraud survival
> close to robust baselines, improves macro corner accuracy, preserves
> substantially more rarity than Zeno++, and supplies explicit update-level
> verdicts that robust aggregation alone does not provide.

## Suggested New Result Files To Cite In Appendix

- `results/real_gradient_adaptive_method_comparison.csv`
- `results/real_gradient_reliability_medium/real_gradient_reliability_runs.csv`
- `results/real_gradient_reliability_medium/real_gradient_reliability_summary.csv`
- `results/real_gradient_reliability_medium/real_gradient_reliability_summary.json`

The appendix can also include the command:

```bash
python scripts/export_real_gradient_reliability_benchmark.py \
  --sources mnist,fashionmnist,femnist \
  --seeds 20260507,20260508,20260509 \
  --max-clients 120 \
  --max-samples-per-client 48 \
  --clients-per-round 20 \
  --rounds 10 \
  --pretrain-steps 50 \
  --output-dir results/real_gradient_reliability_medium
```
