# Revised Thesis Audit Against SAT301 PDF and Real-Gradient Results

> Superseded note: this audit predates the 2026-05-15 held-out real-gradient
> split fix. Use `docs/reports/REVIEWER_AUDIT_FIXES_2026-05-15.md` and the
> regenerated `artifacts/tables/table_5_1_real_gradient_macro.csv` for the
> current reviewer-facing real-gradient numbers.

Audited files:

- Revised thesis: user-supplied final thesis PDF outside the repo
- Previous SAT301 thesis: older same-name local PDF copy outside the repo
- Main result sources:
  - `results/real_gradient_full_method_comparison.csv`
  - `results/real_gradient_adaptive_method_comparison.csv`
  - `results/real_gradient_reliability_medium/real_gradient_reliability_summary.csv`

## Executive Assessment

The revised thesis correctly moves the work from a synthetic-only ALG evaluation
to a two-tier evaluation:

1. Real-gradient benchmark as the primary comparative evidence.
2. ALG benchmark as mechanism-isolation evidence.

The main experimental numbers in the revised PDF match the repository result
files. The thesis is substantially stronger than the SAT301 version because it
no longer leaves real-gradient validation entirely as future work. However,
there are still several issues to fix before submission, mainly around
experiment setup completeness and PDF layout.

## Data Consistency Check

### Table 5.2: Macro Real-Gradient Method Comparison

Status: consistent with `results/real_gradient_reliability_medium/real_gradient_reliability_summary.csv`.

| Method | Main acc | Corner acc | Fraud survival | Rarity retention | Check |
| --- | ---: | ---: | ---: | ---: | --- |
| Multi-Krum | 0.4891 | 0.6878 | 0.0533 | 0.8858 | matches CSV macro average |
| FLTrust | 0.4691 | 0.6489 | 0.1178 | 0.6308 | matches CSV macro average |
| Zeno | 0.4970 | 0.6824 | 0.1244 | 0.9788 | matches CSV macro average |
| Zeno++ | 0.4946 | 0.6936 | 0.0000 | 0.4865 | matches CSV macro average |
| CornerDrive | 0.4707 | 0.7050 | 0.0489 | 0.7789 | matches CSV macro average |

### Table 5.3: CornerDrive Reliability Results

Status: consistent with `results/real_gradient_reliability_medium/real_gradient_reliability_summary.csv`.

| Dataset | Fraud survival | Rarity retention | L1 review | Check |
| --- | ---: | ---: | ---: | --- |
| MNIST | 0.0267 +/- 0.0523 | 0.7131 +/- 0.0803 | 0.8083 +/- 0.0229 | matches CSV |
| FashionMNIST | 0.1000 +/- 0.0987 | 0.6568 +/- 0.0201 | 0.8183 +/- 0.0255 | matches CSV |
| LEAF/FEMNIST | 0.0200 +/- 0.0392 | 0.9667 +/- 0.0653 | 0.8267 +/- 0.0118 | matches CSV |

### Diagnostic Adaptation Claim

Status: consistent with local result files.

| Claim | Source | Check |
| --- | --- | --- |
| Default cosine-only CornerDrive fraud survival = 0.6250 | `real_gradient_full_method_comparison.csv` | matches |
| Adaptive L1V3 small benchmark fraud survival = 0.0521 | `real_gradient_adaptive_method_comparison.csv` | matches |
| Expanded reliability fraud survival = 0.0489 | `real_gradient_reliability_summary.csv` | matches |
| Expanded observations = 1,800; fraud = 450; rarity = 581 | `real_gradient_reliability_summary.csv` | matches |

## Major Improvements Over SAT301 Version

### 1. Abstract Is No Longer Synthetic-Only

Old SAT301 abstract said the evidence remained benchmark-level and real
client-SGD validation was still required. The revised abstract now reports:

- MNIST, FashionMNIST, and LEAF/FEMNIST real-gradient benchmark.
- Multi-Krum, FLTrust, Zeno, Zeno++, and CornerDrive as direct baselines.
- Default CornerDrive fraud survival 0.6250 on real gradients.
- Adaptive L1V3 CornerDrive macro fraud survival 0.0489.
- Rarity retention 0.7789 and highest macro corner accuracy.

This is a major improvement and aligns with the new experiment data.

### 2. Chapter 5 Is Correctly Reframed

Old SAT301 Chapter 5 was centred on ALG. Revised Chapter 5 now opens with the
real-gradient benchmark and keeps ALG as mechanism-isolation. This is the right
structure for the new evidence.

### 3. Future Work Is Updated Appropriately

The old "Real client-SGD validation" future-work item has been replaced by a
more precise limitation:

- real-gradient bridge completed;
- full IoV/BDD100K-scale validation remains future work;
- latency, adaptive attackers, and production readiness remain future work.

This is much more defensible.

## Findings To Fix

### P1. Table 5.1 Breaks A Sentence Across Pages

Location: revised PDF pages 26-27.

The sentence before Table 5.1 ends as:

> keeping fraud survival close to robust suppres-

Then Table 5.1 appears. After the table, the fragment continues:

> sion baselines.

This is a visible layout defect. It makes the thesis look less polished and can
interrupt reviewer reading flow.

Recommended fix:

- Move Table 5.1 after the paragraph, or force it to top/bottom placement.
- Rewrite the sentence to avoid hyphenation immediately before a float.
- Example: "CornerDrive keeps fraud survival close to robust suppression
  baselines while preserving substantially more rarity than Zeno++." Then place
  the table after the full sentence.

### P1. Real-Gradient Setup Table Is Still Too Thin For Reproducibility

Location: Table 5.1.

The table includes datasets, seeds, clients, rounds, observations, and methods,
but omits several settings that directly affect the reported results:

- max samples per client = 48;
- min samples per client = 8;
- local batch size = 16;
- pretrain steps = 50;
- attack fraction = 0.20;
- corner-harm fraction = 0.05;
- noise fraction = 0.05;
- rarity threshold = 30% corner-label fraction;
- policy profile = `real_data_adaptive`;
- `theta_tol = 0.02`;
- `theta_rare = -0.005`;
- `cosine_filter_threshold = 0.60`;
- `recheck_probability = 0.25`;
- L1 mode = `v3_m2_norm_sign_fixed`;
- norm MAD threshold = 2.5;
- sign threshold = 0.55.

The text later mentions some of these, but the setup table should be
self-contained. This matters because the thesis now makes the real-gradient
benchmark the primary comparative benchmark.

Recommended fix:

Add a second panel to Table 5.1:

| Setting | Value |
| --- | --- |
| Max samples per client | 48 |
| Min samples per client | 8 |
| Local batch size | 16 |
| Pretrain steps | 50 |
| Attack / corner-harm / noise fractions | 0.20 / 0.05 / 0.05 |
| Rarity definition | >= 30% corner-label samples |
| Policy profile | `real_data_adaptive` |
| L2 thresholds | `theta_tol = 0.02`, `theta_rare = -0.005` |
| L1 thresholds | cosine 0.60, norm MAD 2.5, sign 0.55, recheck 0.25 |

### P1. Abstract Mixes Small Diagnostic And Expanded Reliability Numbers

Location: Abstract.

The abstract says default CornerDrive reaches 0.6250 fraud survival on real
gradients, then says adaptive L1V3 reaches 0.0489. Both numbers are correct, but
they come from different real-gradient runs:

- 0.6250: small default-policy diagnostic benchmark.
- 0.0489: expanded multi-seed reliability benchmark macro average.

The distinction is clear in Chapter 5, but not in the abstract.

Recommended fix:

Change the abstract sentence to:

> A diagnostic real-gradient run shows that the ALG cosine-only L1 profile does
> not transfer directly: default CornerDrive reaches 0.6250 fraud survival. In
> the expanded three-seed reliability benchmark, adaptive L1V3 routing with
> cosine, norm, sign, and recheck signals reduces macro fraud survival to
> 0.0489, with 0.7789 rarity retention and the highest macro corner accuracy.

### P2. Table 5.3 Float Placement Creates A Mostly Empty Page

Location: revised PDF page 28.

Table 5.3 appears alone in the vertical middle of the page, leaving a large
amount of whitespace. This is not a correctness bug, but it weakens visual
polish.

Recommended fix:

- Use `[!htbp]`, `\FloatBarrier`, or move surrounding paragraph text so the
  table appears near its discussion.
- Alternatively combine Tables 5.2 and 5.3, or move Table 5.3 immediately after
  the first paragraph of Section 5.3.

### P2. Bold Formatting In Table 5.2 Is Ambiguous

Location: Table 5.2.

The table bolds Zeno++ fraud survival and CornerDrive corner accuracy. It does
not bold Zeno main accuracy or Zeno rarity retention, although those are the
best values in their columns. If bold means best-in-column, the formatting is
inconsistent. If bold means "selected thesis emphasis", the table needs a note.

Recommended fix:

Add a note:

> Bold marks the thesis-relevant trade-off endpoints: lowest fraud survival and
> highest corner accuracy.

Or bold every best value consistently.

### P2. Real-Gradient "Client Updates" Could Be Overread As Full Client-SGD

Location: Abstract and Section 5.2.

The revised thesis says the primary benchmark uses "client updates" from
MNIST/FashionMNIST/LEAF-FEMNIST. The benchmark derives real-data gradients from
a fixed model; it is not a full multi-epoch FL deployment.

The thesis does say "validation bridge rather than IoV deployment prototype",
which helps. Still, add one explicit sentence in Section 5.2:

> Each client update is a one-step gradient derived from a fixed model on the
> client's local examples; the experiment does not simulate full multi-epoch
> client training or end-to-end FL convergence.

This prevents an examiner from accusing the thesis of overstating deployment
realism.

### P2. Missing Explanation For Excluding FedAvg/GeoMed From Primary Real-Gradient Table

Old SAT301 Table 5.4 included FedAvg and GeoMed. The revised primary
real-gradient table includes Multi-Krum, FLTrust, Zeno, Zeno++, and
CornerDrive. This is reasonable, but the transition should be explained.

Recommended fix:

Add one sentence in Section 5.2:

> FedAvg and GeoMed remain in the ALG mechanism benchmark; the real-gradient
> table focuses on the strongest robust/trust/validation baselines most directly
> related to malicious-update filtering and server-side validation.

## Claims That Are Now Well Supported

The following claims are data-supported and safe to keep:

- The original cosine-only L1 profile does not transfer directly to real
  gradients.
- The dominant real-gradient failure mode is exposure: fraud can bypass L1 and
  avoid L2 review.
- Adaptive L1V3 reduces CornerDrive fraud survival substantially.
- CornerDrive has the highest macro corner accuracy in the expanded real-gradient
  comparison.
- Zeno++ achieves lower fraud survival but much lower rarity retention.
- Zeno retains more rarity but allows more fraud.
- The benchmark is a validation bridge, not full IoV deployment.

## Claims To Avoid

Avoid these stronger claims:

- "CornerDrive dominates all baselines." It does not: Zeno++ has lower fraud
  survival and Zeno has higher rarity retention.
- "Real-gradient validation proves deployment readiness." It does not: BDD100K,
  latency, adaptive attackers, and real vehicle partitions remain future work.
- "The benchmark uses real vehicular data." It does not yet. It uses real public
  image/federated data and LEAF/FEMNIST writer partitions.
- "L1V3 is low-cost." It reviews about 81-83% of updates in the expanded run.

## Comparison With SAT301 Version

| Area | SAT301 version | Revised version | Assessment |
| --- | --- | --- | --- |
| Main experiment | ALG synthetic benchmark | Real-gradient benchmark + ALG mechanism benchmark | Strong improvement |
| Baselines | FedAvg, GeoMed, Multi-Krum, CornerDrive | Multi-Krum, FLTrust, Zeno, Zeno++, CornerDrive for real gradients; ALG still has original baselines | Stronger literature alignment |
| Real data | Future work | MNIST, FashionMNIST, LEAF/FEMNIST validation bridge | Strong improvement |
| L1 policy | cosine + p-recheck | cosine-only for ALG; L1V3 norm/sign/recheck for real gradients | Data-driven improvement |
| Main limitation | real client-SGD missing | full IoV/BDD100K, latency, adaptive attackers missing | More precise |
| Risk | synthetic-only external validity | high review-rate cost and non-vehicular real datasets | More defensible but still limited |

## Recommended Minimal Revision Before Submission

1. Fix the page 26-27 float/sentence break around Table 5.1.
2. Expand Table 5.1 with the omitted setup and policy parameters.
3. Clarify in the abstract that 0.6250 is from the diagnostic run and 0.0489
   from the expanded multi-seed reliability run.
4. Add one sentence explaining real-data gradients are one-step fixed-model
   client gradients, not full client-SGD deployment.
5. Clarify Table 5.2 bolding.
6. Improve Table 5.3 float placement if time permits.

If these are fixed, the revised thesis will be much more coherent than the
SAT301 version and substantially better supported by the current experiment
data.
