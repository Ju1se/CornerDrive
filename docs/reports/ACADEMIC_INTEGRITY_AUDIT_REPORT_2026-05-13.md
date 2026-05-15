# CornerDrive Thesis/Project Academic Integrity Audit Report

> Superseded note: the real-gradient numbers in this 2026-05-13 audit used the
> earlier same-surface benchmark. The reviewer-facing held-out split fix and
> updated real-gradient table are documented in
> `docs/reports/REVIEWER_AUDIT_FIXES_2026-05-15.md`.

审计日期：2026-05-13
审计对象：

- Thesis PDF：user-supplied final dissertation PDF outside the repo
- 项目目录：repository root

## 1. 总结结论

本次审计没有发现论文核心实验表格存在“确认的数据造假”证据。Downloads 版 thesis 中的主要实验 claim 可以从本地结果文件或重新运行脚本得到支持：

- real-gradient benchmark 的宏平均表、数据集分表、旧版 cosine-only 对比和 adaptive profile 对比，均能从 `results/real_gradient_*` CSV 回算。
- ALG/V2.5 主实验、recheck sweep、L1 routing、L2 confusion、stress tests、corner-family divergence、corner-harm threshold calibration，均可重新导出并与 thesis 表格一致或仅有 0.01 的四舍五入差异。
- 当前核心决策路径未发现把 `ground_truth_label` 直接用于 L1/L2 verdict 的循环证据；标签主要用于指标统计、混淆矩阵和报告。

审计时发现一个高优先级可复现性风险：原 README/documentation 给出的 V2.5 导出命令没有显式设置 `BATCH_SIZE=24 VEHICLE_POOL_SIZE=128`。在当前代码默认值下，脚本会跑成 `BATCH_SIZE=96`、`VEHICLE_POOL_SIZE=384` 的另一组实验规模，不能复现 thesis 的 ALG 数字。这个问题不是数据造假的证据；repo hygiene pass 已通过 README 和 `scripts/reproduce_all.sh` 显式固定 thesis scale。

## 2. 审计范围和证据来源

### 2.1 PDF 版本确认

发现两个同名 PDF，哈希不同。用户指定的 Downloads 版本是本次审计依据：

| 文件 | SHA256 |
|---|---|
| user-supplied final dissertation PDF | `ae51cb3da53db54863dda37a0d2ec7f7fc75b00e80821ad8f217894f69c1c9cb` |
| older same-name local PDF copy | `bc4edd7c57cb32416e7270cc8ad55c34de5398cdffd48c0427aaf441beecb2da` |

Downloads PDF metadata：58 pages，created/modified `2026-05-13 11:17:36 CST`。

### 2.2 关键结果文件哈希

| 文件 | SHA256 |
|---|---|
| `results/real_gradient_reliability_medium/real_gradient_reliability_summary.csv` | `260a3e487d0e22d39af037d0d9b34f7fcce3ad67be03edbeb85e50f75b83368c` |
| `results/real_gradient_reliability_medium/real_gradient_reliability_runs.csv` | `f6d7612a044cb4ae876db8ef8eecddf1455f92cb6357f931948523b5db6ffc93` |
| `results/real_gradient_full_method_comparison.csv` | `32b1abd2e4bb7c0c7a87993ed493187b5fdb1403f6c6de7184fda5c27b33bb87` |
| `results/real_gradient_adaptive_method_comparison.csv` | `386a3b2e77af88c12ca7877f76beeef19026c9a99fc68a3fba53de69bdf4f77b` |
| `results/audit_reproduction/v25_artifacts_b24/v25_main_result_table.csv` | `fb8571d709a96e18a915323a66b517e64be3bd1d235719b8ad42e5f174da14d9` |
| `results/audit_reproduction/v25_artifacts_b24/v25_recheck_sweep_table.csv` | `c7b469df72fd24b218a0b8df35c6b113493caa48f43e28ae964165491d49e4e9` |
| `results/audit_reproduction/v25_stress_tests_b24/stress_rarity_overlap_summary.csv` | `b41242629dbeb8aaab050657b282b231c1c11bacbfc843ac25115178c77d7846` |
| `results/audit_reproduction/v25_stress_tests_b24/stress_proxy_sensitivity_summary.csv` | `7d026a8edd8cfbf0806ba3e93ca35e861e626237005e4579862b32ab1b84fef4` |
| `results/audit_reproduction/corner_family_divergence_b24/corner_family_divergence_summary.csv` | `415a4c2db6b3f91f384a9e2da5e6205044c2e5e0cb37dbcf8cd466dca785c1d1` |
| `results/audit_reproduction/corner_harm_threshold_calibration_b24/corner_harm_threshold_calibration_summary.csv` | `1d62a6b56f881ee0243cad6b70886d55a25dafa722e6911fb50e9ecb16567ca9` |

## 3. 复现实验命令

### 3.1 ALG/V2.5 主结果，thesis-matching 命令

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 ./.venv/bin/python scripts/export_v25_artifacts.py \
  --seeds 20260318,20260319,20260320,20260321,20260322 \
  --recheck-values 0.00,0.05,0.10,0.20,0.30 \
  --output-dir results/audit_reproduction/v25_artifacts_b24
```

输出配置确认：

- `clients_per_round=24`
- `clients_total=128`
- seeds：`20260318` 至 `20260322`
- ground-truth totals：`RARITY=410`、`HONEST=2000`、`NOISE=220`、`FRAUD=250`

### 3.2 stress tests

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 ./.venv/bin/python scripts/export_v25_stress_tests.py \
  --seeds 20260318,20260319,20260320,20260321,20260322 \
  --threshold-seeds 20260318,20260319,20260320,20260321,20260322 \
  --output-dir results/audit_reproduction/v25_stress_tests_b24
```

### 3.3 corner-family divergence

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 ./.venv/bin/python scripts/export_corner_family_divergence.py \
  --seeds 20260318,20260319,20260320,20260321,20260322 \
  --output-dir results/audit_reproduction/corner_family_divergence_b24
```

### 3.4 corner-harm threshold calibration

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 ./.venv/bin/python scripts/export_corner_harm_threshold_calibration.py \
  --skip-oracle-drift \
  --output-dir results/audit_reproduction/corner_harm_threshold_calibration_b24
```

### 3.5 相关测试

```bash
./.venv/bin/python -m pytest \
  backend/tests/test_baseline_analysis.py \
  backend/tests/test_l1v3_router.py \
  backend/tests/test_real_gradient_bdd100k.py \
  -q
```

结果：`10 passed, 1 warning in 13.02s`。

## 4. Real-gradient claim 审计

### 4.1 宏平均结果

从 `real_gradient_reliability_summary.csv` 按 MNIST、FashionMNIST、FEMNIST 三个 source 对每个 method 取宏平均，复现 thesis Table 5.1：

| Method | Main acc. | Corner acc. | Fraud survival | Rarity retention |
|---|---:|---:|---:|---:|
| Multi-Krum | 0.4891 | 0.6878 | 0.0533 | 0.8858 |
| FLTrust | 0.4691 | 0.6489 | 0.1178 | 0.6308 |
| Zeno | 0.4970 | 0.6824 | 0.1244 | 0.9788 |
| Zeno++ | 0.4946 | 0.6936 | 0.0000 | 0.4865 |
| CornerDrive | 0.4707 | 0.7050 | 0.0489 | 0.7789 |

结论：thesis 中 “CornerDrive adaptive multi-signal L1 reduces macro fraud survival to 0.0489 and keeps 0.7789 rarity retention” 的 claim 可由本地 CSV 回算支持。

### 4.2 CornerDrive 数据集分表

| Dataset | Main acc. | Corner acc. | Fraud survival | Rarity retention | L1 review |
|---|---:|---:|---:|---:|---:|
| MNIST | 0.7214 ± 0.0237 | 0.8607 ± 0.0371 | 0.0267 ± 0.0523 | 0.7131 ± 0.0803 | 0.8083 ± 0.0229 |
| FashionMNIST | 0.6249 ± 0.0119 | 0.9189 ± 0.0073 | 0.1000 ± 0.0987 | 0.6568 ± 0.0201 | 0.8183 ± 0.0255 |
| FEMNIST | 0.0659 ± 0.0090 | 0.3352 ± 0.0172 | 0.0200 ± 0.0392 | 0.9667 ± 0.0653 | 0.8267 ± 0.0118 |

样本规模核对：

- client observations：1800
- fraud observations：450
- rarity observations：581

### 4.3 旧版 diagnostic claim

旧版 cosine-only CornerDrive：

- `main_acc=0.4475`
- `corner_acc=0.4723`
- `fraud_survival=0.6250`
- `rarity_retention=0.8545`

adaptive profile：

- `main_acc=0.4783`
- `corner_acc=0.5918`
- `fraud_survival=0.0521`
- `rarity_retention=0.8402`

结论：thesis 中 “cosine-only 0.6250 fraud survival, adaptive profile sharply reduces survival” 的方向和数字均有本地 CSV 支撑。

### 4.4 决策路径检查

关键代码路径：

- `_build_attack_plan` 使用随机打乱后的 index 分配 `sign_flip_proxy`、`corner_harm`、`benign_noise`，见 `backend/policy_agent/analysis/real_gradient_benchmark.py:968`。
- `_round_truth` 生成 truth labels 供指标统计使用，见 `backend/policy_agent/analysis/real_gradient_benchmark.py:995`。
- baseline 方法使用 Krum/FLTrust/Zeno/Zeno++ 的梯度距离、trust score 或 validation score 决策，见 `backend/policy_agent/analysis/real_gradient_benchmark.py:1197` 至 `1259`。
- CornerDrive 使用 `filter_suspects` 做 L1，再用 `DualChannelAuditor.audit` 做 L2，见 `backend/policy_agent/analysis/real_gradient_benchmark.py:1261` 至 `1288`。
- `selected_truth`、fraud survival、rarity retention 在 aggregation 后用于指标统计，见 `backend/policy_agent/analysis/real_gradient_benchmark.py:1297` 至 `1310`。

审计判断：没有发现 real-gradient benchmark 把 truth label 直接用于 L1/L2 verdict 或 baseline accept/reject 的证据。truth label 用于指标统计，这是正常实验评价方式。

## 5. ALG/V2.5 claim 审计

### 5.1 主结果表

重新导出的 `v25_main_result_table.csv` 与 thesis 主要 claim 对齐：

| Method | Main acc. | Corner acc. | Rarity recall | Sign-flip survival | Corner-harm survival |
|---|---:|---:|---:|---:|---:|
| FedAvg | 75.43 ± 0.22 | 51.84 ± 0.11 | n/a | 100.00 ± 0.00 | 100.00 ± 0.00 |
| GeoMed | 84.21 ± 0.01 | 36.66 ± 0.05 | n/a | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Multi-Krum | 84.27 ± 0.03 | 36.47 ± 0.04 | n/a | 0.00 ± 0.00 | 0.00 ± 0.00 |
| CornerDrive p=0.00 | 86.62 ± 0.29 | 60.39 ± 0.25 | 100.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 |
| CornerDrive p=0.05 | 86.19 ± 0.35 | 60.77 ± 0.43 | 100.00 ± 0.00 | 0.00 ± 0.00 | 93.00 ± 2.74 |
| CornerDrive p=0.10 | 85.58 ± 0.55 | 61.24 ± 0.53 | 100.00 ± 0.00 | 0.00 ± 0.00 | 84.00 ± 5.48 |
| CornerDrive p=0.20 | 85.36 ± 0.46 | 61.43 ± 0.47 | 100.00 ± 0.00 | 0.00 ± 0.00 | 81.00 ± 6.52 |
| CornerDrive p=0.30 | 84.57 ± 0.22 | 62.05 ± 0.37 | 100.00 ± 0.00 | 0.00 ± 0.00 | 71.00 ± 4.18 |

差异说明：Multi-Krum corner accuracy 和 p=0.05 CornerDrive corner accuracy 与 thesis 个别地方可能出现 `36.48`/`60.78`，新导出表为 `36.47`/`60.77`，属于 0.01 级四舍五入显示差异。

### 5.2 L1 routing claim

聚合五个 seeds 后的 L1 routing：

| Setting | Archetype | Total | Cosine route | Recheck route | Bypassed |
|---|---|---:|---:|---:|---:|
| p=0.00 | HONEST | 2000 | 0 | 0 | 2000 |
| p=0.00 | RARITY | 410 | 410 | 0 | 0 |
| p=0.00 | NOISE | 220 | 220 | 0 | 0 |
| p=0.00 | FRAUD sign-flip | 150 | 150 | 0 | 0 |
| p=0.00 | FRAUD corner-harm | 100 | 0 | 0 | 100 |
| p=0.10 | HONEST | 2000 | 0 | 176 | 1824 |
| p=0.10 | RARITY | 410 | 410 | 0 | 0 |
| p=0.10 | NOISE | 220 | 220 | 0 | 0 |
| p=0.10 | FRAUD sign-flip | 150 | 150 | 0 | 0 |
| p=0.10 | FRAUD corner-harm | 100 | 0 | 16 | 84 |

结论：thesis 中 “corner-harm survival is due to L1 routing visibility, not L2 inability” 的 claim 被 routing 表支持。被 L1 送入 L2 的 16 个 corner-harm 全部被判 FRAUD；仍 survival 的 84 个是 bypassed。

### 5.3 L2/update confusion at p=0.10

| Ground truth | Audited in L2 | Fraud | Rarity | HonestSafe | Noise |
|---|---:|---:|---:|---:|---:|
| HONEST | 176 | 0 | 0 | 2000 | 0 |
| RARITY | 410 | 0 | 410 | 0 | 0 |
| NOISE | 220 | 0 | 0 | 100 | 120 |
| FRAUD sign-flip | 150 | 150 | 0 | 0 | 0 |
| FRAUD corner-harm | 16 | 100 | 0 | 0 | 0 |

注：表中 `FRAUD corner-harm=100` 是 end-to-end confusion 口径，包含未审计但仍归为 Fraud truth family 的总数；`Audited in L2=16` 说明其中 16 个真的进入 L2。

### 5.4 audit/oracle consistency

新导出 `v25_audit_oracle_consistency.csv`：

| Setting | Signal | Audited updates | Sign agree | Spearman |
|---|---|---:|---:|---:|
| p=0.00 | main drift | 780 | 1.0000 | 0.9999 |
| p=0.00 | corner drift | 780 | 1.0000 | 0.9997 |
| p=0.10 | main drift | 972 | 1.0000 | 0.9999 |
| p=0.10 | corner drift | 972 | 1.0000 | 0.9996 |
| p=0.30 | main drift | 1404 | 1.0000 | 0.9999 |
| p=0.30 | corner drift | 1404 | 1.0000 | 0.9993 |

结论：thesis 中 “proxy audit is highly consistent with oracle drift” 的 claim 有重新导出支撑。

### 5.5 attack energy validation

重新导出的 mean drift：

- sign-flip proxy mean Δmain：约 `0.0759`
- corner-harm mean Δcorner：约 `0.0547`

两类 attack 的 pass rate 均为 `1.0`。这支持 thesis 中 “attack signal is not vanishing/no-op” 的描述。

## 6. Stress-test claim 审计

### 6.1 Rarity-overlap stress

| Setting | Rarity recog. | Retention | False rarity | Corner acc. |
|---|---:|---:|---:|---:|
| Baseline outlier | 100.00 | 100.00 | 0 | 61.24 |
| Mixed overlap | 53.41 | 100.00 | 0 | 63.29 |
| Hard mixed | 40.49 | 100.00 | 0 | 57.33 |

与 thesis Table A.7 对齐。

### 6.2 Corner-proxy sensitivity

| Proxy | Rarity recog. | Retention | Corner acc. | Spearman | False rarity |
|---|---:|---:|---:|---:|---:|
| Default proxy | 100.00 | 100.00 | 61.24 | 0.9995 | 0 |
| Small proxy (50) | 100.00 | 100.00 | 61.24 | 0.9994 | 0 |
| Mild label bias | 100.00 | 100.00 | 61.24 | 0.9964 | 0 |
| Random main proxy | 0.00 | 0.00 | 20.15 | 0.2995 | 16 |

与 thesis Table A.8 对齐。random main proxy 失败边界也被复现；它不是隐藏失败，而是被 thesis 明确列出。

### 6.3 Threshold sensitivity

hard-mixed, `p_recheck=0.50`：

| Parameter | Value | E2E rarity recog. | L2 rarity recog. | Retention | False rarity |
|---|---:|---:|---:|---:|---:|
| theta_rare | -0.05 | 46.10 ± 3.70 | 74.13 ± 3.95 | 99.27 | 0 |
| theta_rare | -0.03 | 62.20 ± 3.86 | 100.00 ± 0.00 | 100.00 | 0 |
| theta_rare | -0.01 | 62.20 ± 3.86 | 100.00 ± 0.00 | 100.00 | 0 |
| theta_tol | 0.025 | 62.20 ± 3.86 | 100.00 ± 0.00 | 100.00 | 0 |
| theta_tol | 0.050 | 62.20 ± 3.86 | 100.00 ± 0.00 | 100.00 | 0 |
| theta_tol | 0.075 | 62.20 ± 3.86 | 100.00 ± 0.00 | 100.00 | 0 |

与 thesis A.8 Threshold-sensitivity Notes 对齐。

### 6.4 Corner-family divergence

| rho | Runs | Rarity recog. | Retention | False rarity | Main acc. | Corner acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 | 84.49 ± 0.10 | 20.73 ± 0.63 |
| 0.30 | 5 | 58.78 ± 3.16 | 58.78 ± 3.16 | 0 | 84.37 ± 0.10 | 34.16 ± 0.82 |
| 0.50 | 5 | 99.27 ± 0.67 | 99.27 ± 0.67 | 0 | 80.19 ± 0.59 | 46.18 ± 0.41 |
| 0.70 | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0 | 81.23 ± 0.54 | 51.10 ± 0.41 |
| 1.00 | 5 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0 | 85.58 ± 0.55 | 61.24 ± 0.53 |

与 thesis Table A.9 对齐。`rho=0.00` 的 corner accuracy 新导出 raw mean 为 `20.725%`，显示为 `20.73%`。

### 6.5 Corner-harm threshold calibration

| Setting | theta_corner_harm | Main acc. | Corner acc. | Corner-harm survival | Honest false corner-harm reject |
|---|---:|---:|---:|---:|---:|
| strict -0.005 | -0.00500 | 85.08 ± 0.60 | 63.21 ± 0.56 | 84.00 ± 5.48 | 100.00 ± 0.00 |
| calibrated mu+1sigma | -0.00290 ± 0.00002 | 85.50 ± 0.58 | 61.79 ± 0.53 | 84.00 ± 5.48 | 12.28 ± 5.88 |
| default 0.000 | 0.00000 | 85.58 ± 0.55 | 61.24 ± 0.53 | 84.00 ± 5.48 | 0.00 ± 0.00 |
| relaxed 0.005 | 0.00500 | 85.58 ± 0.55 | 61.24 ± 0.53 | 84.00 ± 5.48 | 0.00 ± 0.00 |

与 thesis A.8 的 corner-harm threshold notes 对齐。这里能清楚说明：负阈值不是更安全，反而会误拒 Honest；默认 0 阈值是有实验依据的。

## 7. 反造假/反循环性检查

### 7.1 未发现核心指标硬编码

搜索核心代码中 thesis 关键数值（如 `85.58`、`61.24`、`0.0489`、`410` 等）未发现用于核心计算的硬编码指标注入。相关数字主要出现在 PDF、docs、result CSV 或报告文本中。

### 7.2 L2 判决使用 loss drift，不使用 truth label

`DualChannelAuditor.audit` 的 runtime 判决逻辑：

- `delta_main > theta_tol` -> FRAUD
- `delta_corner <= theta_rare and delta_main <= theta_tol` -> RARITY
- `delta_main <= 0 and delta_corner > theta_corner_harm` -> FRAUD
- `delta_main < 0` -> HONEST
- otherwise -> NOISE

对应代码位于 `backend/l2_dual_audit/classifier.py:190` 至 `276`。这条路径基于模型更新后的 main/corner loss drift，不基于 ground-truth label。

### 7.3 ALG truth label 的用途

ALG generator 使用 archetype truth 是为了构造受控 benchmark 和统计指标。`export_v25_artifacts.py` 汇总的是重新运行后生成的 L1/L2 rows，并非直接把 thesis 表格写入结果。`build_update_confusion_matrix`、`build_rarity_recognition_retention_rows`、`build_audit_oracle_consistency_rows` 从 raw rows 生成汇总表，见 `scripts/export_v25_artifacts.py:1210` 至 `1224`。

### 7.4 Real-gradient 的边界

real-gradient benchmark 使用 MNIST/FashionMNIST/LEAF-FEMNIST client updates，并注入 sign-flip/corner-harm/noise 攻击更新。它不是完整真实车队部署，也不是 BDD100K live federated run。当前 thesis 已说明这是 real-data-gradient benchmark 和 ALG synthetic diagnostic 的组合，避免把 “real-gradient” 夸大成真实车辆线上实验。

## 8. 风险和修复项

### R1. README 复现命令不完整，高优先级

当前默认值在 `scripts/generate_demo_data.py:73` 至 `77`：

```python
BASE_BATCH_SIZE = 32
SIMULATION_SCALE_FACTOR = 3
BATCH_SIZE = BASE_BATCH_SIZE * SIMULATION_SCALE_FACTOR  # 96
VEHICLE_POOL_SIZE = max(128, BATCH_SIZE * 4)            # 384
```

但是 thesis ALG 表格使用：

- `BATCH_SIZE=24`
- `VEHICLE_POOL_SIZE=128`

审计时的 README 命令没有设置这两个环境变量：

```bash
python scripts/export_v25_artifacts.py --rounds 24 --cycle-rounds 12 --pretrain-epochs 5 --output-dir results/v25_artifacts
```

实际审计中，不设置 env var 会生成另一组规模的结果，不能复现 thesis。repo hygiene pass 已把顶层 README 和 `scripts/reproduce_all.sh` 改为显式使用 thesis scale。等价命令为：

```bash
BATCH_SIZE=24 VEHICLE_POOL_SIZE=128 python scripts/export_v25_artifacts.py \
  --rounds 24 --cycle-rounds 12 --pretrain-epochs 5 \
  --seeds 20260318,20260319,20260320,20260321,20260322 \
  --recheck-values 0.00,0.05,0.10,0.20,0.30 \
  --output-dir results/v25_artifacts
```

后续更稳妥的工程修复仍然建议给 `export_v25_artifacts.py`、stress、divergence、threshold calibration 脚本增加显式 CLI 参数，例如 `--batch-size` 和 `--vehicle-pool-size`，进一步减少隐藏环境变量。

### R2. 结果文件未纳入可审计归档

审计时 `results/` 下的复现实验文件未见 tracked 状态，likely 被 `.gitignore` 排除。repo hygiene pass 已保留 `results/README.md` 和 `results/expected_results.csv` 作为轻量复现索引，并继续忽略大体积 runtime outputs。

建议创建并提交一个独立 evidence package：

- `results/real_gradient_reliability_medium/`
- `results/real_gradient_full_method_comparison.csv`
- `results/real_gradient_adaptive_method_comparison.csv`
- `results/audit_reproduction/v25_artifacts_b24/`
- `results/audit_reproduction/v25_stress_tests_b24/`
- `results/audit_reproduction/corner_family_divergence_b24/`
- `results/audit_reproduction/corner_harm_threshold_calibration_b24/`
- SHA256 manifest
- reproduction commands
- Python/package/environment snapshot

如果不提交大文件，也应至少打包成 zip 并在 thesis appendix 或 viva materials 中提供 manifest。

### R3. 文档/schema 中旧阈值描述已修复

审计时发现 stale description/comment，现已在 runtime-facing docs/code comments 中修复：

- `backend/common/schemas/policy.py`
- `backend/l2_dual_audit/classifier.py`
- `docs/formulas/MATHEMATICAL_FORMULAS.md`
- `docs/architecture/ARCHITECTURE.md`
- `docs/architecture/L2_DUAL_AUDIT.md`

但 runtime 逻辑和 thesis 当前 Table 4.2 使用的是 `ΔL_main ≤ theta_tol`，见 `backend/l2_dual_audit/classifier.py:196` 和 `223`。

后续 repo hygiene pass 已将这些 runtime-facing 描述统一为：

```text
RARITY iff ΔL_corner ≤ theta_rare and ΔL_main ≤ theta_tol
```

保留 legacy 报告中的历史描述时，应明确其为 legacy/design-history，不作为当前实现依据。

### R4. 两个同名 PDF 容易混淆

本地曾存在两个同名 PDF，哈希不同。建议保留一个最终版，或在文件名中加入日期/版本号，例如：

```text
SAT301_FYP_Dissertation_CornerDrive_final_2026-05-13.pdf
```

并在 evidence package 中记录最终 PDF SHA256。

## 9. 最终判定

| Claim 类别 | 审计结果 |
|---|---|
| real-gradient macro/adaptive 数字 | 通过：CSV 可回算 |
| ALG 主表 p=0.10 85.58/61.24/410 rarity/0 sign-flip | 通过：带正确 env var 可复现 |
| corner-harm survival 来自 L1 bypass 而非 L2 漏判 | 通过：p=0.10 routing 和 L2 confusion 支持 |
| stress rarity/proxy/divergence | 通过：重新导出一致 |
| threshold sensitivity/corner-harm calibration | 通过：重新导出一致 |
| 无标签泄漏/无循环判决 | 基本通过：未发现 verdict 使用 truth label |
| 无数据造假 | 未发现证据；但需要修复 reproducibility/doc hygiene 风险 |

结论：当前 thesis 的主要实验 claim 是可验证的；最大问题不是 fabricated data，而是 reproducibility packaging 和 README 命令不完整。如果按 R1-R3 修复，学术诚信风险会显著降低。
