# FLPG 梯度稀有性与欺诈性判定机制报告

> Legacy note: this report predates the V2.5 benchmark cleanup. It is retained
> only as historical design context. Do not use its numeric claims or blind-spot
> statements for Chapter 4; use `docs/reports/V25_CODE_AUDIT_AND_CLEANUP.md`
> and `results/v25_artifacts/` instead.

## 摘要

本文对 FLPG 当前实现中的梯度审计机制进行形式化说明，重点回答系统如何区分 `RARITY` 与 `FRAUD` 两类更新。该系统采用两阶段判定框架：第一阶段由 `L1 Linear Defense` 通过几何中位数与余弦偏离分数进行快速可疑样本筛查；第二阶段由 `L2 Dual-Purpose Audit` 在主任务数据集与 corner-case 数据集上分别计算梯度应用前后的损失漂移，并依据阈值规则给出 `FRAUD`、`RARITY`、`HONEST` 或 `NOISE` 判定。与传统仅关注主任务效用的联邦学习防御不同，该机制显式保留了“对角落场景有价值但对总体分布不常见”的更新，从而在安全性与长尾价值挖掘之间建立平衡。本文进一步给出核心数学公式、判定优先级、实现代码映射与当前实现边界，以便用于答辩、文档撰写与系统审计。

## 1. 引言

在联邦学习场景中，异常梯度并不总是等价于恶意梯度。某些更新可能偏离主流方向，却携带对稀有场景、长尾样本或 corner cases 至关重要的信息。若系统只依据“是否偏离多数”进行剔除，将导致真实但罕见的知识被误杀；若完全不做防御，则又可能引入投毒或破坏性更新。FLPG 的设计目标正是在这两类风险之间实现可解释、可执行的区分。

当前仓库中的实现将这一问题拆解为两个层次。`L1` 只回答“该梯度是否值得进一步审查”；`L2` 才回答“该梯度究竟是欺诈、稀有、有益还是噪声”。因此，系统并非直接从原始梯度一步判断 `RARITY` 或 `FRAUD`，而是先进行鲁棒筛查，再进行双通道损失分析。

## 2. 系统判定流程概述

FLPG 中与本问题直接相关的链路如下：

1. 车辆提交梯度到 `L1`。
2. `L1` 对同批梯度求几何中位数，并计算每个梯度相对中位数的余弦偏离分数。
3. 偏离分数高于阈值的梯度被标记为 `suspect`，推入 `L2` 审计队列。
4. `L2` 将候选梯度虚拟施加到当前模型参数上，并分别评估其对主任务数据集与 corner-case 数据集的损失变化。
5. `L2` 依据固定顺序的规则进行分类：
   - 优先判定 `FRAUD`
   - 其次判定 `RARITY`
   - 再判定 `HONEST`
   - 其余归为 `NOISE`

这一流程意味着：`L1` 的作用是降低审计成本，`L2` 的作用是完成语义判定。

## 3. L1: 可疑梯度筛查

### 3.1 几何中位数聚合

对一批梯度向量 $\{g_i\}_{i=1}^{n}$，系统首先计算几何中位数：

```math
w^* = \arg\min_w \sum_{i=1}^{n} \lVert w - g_i \rVert_2
```

该优化问题在实现中通过 Weiszfeld 迭代求解。其更新形式为：

```math
w_{t+1} = \frac{\sum_{i=1}^{n} \frac{g_i}{\lVert w_t - g_i \rVert_2}}{\sum_{i=1}^{n} \frac{1}{\lVert w_t - g_i \rVert_2}}
```

几何中位数相较于算术平均对离群点更鲁棒，因此适合作为联邦学习聚合中的“主流方向”参考。

### 3.2 余弦偏离分数与统计量选择

本文将 L1 的筛查统计量明确固定为“相对几何中位数的余弦偏离分数”，而不是在多种候选统计量之间保留一个开放菜单。对每个梯度 $g_i$，先计算其与参考向量 $w^*$ 的余弦相似度：

```math
\text{sim}(g_i, w^*) = \frac{g_i \cdot w^*}{\lVert g_i \rVert_2 \lVert w^* \rVert_2}
```

再定义偏离分数为：

```math
\text{Score}_i = 1 - \text{sim}(g_i, w^*)
```

其解释为：

- 当 `Score` 接近 `0` 时，梯度与主流方向基本一致。
- 当 `Score` 接近 `1` 时，梯度与主流方向近似正交，具备明显可疑性。
- 当 `Score` 接近 `2` 时，梯度与主流方向相反，通常具有更高风险。

若

```math
\text{Score}_i > \tau_{\text{cos}}
```

则该梯度被路由为 `suspect`，其中 $\tau_{\text{cos}}$ 对应当前 policy 中的 `cosine_filter_threshold`。

之所以选择余弦偏离而不是单纯比较梯度范数或欧氏距离，原因有三。第一，L1 的目标是识别“方向上偏离主流”的更新，而不是仅仅发现幅度较大的更新；在 non-IID 联邦学习中，局部样本量、局部学习率和数据难度都会引入幅度波动，单看范数容易把统计噪声当成异常。第二，余弦相似度对尺度不敏感，更适合作为 heterogeneous client updates 的第一层快速筛查指标。第三，在实现层面，一旦参考向量 $w^*$ 已得到，每个客户端的余弦分数计算复杂度仅为 $O(d)$，便于在高维梯度上批量执行。

需要强调的是，这里的参考向量并非算术平均，而是几何中位数 $w^*$。这使得“方向比较”与“鲁棒中心估计”配套出现：几何中位数负责提供不易被离群点拖拽的主流方向，余弦偏离负责刻画单个客户端相对这一主流方向的角度偏离。

### 3.3 L1 的角色边界与 L1-L2 衔接

需要强调的是，`L1` 只执行“筛查”，并不直接输出 `RARITY` 或 `FRAUD`。因此，`L1` 中被判为 suspect 的梯度只是“值得进一步调查”，最终语义仍由 `L2` 决定。即便一个更新被 L1 成功识别为方向异常，它仍可能在 L2 中被进一步区分为 `FRAUD`、`RARITY` 或 `NOISE`；换言之，L1 提供的是审计优先级，而不是最终语义标签。

这一边界定义也决定了 L1 的设计原则应偏向“控制进入 L2 的队列规模，同时尽量不漏掉真正危险的更新”。如果 L1 过于宽松，大量本应直接聚合的更新会被送入 L2，副本评估开销将近似线性增长；如果 L1 过于严格，则真正需要审计的欺诈或稀有更新会在第一层被直接放行，导致后续判定链条失效。

### 3.4 L1 的可复现实验指标

在仓库的统一 24 轮 benchmark 中，实验配置为每轮 96 个客户端，共 2304 个客户端更新；其中 ground-truth 预分类总量为 `HONEST = 1846`、`RARITY = 232`、`FRAUD = 140`、`NOISE = 86`。采用默认初始策略 `cosine_filter_threshold = 0.70` 时，L1 共将 306 个更新路由至 L2 审计队列，占全部更新的 `13.28%`，即平均每轮 `12.75` 个候选需要进入副本审计。

从队列构成看，这 306 个候选分别包含 `140` 个 `FRAUD`、`80` 个 `RARITY`、`86` 个 `NOISE`、`0` 个 `HONEST`。因此，若将“进入队列的非 HONEST 比例”定义为 L1 的 routing precision，则该实验下 L1 precision 为 `100%`；相应地，`HONEST` 更新的 L1 false-positive rate 为 `0%`。

进一步看各类更新的 routing recall：

- `FRAUD`: `140 / 140 = 100%`
- `NOISE`: `86 / 86 = 100%`
- `RARITY`: `80 / 232 = 34.48%`

这组数字揭示了 L1 与 L2 的分工关系。L1 对明显破坏性的 `FRAUD` 和高偏离 `NOISE` 具有极高的召回能力，但只会把约三分之一的 `RARITY` 更新送入审计队列；因此，L2 的职责不是简单复核一个“几乎全是恶意”的队列，而是在一个已经被压缩到 13.28% 规模的候选集合中，进一步区分“应惩罚的欺诈”和“应保留的稀有价值”。在同一组 24 轮 benchmark 中，完整 FLPG 最终达到 `fraud precision = 100%`、`fraud recall = 100%`、`rarity precision = 100%`、`rarity recall = 8.19%`，这说明 L2 的意义不在于扩大队列，而在于对已路由候选执行高精度语义裁决。

队列规模也具有明显的 phase sensitivity。在 `steady` 阶段，L1 平均每轮仅路由 `6` 个候选；在 `fraud_wave` 阶段平均为 `30` 个，在 `drift_burst` 阶段平均为 `26` 个，在 `corner_gap` 阶段平均为 `13` 个，在 `false_slash_risk` 阶段平均为 `5` 个。全实验中最大队列规模出现在 `fraud_wave`，单轮为 `30` 个候选。换言之，L1 的保守性直接决定了 L2 的最大副本审计开销，而当前阈值设置将这一额外成本稳定控制在每轮参与客户端的约八分之一量级。

## 4. L2: 稀有性与欺诈性的双通道审计

### 4.1 虚拟施加梯度

设当前模型参数为 $W$，候选梯度为 $g$，学习率为 $\eta$。系统不直接更新真实全局模型，而是先在模型副本上构造候选参数：

```math
W' = W - \eta g
```

这一步允许系统在不污染真实模型的前提下，分析该梯度的潜在影响。

### 4.2 主任务损失漂移

在主任务数据集 $D_{\text{main}}$ 上，系统定义主任务损失漂移为：

```math
\Delta L_{\text{main}} = L(W - \eta g; D_{\text{main}}) - L(W; D_{\text{main}})
```

其意义为：

- $\Delta L_{\text{main}} < 0$：该梯度改善主任务性能。
- $\Delta L_{\text{main}} > 0$：该梯度恶化主任务性能。

如果这一恶化超过容忍阈值 $\theta_{\text{tol}}$，则系统认为该梯度对主任务造成了不可接受的破坏。

### 4.3 Corner-case 损失漂移

在 corner-case 数据集 $D_{\text{corner}}$ 上，系统定义角落场景损失漂移为：

```math
\Delta L_{\text{corner}} = L(W - \eta g; D_{\text{corner}}) - L(W; D_{\text{corner}})
```

其意义为：

- $\Delta L_{\text{corner}} < 0$：该梯度改善 corner cases。
- $\Delta L_{\text{corner}} > 0$：该梯度恶化 corner cases。

其中，系统以负阈值 $\theta_{\text{rare}} < 0$ 作为“显著稀有价值”的判定边界。也就是说，只有当 corner-case 损失下降到足够明显的程度时，系统才会将其认定为真正的 `RARITY`。

## 5. 分类规则与判定优先级

FLPG 在 `L2` 中采用严格的顺序判定，而非并行投票。具体规则可形式化写为：

### 5.1 欺诈判定

若

```math
\Delta L_{\text{main}} > \theta_{\text{tol}}
```

则分类为：

```text
FRAUD
```

该规则表达的是“主任务效用显著受损”。在系统设计上，这一规则具有最高优先级。

### 5.2 稀有判定

若上一条不成立，且

```math
(\Delta L_{\text{corner}} \le \theta_{\text{rare}}) \land (\Delta L_{\text{main}} \le 0)
```

则分类为：

```text
RARITY
```

这里的“稀有”并非统计频率意义上的低频，而是“对角落场景具有显著正效用，且不会伤害主任务”的语义定义。

### 5.3 诚实判定

若前两条均不成立，且

```math
\Delta L_{\text{main}} < 0
```

则分类为：

```text
HONEST
```

这表示该梯度直接改善主任务，但并未触发显著 corner-case 奖励机制。

需要指出的是，这一定义对应的是“当前实现规则”，而不是一个已经完全消除边界漏洞的理论最优规则。由于该规则只检查主任务通道，未对 $\Delta L_{\text{corner}}$ 施加保护条件，因此无法排除如下事件：

```math
(\Delta L_{\text{main}} < 0) \land (\Delta L_{\text{corner}} > 0)
```

也就是说，某个更新可能对主任务有轻微帮助，但同时伤害角落案例；在当前实现中，这类更新仍会被判为 `HONEST` 并进入标准聚合。这个问题可以称为 `HONEST` 判决的 corner-case blind spot。

在当前 synthetic benchmark 中，这类案例并未被显式构造出来，因此实验没有观测到该盲区被触发；但这只能说明 benchmark 尚未覆盖该边界情形，不能说明当前规则已经从定义上排除了该风险。若论文中的 Table 3.3 需要消除此逻辑漏洞，更保守的写法应为：

```math
\text{HONEST}_{\text{safe}} \iff (\Delta L_{\text{main}} < 0) \land (\Delta L_{\text{corner}} \le \theta_{\text{corner-harm}})
```

其中 $\theta_{\text{corner-harm}} \ge 0$ 是一个小的容忍阈值；默认最保守的选择可取 $\theta_{\text{corner-harm}} = 0$。若某个更新满足 $\Delta L_{\text{main}} < 0$ 但 $\Delta L_{\text{corner}} > \theta_{\text{corner-harm}}$，则更合理的做法是将其下放到 `NOISE` 或 `RECHECK`，而不是直接作为 `HONEST` 纳入聚合。这样可以把“有益于主任务”和“不损害稀有场景”同时纳入诚实判决条件。

### 5.4 噪声判定

若以上条件均不满足，则分类为：

```text
NOISE
```

这类梯度既没有明显伤害主任务，也没有为 corner cases 提供足够强的帮助，因此被视为影响有限的更新。

### 5.5 判定顺序的重要性

由于系统先判断 `FRAUD`，再判断 `RARITY`，并且 `RARITY` 还要求 `ΔL_main ≤ 0`，因此当某个梯度同时满足“损害主任务”与“改善 corner cases”两种性质时，最终结果不会落入 `RARITY`。这一设计体现了安全性优先于稀有奖励的原则。

## 6. 评分与证据生成

除了离散分类标签外，系统还计算一个用于结算优先级的连续分数：

```math
\text{Score} = |\Delta L_{\text{main}}| + \lambda \cdot \max(0, -\Delta L_{\text{corner}})
```

其中：

```math
\lambda = 0.5 \times \text{corner\_weight}
```

该分数并不改变分类结果，而是用于刻画一次更新在“主任务影响”和“corner-case 价值”上的综合显著性。

对于被判为 `FRAUD` 的梯度，系统会生成 fraud proof，记录：

- `vehicle_id`
- `delta_loss_main`
- `delta_loss_corner`
- `fraud_threshold`
- `gradient_hash`
- `timestamp`

对于被判为 `RARITY` 的梯度，系统会生成 rarity certificate，记录：

- `vehicle_id`
- `delta_loss_main`
- `delta_loss_corner`
- `rarity_threshold`
- `corner_improvement`
- `gradient_hash`
- `timestamp`

因此，FLPG 的 `RARITY/FRAUD` 判定不仅是标签输出，也伴随可追溯的证据对象。

## 7. 当前实现中的参数含义

在当前 policy schema 中，相关核心参数如下：

- `theta_tol`：主任务损失容忍阈值，决定多大程度的主任务恶化会被视为欺诈。
- `theta_rare`：corner-case 稀有奖励阈值，为负值，越负代表越严格。
- `cosine_filter_threshold`：L1 的 suspect 路由阈值。
- `corner_weight`：影响结算优先分数中的 rarity bonus 权重。
- `slash_multiplier`：欺诈惩罚倍数。
- `rarity_reward_multiplier`：稀有奖励倍数。

默认策略下，系统使用：

- `theta_tol = 0.05`
- `theta_rare = -0.03`
- `cosine_filter_threshold = 0.70`

在当前实现中，这组默认值同时承担“实验初始化”的角色。也就是说，L1 的筛查阈值并不是在 benchmark 开始时临时搜索得到，而是以 `0.70` 作为初始 policy prior 启动；之后若启用 adaptive policy，policy agent 可以基于回合 telemetry 对这些阈值进行调整，但 `L2` 的判定逻辑本身保持确定性不变。因此，要保证实验可复现，论文中至少应同时报告初始 policy、是否启用 policy adaptation，以及 benchmark 轮数与每轮客户端数。

### 7.1 对应 Thesis Section 3.5 的策略层文献支撑

若将第 3.5 节描述为“基于遥测的自适应阈值机制”，则这一机制不应被表述为凭空出现的工程启发式，而应明确放在在线学习、概念漂移检测和联邦学习异质性控制的文献脉络中。

第一，FLPG 的 policy layer 是按轮次观察 telemetry 并执行小步长参数更新，这一设计更接近 online learning / online convex optimization 中“在序列反馈下持续修正决策”的框架，而不是一次性离线调参。Shalev-Shwartz 对在线学习的综述可作为这一设计的基础性方法论支撑。

第二，策略层使用 recent telemetry、`recent_attack_pressure`、`suspect_queue_length` 等统计量来决定是否收紧或放宽阈值，这与概念漂移检测中“根据数据流变化自适应调整窗口和敏感度”的思想一致。Bifet 和 Gavaldà 的 ADWIN 工作说明了：在非平稳数据流中，窗口长度与阈值不应固定，而应随变化强度在线调整。

第三，在联邦学习场景中，这种自适应控制还可直接借助 client drift 文献来获得问题动机。Karimireddy et al. 指出，non-IID 异质性会导致客户端更新相对全局目标发生 drift；Panchal et al. 进一步表明，在 FL 中 concept drift 应被显式检测，并以 drift-aware adaptive optimization 的方式修正服务器端决策。FLPG 的策略层并未声称满足这些工作的理论收敛界或 regret bound，而是一个受其启发的 bounded heuristic controller：它以轮次 telemetry 作为在线反馈，对 `theta_tol`、`theta_rare`、`cosine_filter_threshold` 等参数施加受限的增量更新，以在 fraud suppression、rarity preservation 与 false-slash protection 之间取得折中。

因此，在 thesis 中更稳妥的写法是：第 3.5 节提出的是一种“受在线学习与漂移自适应思想启发的轮次级策略控制层”，而非一个已经完成理论证明的最优控制器。

### 7.2 威胁模型与实验构造的对应关系

`Attack of the Tails` 的核心威胁叙事是 edge-case backdoor：攻击者并不一定大范围破坏整体精度，而是让模型在输入分布尾部、训练中少见但现实中敏感的样本上发生系统性错误。该论文强调的重点是“tail-distribution / edge-case failure”与“可绕过常见防御的后门注入”。

FLPG 当前 benchmark 与该威胁模型的关系，应准确表述为“概念上受其启发，但实验上采用了梯度层 surrogate，而非语义触发器的一比一复现”。具体来说，实验中的 `FRAUD` 更新不是通过像素触发器、文本触发 token 或显式后门样本训练出来的，而是在梯度生成器中被构造成一种具有以下服务器侧可观测特征的有害更新：

- 与 honest reference 存在较大方向偏离，因此更容易越过 L1 的 cosine screening cutoff。
- 在 L2 虚拟施加后满足 $\Delta L_{\text{main}} > \theta_{\text{tol}}$，即对主任务表现造成可量化破坏。
- 在 `fraud_wave` 和 `drift_burst` 等攻击压力更高的 phase 中，其出现比例被显式提高，以模拟阶段性攻击爆发。

从实现细节看，`FRAUD` 候选梯度由“反向主任务梯度 + sideband / drift 分量”混合生成，并且只有在同时满足 `FRAUD` 分类条件和足够大的方向偏离时才会被保留为最终 preflight fraud sample。也就是说，实验实际上测试的是一种 gradient-level functional proxy：它捕获了 edge-case/backdoor 攻击在服务器端留下的结果性信号，即“高偏离、破坏主任务、在异常阶段集中出现”，但并未完整实现 `Attack of the Tails` 那种带有语义触发条件的 tail-targeted backdoor training pipeline。

因此，在 thesis 中最安全的表述不应是“本实验复现了 Attack of the Tails”，而应是：

> 本文以 `Attack of the Tails` 作为 tail-focused backdoor threat 的概念动机；在实验层面，则使用一个梯度空间中的功能等价代理威胁模型来近似其服务器侧表现。该代理重点保留了尾部分布攻击的三个可审计特征：方向异常、主任务效用破坏和阶段性攻击压力，但未复现其完整的语义触发器机制。

这样写可以把文献综述中的威胁叙事与当前实验生成方式严密接起来，同时避免被答辩人追问“你是不是实际复现了原论文攻击”时出现表述过度。

## 8. 与实现代码的对应关系

本文公式与仓库实现一一对应：

- `L1` 几何中位数与余弦偏离：`backend/l1_linear_defense/aggregation.py`
- `L1` 动态读取 policy 阈值并将 suspect 推入队列：`backend/l1_linear_defense/server.py`
- `L2` 双通道损失计算与最终分类：`backend/l2_dual_audit/classifier.py`
- policy 参数定义：`backend/common/schemas/policy.py`
- 数学公式总览：`docs/formulas/MATHEMATICAL_FORMULAS.md`

从代码角度看，真正决定 `RARITY` 与 `FRAUD` 的语句是：

```python
if delta_main > self.fraud_threshold:
    classification = Classification.FRAUD
elif delta_corner <= self.rarity_threshold and delta_main <= 0:
    classification = Classification.RARITY
elif delta_main < 0:
    classification = Classification.HONEST
else:
    classification = Classification.NOISE
```

该片段构成了整个语义判定的核心。

## 9. 方法优势与局限性

### 9.1 优势

首先，该方法具有明确的可解释性。系统并非依赖不可解释的黑盒分类器，而是直接以损失变化为依据进行判定。其次，系统显式保留了角落场景价值，使得“罕见但有用”的更新不会因为偏离主流就被机械剔除。再次，分类规则具有确定性，便于审计、调参和与经济激励层对接。

### 9.2 局限性

当前实现仍存在若干边界。

其一，`RARITY` 的定义依赖于 `D_corner` 的构造质量。如果 corner dataset 代表性不足，则“稀有价值”的判定基础也会偏弱。其二，`L1` 只将 suspect 路由到 `L2`，这意味着某些与主流方向较接近、但对 corner cases 有价值的梯度，理论上可能不会进入 `L2` 审计链路。其三，当前规则虽然避免了奖励会伤害主任务的梯度，但仍然采用单标签输出，尚不能表达“corner 很有帮助但主任务略有代价”的混合型更新。其四，当前 demo 运行主要基于 synthetic assets，因此该机制的工程原型是成立的，但其现实世界泛化能力仍需基于真实任务与真实梯度进一步验证。

此外，仓库中的部分说明文档仍保留了早期阈值示例，而实际生效阈值应以当前 policy schema 和运行时 policy 为准。

## 10. 结论

FLPG 判断梯度是“稀有”还是“欺诈”的核心，不在于梯度本身是否偏离多数，而在于该梯度在两个任务通道上的效用表现：

- 若显著损害主任务，则判为 `FRAUD`。
- 若显著改善 corner cases，且不伤害主任务，则判为 `RARITY`。

形式化地说，系统通过 `L1` 完成鲁棒筛查，通过 `L2` 完成基于损失漂移的双通道判定，并以固定优先级规则将候选梯度映射为 `FRAUD`、`RARITY`、`HONEST` 或 `NOISE`。这一设计使 FLPG 能够在安全防御与稀有知识保留之间实现可解释的折中，也构成了该系统最关键的算法核心之一。

## 参考实现文件

- `backend/l1_linear_defense/aggregation.py`
- `backend/l1_linear_defense/server.py`
- `backend/l2_dual_audit/classifier.py`
- `backend/common/schemas/policy.py`
- `docs/formulas/MATHEMATICAL_FORMULAS.md`

## 可用于论文引用的外部文献锚点

- Blanchard, El Mhamdi, Guerraoui, and Stainer, *Byzantine-Tolerant Machine Learning* (`Krum`), 2017.
- Xu, Huang, Song, and Lan, *Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering* (`SignGuard`), 2021/2023 arXiv version.
- Wang, Sreenivasan, Rajput, Vishwakarma, Agarwal, Sohn, Lee, and Papailiopoulos, *Attack of the Tails: Yes, You Really Can Backdoor Federated Learning*, NeurIPS 2020.
- Shalev-Shwartz, *Online Learning and Online Convex Optimization*, 2012.
- Bifet and Gavaldà, *Learning from Time-Changing Data with Adaptive Windowing*, 2007.
- Karimireddy, Kale, Mohri, Reddi, Stich, and Suresh, *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*, ICML 2020.
- Panchal, Choudhary, Mitra, Mukherjee, Sarkhel, Mitra, and Guan, *Flash: Concept Drift Adaptation in Federated Learning*, ICML 2023.
