export interface Policy {
  round_id: number
  theta_tol: number
  theta_rare: number
  theta_drift: number
  cosine_filter_threshold: number
  recheck_probability: number
  honest_reward_multiplier: number
  slash_multiplier: number
  rarity_reward_multiplier: number
  corner_weight: number
  policy_version: string
  effective_from_round: number
  created_at: string
}

export interface RoundTelemetry {
  round_id: number
  fraud_rate: number
  rarity_rate: number
  honest_rate: number
  noise_rate: number
  main_accuracy: number
  corner_accuracy: number
  main_loss_delta_avg: number
  corner_loss_delta_avg: number
  false_slash_estimate: number
  rarity_retention_rate: number
  golden_drift_score: number
  reject_rate_l3: number
  cosine_outlier_ratio: number
  suspect_queue_length: number
  audit_sample_size: number
  avg_sbt_score: number
  new_vehicle_ratio: number
  hash_mismatch_rate: number
  recent_attack_pressure: number
  created_at: string
}

export type RoundTelemetryInput = Omit<RoundTelemetry, 'created_at'>

export interface PolicyProposal {
  current_policy: Policy
  proposed_policy: Policy
  reasons: string[]
  validator_passed: boolean
  safety_guard_passed: boolean
  blocked_reasons: string[]
  validator_messages: string[]
  round_id: number
  created_at: string
  source_engine: string
  approved: boolean
  explanation?: string | null
  complexity_score?: number | null
  complexity_reason?: string | null
  llm_used?: boolean | null
}

export interface PolicyExplanation {
  explanation: string
  round_id: number
  created_at?: string
  metadata?: Record<string, unknown>
}

export interface GLMParameterChange {
  param: string
  before: number
  after: number
}

export interface GLMDecision {
  round_id: number
  timestamp: string
  llm_used: boolean
  source_engine: string
  blocked: boolean
  reasons: string[]
  validator_messages: string[]
  parameters_changed: GLMParameterChange[]
  telemetry: Pick<
    RoundTelemetry,
    'fraud_rate' | 'rarity_rate' | 'honest_rate' | 'main_accuracy' | 'corner_accuracy'
  > | null
}

export interface GLMDecisionLogResponse {
  total: number
  offset: number
  limit: number
  data: GLMDecision[]
}

export interface BaselineRoundPoint {
  round_id: number
  phase: string
  main_accuracy: number
  corner_accuracy: number
  false_slash_estimate: number
  rarity_retention_rate: number
  fraud_rate: number
  rarity_rate: number
  honest_rate: number
  noise_rate: number
  main_loss_delta_avg: number
  corner_loss_delta_avg: number
  theta_rare: number
  rarity_reward_multiplier: number
  slash_multiplier: number
  corner_weight: number
}

export interface BaselineSummary {
  main_accuracy_avg: number
  corner_accuracy_avg: number
  false_slash_estimate_avg: number
  rarity_retention_rate_avg: number
  rarity_precision: number
  rarity_recall: number
  fraud_precision: number
  fraud_recall: number
}

export interface BaselineAnalysisEntry {
  id: string
  label: string
  description: string
  summary: BaselineSummary
  rounds: BaselineRoundPoint[]
}

export interface BaselineAnalysisResponse {
  generated_at: string
  scenario_policy_source: string
  scenario_policy_round: number
  classification_rounds: number
  baselines: BaselineAnalysisEntry[]
}
