export interface SystemStats {
  total_vehicles: number
  total_audits: number
  fraud_count: number
  rare_count: number
  honest_count: number
  noise_count: number
  fraud_rate: number
  total_rewards_distributed: number
  total_slashed: number
}

export interface VehicleStats {
  address: string
  reputation: number
  tier: string
  tier_multiplier: number
  total_contributions: number
  fraud_count: number
  rare_count: number
  stake: number
  rewards_earned: number
  is_registered: boolean
}

export interface TierDistribution {
  bronze: number
  silver: number
  gold: number
  platinum: number
}

export interface RecentAudit {
  vehicle_id: string
  classification: 'FRAUD' | 'RARITY' | 'HONEST' | 'NOISE'
  delta_loss_main: number
  delta_loss_corner: number
  sbt_points: number
  timestamp: string
}

export interface HealthCheck {
  status: string
  layer: string
  service: string
  timestamp: string
  checks: Record<string, string>
}

export interface L3Status {
  lifecycle: string
  dataset_source: string
  dataset_path: string
  dataset_artifacts_present: boolean
  sample_count: number | null
  sample_shape: number[] | null
  drift_threshold: number
  policy_round: number | null
  detail: string
}
