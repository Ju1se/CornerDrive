import axios from 'axios'

import type {
  SystemStats,
  VehicleStats,
  TierDistribution,
  RecentAudit,
  HealthCheck,
  L3Status,
} from '../types'
import type {
  GLMDecisionLogResponse,
  BaselineAnalysisResponse,
  Policy,
  PolicyExplanation,
  PolicyProposal,
  RoundTelemetry,
  RoundTelemetryInput,
} from '../types/policy'

const rawL1Api = import.meta.env.VITE_L1_API_URL || '/api/l1'
const rawL4Api = import.meta.env.VITE_L4_API_URL || '/api/l4'
const rawPolicyApi = import.meta.env.VITE_POLICY_AGENT_URL || '/api/policy'
const apiKey = import.meta.env.VITE_API_KEY || ''

function normalizePolicyApiBase(baseUrl: string) {
  const trimmed = baseUrl.replace(/\/$/, '')
  if (!trimmed.startsWith('http')) {
    return trimmed
  }
  return trimmed.endsWith('/api/v1/policy') ? trimmed : `${trimmed}/api/v1/policy`
}

const POLICY_API = normalizePolicyApiBase(rawPolicyApi)
const POLICY_HEALTH_API = `${POLICY_API}/health`

const api = axios.create({
  timeout: 10000,
})

const policyApi = axios.create({
  timeout: 30000,
})

const slowPolicyApi = axios.create({
  timeout: 90000,
})

for (const client of [api, policyApi, slowPolicyApi]) {
  if (apiKey) {
    client.defaults.headers.common['X-API-Key'] = apiKey
  }
}

function isNotFoundError(error: unknown) {
  return axios.isAxiosError(error) && error.response?.status === 404
}

async function getOrNullOnNotFound<T>(url: string, client = api): Promise<T | null> {
  try {
    const { data } = await client.get<T>(url)
    return data
  } catch (error) {
    if (isNotFoundError(error)) {
      return null
    }
    throw error
  }
}

// L1 API
export async function fetchL1Health(): Promise<HealthCheck> {
  const { data } = await api.get(`${rawL1Api}/health`)
  return data
}

// L4 API
export async function fetchL4Health(): Promise<HealthCheck> {
  const { data } = await api.get(`${rawL4Api}/health`)
  return data
}

export async function fetchL3Status(): Promise<L3Status> {
  const { data } = await api.get(`${rawL4Api}/api/v1/l3/status`)
  return data
}

export async function fetchPolicyHealth(): Promise<HealthCheck> {
  const { data } = await policyApi.get(POLICY_HEALTH_API)
  return data
}

export async function fetchSystemStats(): Promise<SystemStats> {
  const { data } = await api.get(`${rawL4Api}/api/v1/stats`)
  return data
}

export async function fetchVehicleStats(address: string): Promise<VehicleStats> {
  const { data } = await api.get(`${rawL4Api}/api/v1/vehicle/${address}`)
  return data
}

export async function fetchTierDistribution(): Promise<TierDistribution> {
  const { data } = await api.get(`${rawL4Api}/api/v1/tiers`)
  return data
}

export async function fetchRecentAudits(limit = 20): Promise<RecentAudit[]> {
  const { data } = await api.get(`${rawL4Api}/api/v1/recent-audits?limit=${limit}`)
  return data
}

export async function fetchVehicles(page = 1, limit = 10) {
  const { data } = await api.get(`${rawL4Api}/api/v1/vehicles?page=${page}&limit=${limit}`)
  return data
}

// Policy Agent API
export async function fetchCurrentPolicy(): Promise<Policy | null> {
  return getOrNullOnNotFound<Policy>(`${POLICY_API}/current`, policyApi)
}

export async function fetchNextPolicy(): Promise<Policy | null> {
  return getOrNullOnNotFound<Policy>(`${POLICY_API}/next`, policyApi)
}

export async function fetchLatestPolicyProposal(): Promise<PolicyProposal | null> {
  return getOrNullOnNotFound<PolicyProposal>(`${POLICY_API}/proposal/latest`, policyApi)
}

export async function fetchPolicyHistory(limit = 10): Promise<Policy[]> {
  const { data } = await slowPolicyApi.get(`${POLICY_API}/history?limit=${limit}`)
  return data
}

export async function fetchLatestTelemetry(): Promise<RoundTelemetry | null> {
  return getOrNullOnNotFound<RoundTelemetry>(`${POLICY_API}/telemetry/latest`, policyApi)
}

export async function fetchTelemetryHistory(limit = 24): Promise<RoundTelemetry[]> {
  const { data } = await slowPolicyApi.get(`${POLICY_API}/telemetry?limit=${limit}`)
  return data
}

export async function proposePolicy(telemetry: RoundTelemetryInput): Promise<PolicyProposal> {
  const { data } = await policyApi.post(`${POLICY_API}/propose`, telemetry)
  return data
}

export async function activatePolicy(roundId: number) {
  const { data } = await policyApi.post(`${POLICY_API}/activate`, null, {
    params: { round_id: roundId },
  })
  return data
}

export async function fetchPolicyExplanation(roundId: number): Promise<PolicyExplanation | null> {
  return getOrNullOnNotFound<PolicyExplanation>(`${POLICY_API}/explanation/${roundId}`, policyApi)
}

export async function fetchGLMDecisions(): Promise<GLMDecisionLogResponse> {
  const { data } = await policyApi.get(`${POLICY_API}/glm-decisions`)
  return data
}

export async function fetchBaselineAnalysis(rounds = 12): Promise<BaselineAnalysisResponse> {
  const { data } = await slowPolicyApi.get(`${POLICY_API}/analysis/baselines?rounds=${rounds}`)
  return data
}
