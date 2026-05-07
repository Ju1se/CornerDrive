import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, Bot, CircuitBoard, History, Sparkles } from 'lucide-react'

import GLMDecisionLog from '../components/policy/GLMDecisionLog'
import { MetricTile, PageIntro, Panel, StateCallout, StatusBadge } from '../components/ui/primitives'
import {
  activatePolicy,
  fetchCurrentPolicy,
  fetchLatestPolicyProposal,
  fetchLatestTelemetry,
  fetchPolicyExplanation,
  fetchPolicyHistory,
  proposePolicy,
} from '../services/api'
import type { Policy, RoundTelemetryInput } from '../types/policy'

function createBlankTelemetryDraft(): RoundTelemetryInput {
  return {
    round_id: 0,
    fraud_rate: 0,
    rarity_rate: 0,
    honest_rate: 0,
    noise_rate: 0,
    main_accuracy: 0,
    corner_accuracy: 0,
    main_loss_delta_avg: 0,
    corner_loss_delta_avg: 0,
    false_slash_estimate: 0,
    rarity_retention_rate: 1,
    golden_drift_score: 0,
    reject_rate_l3: 0,
    cosine_outlier_ratio: 0,
    suspect_queue_length: 0,
    audit_sample_size: 0,
    avg_sbt_score: 0,
    new_vehicle_ratio: 0,
    hash_mismatch_rate: 0,
    recent_attack_pressure: 0,
  }
}

const numericFields: Array<{ key: keyof RoundTelemetryInput; label: string; step?: string }> = [
  { key: 'round_id', label: 'Round ID', step: '1' },
  { key: 'fraud_rate', label: 'Fraud Rate', step: '0.01' },
  { key: 'rarity_rate', label: 'Beneficial Rarity Rate', step: '0.01' },
  { key: 'honest_rate', label: 'Honest Rate', step: '0.01' },
  { key: 'noise_rate', label: 'Noise Rate', step: '0.01' },
  { key: 'main_accuracy', label: 'Main Accuracy', step: '0.01' },
  { key: 'corner_accuracy', label: 'Corner Accuracy', step: '0.01' },
  { key: 'false_slash_estimate', label: 'False Slash Estimate', step: '0.01' },
  { key: 'rarity_retention_rate', label: 'Beneficial Rarity Retention', step: '0.01' },
  { key: 'golden_drift_score', label: 'Golden Drift Score', step: '0.01' },
  { key: 'suspect_queue_length', label: 'Suspect Queue', step: '1' },
  { key: 'audit_sample_size', label: 'Audit Sample Size', step: '1' },
  { key: 'recent_attack_pressure', label: 'Attack Pressure', step: '0.01' },
]

const telemetrySections: Array<{
  title: string
  description: string
  fields: Array<{ key: keyof RoundTelemetryInput; label: string; step?: string }>
}> = [
  {
    title: 'Round framing',
    description: 'Basic round identity and high-level mix of observed outcomes.',
    fields: numericFields.filter((field) =>
      ['round_id', 'fraud_rate', 'rarity_rate', 'honest_rate', 'noise_rate'].includes(field.key),
    ),
  },
  {
    title: 'Task quality',
    description: 'Main-task and corner-task behavior used to infer whether novelty is beneficial or harmful.',
    fields: numericFields.filter((field) =>
      ['main_accuracy', 'corner_accuracy', 'rarity_retention_rate', 'golden_drift_score'].includes(field.key),
    ),
  },
  {
    title: 'Risk signals',
    description: 'Inputs that drive uncertainty protection, fraud pressure and review intensity.',
    fields: numericFields.filter((field) =>
      ['false_slash_estimate', 'suspect_queue_length', 'audit_sample_size', 'recent_attack_pressure'].includes(
        field.key,
      ),
    ),
  },
]

const policyFields: Array<{ key: keyof Policy; label: string }> = [
  { key: 'theta_tol', label: 'Theta Tol' },
  { key: 'theta_rare', label: 'Theta Rare' },
  { key: 'theta_drift', label: 'Theta Drift' },
  { key: 'cosine_filter_threshold', label: 'Cosine Filter' },
  { key: 'recheck_probability', label: 'Recheck Probability' },
  { key: 'honest_reward_multiplier', label: 'Honest Reward' },
  { key: 'slash_multiplier', label: 'Slash Multiplier' },
  { key: 'rarity_reward_multiplier', label: 'Beneficial Rarity Reward' },
  { key: 'corner_weight', label: 'Corner Weight' },
]

function formatPolicyNumber(value: number) {
  return Math.abs(value) < 0.1 ? value.toFixed(4) : value.toFixed(2)
}

function describeEngine(sourceEngine: string | undefined, llmUsed = false) {
  if (!sourceEngine) {
    return {
      label: 'Awaiting proposal',
      supporting: 'No proposal engine has been recorded yet.',
      raw: null,
    }
  }

  if (sourceEngine === 'rule_engine_fallback') {
    return {
      label: 'Rule engine fallback',
      supporting: 'Deterministic fallback path used instead of remote GLM reasoning.',
      raw: sourceEngine,
    }
  }

  if (sourceEngine === 'glm_policy_engine') {
    return {
      label: 'GLM policy engine',
      supporting: llmUsed
        ? 'Direct GLM reasoning generated this proposal.'
        : 'GLM route selected, but no remote model output was used.',
      raw: sourceEngine,
    }
  }

  return {
    label: sourceEngine.replace(/_/g, ' '),
    supporting: llmUsed ? 'Model-assisted proposal path.' : 'Deterministic proposal path.',
    raw: sourceEngine,
  }
}

function ProposalStatus({
  blocked,
  approved,
}: {
  blocked: boolean
  approved: boolean
}) {
  if (blocked) {
    return <StatusBadge label="Blocked" tone="red" />
  }

  if (approved) {
    return <StatusBadge label="Approved" tone="teal" />
  }

  return <StatusBadge label="Pending" tone="amber" />
}

function ParameterGrid({
  policy,
  previousPolicy,
}: {
  policy: Policy
  previousPolicy?: Policy | null
}) {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      {policyFields.map(({ key, label }) => {
        const currentValue = policy[key] as number
        const previousValue = previousPolicy?.[key] as number | undefined
        const changed = previousValue !== undefined && previousValue !== currentValue
        const increased = typeof previousValue === 'number' && currentValue > previousValue

        return (
          <div key={key} className="rounded-[18px] border border-slate-900/8 bg-white/70 p-4">
            <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">{label}</div>
            <div
              className={`mt-2 font-mono text-base font-semibold ${
                changed ? (increased ? 'text-rose-700' : 'text-emerald-700') : 'text-slate-900'
              }`}
            >
              {formatPolicyNumber(currentValue)}
            </div>
            {changed && typeof previousValue === 'number' ? (
              <p className="mt-2 text-xs text-slate-500">was {formatPolicyNumber(previousValue)}</p>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

export default function PolicyDashboard() {
  const queryClient = useQueryClient()
  const [telemetryDraft, setTelemetryDraft] = useState<RoundTelemetryInput>(createBlankTelemetryDraft)
  const [draftHydratedFromLatest, setDraftHydratedFromLatest] = useState(false)

  const { data: currentPolicy, isLoading: currentLoading, isError: currentPolicyError } = useQuery({
    queryKey: ['policy', 'current'],
    queryFn: fetchCurrentPolicy,
    retry: 1,
    refetchInterval: 10000,
  })

  const { data: latestProposal, isError: latestProposalError } = useQuery({
    queryKey: ['policy', 'proposal', 'latest'],
    queryFn: fetchLatestPolicyProposal,
    refetchInterval: 10000,
  })

  const { data: latestTelemetry, isError: latestTelemetryError } = useQuery({
    queryKey: ['policy', 'telemetry', 'latest'],
    queryFn: fetchLatestTelemetry,
    refetchInterval: 10000,
  })

  const { data: policyHistory, isError: policyHistoryError } = useQuery({
    queryKey: ['policy', 'history'],
    queryFn: () => fetchPolicyHistory(8),
    refetchInterval: 15000,
  })

  const { data: savedExplanation, isError: explanationError } = useQuery({
    queryKey: ['policy', 'explanation', latestProposal?.round_id],
    queryFn: () => fetchPolicyExplanation(latestProposal!.round_id),
    enabled: Boolean(latestProposal?.round_id),
  })

  const policyReadError = currentPolicyError || latestProposalError || latestTelemetryError || policyHistoryError || explanationError

  useEffect(() => {
    if (!latestTelemetry || draftHydratedFromLatest) return
    const { created_at, ...restTelemetry } = latestTelemetry
    void created_at
    setTelemetryDraft(restTelemetry)
    setDraftHydratedFromLatest(true)
  }, [draftHydratedFromLatest, latestTelemetry])

  const proposeMutation = useMutation({
    mutationFn: proposePolicy,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['policy', 'proposal'] }),
        queryClient.invalidateQueries({ queryKey: ['policy', 'telemetry'] }),
        queryClient.invalidateQueries({ queryKey: ['policy', 'history'] }),
      ])
    },
  })

  const activateMutation = useMutation({
    mutationFn: activatePolicy,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['policy', 'current'] }),
        queryClient.invalidateQueries({ queryKey: ['policy', 'proposal'] }),
        queryClient.invalidateQueries({ queryKey: ['policy', 'history'] }),
      ])
    },
  })

  const proposal = latestProposal
  const explanation = proposal?.explanation ?? savedExplanation?.explanation ?? null
  const blocked = proposal ? !proposal.safety_guard_passed || proposal.blocked_reasons.length > 0 : false
  const proposalEngine = describeEngine(proposal?.source_engine, Boolean(proposal?.llm_used))

  const historyRows = useMemo(
    () => (policyHistory ?? []).slice().sort((a, b) => b.round_id - a.round_id),
    [policyHistory],
  )

  const handleTelemetryChange = (key: keyof RoundTelemetryInput, value: string) => {
    setTelemetryDraft((current) => ({
      ...current,
      [key]:
        key === 'round_id' || key === 'suspect_queue_length' || key === 'audit_sample_size'
          ? Number.parseInt(value || '0', 10)
          : Number.parseFloat(value || '0'),
    }))
  }

  const loadLatestTelemetryIntoDraft = () => {
    if (!latestTelemetry) return
    const { created_at, ...restTelemetry } = latestTelemetry
    void created_at
    setTelemetryDraft(restTelemetry)
    setDraftHydratedFromLatest(true)
  }

  if (currentLoading) {
    return (
      <div className="page-shell">
        <PageIntro
          eyebrow="Adaptive Control"
          title="Policy console"
          description="Loading the current policy, latest telemetry and recent proposal history."
        />
      </div>
    )
  }

  return (
    <div className="page-shell">
      <PageIntro
        eyebrow="Adaptive Control"
        title="Policy console"
        description="The control plane for next-round thresholds and incentives. This is where telemetry becomes a proposal, then a validated policy, and finally the active strategy for later rounds."
        actions={
          proposal ? (
            <ProposalStatus blocked={blocked} approved={proposal.approved} />
          ) : (
            <StatusBadge label="No pending proposal" tone="slate" />
          )
        }
      />

      {policyReadError ? (
        <StateCallout
          icon={<AlertTriangle size={18} />}
          title="Policy console is partially disconnected"
          description="At least one policy-agent read failed. The page now keeps those sections explicit instead of silently rendering missing proposals or telemetry as normal empty state."
          tone="red"
        />
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Current round"
          value={currentPolicyError ? 'Unavailable' : currentPolicy ? `R${currentPolicy.round_id}` : 'N/A'}
          detail={currentPolicyError ? 'Policy agent current-policy read failed' : currentPolicy ? currentPolicy.policy_version : 'No active policy yet'}
          icon={<CircuitBoard size={22} />}
          tone={currentPolicyError ? 'red' : 'blue'}
        />
        <MetricTile
          label="Latest proposal"
          value={latestProposalError ? 'Unavailable' : proposal ? `R${proposal.round_id}` : 'None'}
          detail={latestProposalError ? 'Latest proposal endpoint unavailable' : proposalEngine.supporting}
          icon={<Bot size={22} />}
          tone={latestProposalError ? 'red' : 'teal'}
        />
        <MetricTile
          label="Fraud telemetry"
          value={latestTelemetryError ? 'Unavailable' : latestTelemetry ? `${(latestTelemetry.fraud_rate * 100).toFixed(1)}%` : 'N/A'}
          detail={latestTelemetryError ? 'Latest telemetry endpoint unavailable' : 'Recent fraud rate used for policy reasoning'}
          icon={<AlertTriangle size={22} />}
          tone="red"
        />
        <MetricTile
          label="Drift telemetry"
          value={latestTelemetryError ? 'Unavailable' : latestTelemetry ? latestTelemetry.golden_drift_score.toFixed(3) : 'N/A'}
          detail={latestTelemetryError ? 'Latest telemetry endpoint unavailable' : 'High drift should trigger review before punishment'}
          icon={<Sparkles size={22} />}
          tone={latestTelemetryError ? 'red' : 'amber'}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel
          title="Current policy"
          description="The frozen policy that L1, L2 and later rounds should be using right now."
          action={currentPolicy ? <StatusBadge label={`Round ${currentPolicy.round_id}`} tone="teal" /> : null}
        >
          {currentPolicyError ? (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Unable to load the current policy"
              description="The policy-agent current-policy endpoint is unavailable. This panel now reports that read failure directly."
              tone="red"
            />
          ) : currentPolicy ? (
            <div className="space-y-5">
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="rounded-[18px] border border-slate-900/8 bg-white/70 p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Policy version</div>
                  <div className="mt-2 font-mono text-sm text-slate-900">{currentPolicy.policy_version}</div>
                </div>
                <div className="rounded-[18px] border border-slate-900/8 bg-white/70 p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Effective from</div>
                  <div className="mt-2 font-mono text-sm text-slate-900">Round {currentPolicy.effective_from_round}</div>
                </div>
              </div>
              <ParameterGrid policy={currentPolicy} />
            </div>
          ) : (
            <p className="text-sm text-slate-500">No current policy stored yet.</p>
          )}
        </Panel>

        <Panel
          title="Latest proposal"
          description="Most recent candidate policy after generation, validation and safety-guard checks."
          action={proposal ? <ProposalStatus blocked={blocked} approved={proposal.approved} /> : null}
        >
          {latestProposalError ? (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Unable to load the latest proposal"
              description="The proposal feed from policy-agent is unavailable, so this section no longer falls back to a misleading “no pending proposal” message."
              tone="red"
            />
          ) : proposal ? (
            <div className="space-y-5">
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="rounded-[18px] border border-slate-900/8 bg-white/70 p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Proposed round</div>
                  <div className="mt-2 font-mono text-sm text-slate-900">{proposal.round_id}</div>
                </div>
                <div className="rounded-[18px] border border-slate-900/8 bg-white/70 p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Engine</div>
                  <div className="mt-2 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-base font-semibold text-slate-900">{proposalEngine.label}</span>
                      <StatusBadge label={proposal.llm_used ? 'GLM used' : 'Fallback'} tone={proposal.llm_used ? 'teal' : 'slate'} />
                    </div>
                    <p className="text-xs leading-5 text-slate-500">{proposalEngine.supporting}</p>
                    {proposalEngine.raw ? (
                      <div className="inline-flex max-w-full rounded-full bg-slate-100/80 px-3 py-1 text-[11px] leading-5 text-slate-500 ring-1 ring-slate-900/6">
                        <span className="truncate font-mono">{proposalEngine.raw}</span>
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>

              <ParameterGrid policy={proposal.proposed_policy} previousPolicy={proposal.current_policy} />

              {proposal.reasons.length > 0 ? (
                <div className="section-note space-y-2">
                  <h3 className="text-sm font-semibold text-slate-900">Reasons</h3>
                  {proposal.reasons.map((reason: string) => (
                    <p key={reason} className="text-sm leading-6 text-slate-600">{reason}</p>
                  ))}
                </div>
              ) : null}

              {proposal.validator_messages.length > 0 ? (
                <div className="section-note space-y-2">
                  <h3 className="text-sm font-semibold text-slate-900">Validator messages</h3>
                  {proposal.validator_messages.map((message: string) => (
                    <p key={message} className="text-sm leading-6 text-amber-700">{message}</p>
                  ))}
                </div>
              ) : null}

              {proposal.blocked_reasons.length > 0 ? (
                <div className="section-note space-y-2">
                  <h3 className="text-sm font-semibold text-slate-900">Blocked reasons</h3>
                  {proposal.blocked_reasons.map((reason: string) => (
                    <p key={reason} className="text-sm leading-6 text-rose-700">{reason}</p>
                  ))}
                </div>
              ) : null}

              {!blocked ? (
                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => activateMutation.mutate(proposal.round_id)}
                    disabled={activateMutation.isPending}
                    className="soft-button-primary"
                  >
                    {activateMutation.isPending ? 'Activating...' : 'Activate for next round'}
                  </button>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="text-sm text-slate-500">No pending proposal has been created yet.</p>
          )}
        </Panel>
      </div>

      {explanation ? (
        <Panel
          title="Stored explanation"
          description="Human-readable reasoning associated with the latest proposal."
        >
          <p className="text-sm leading-7 text-slate-600">{explanation}</p>
        </Panel>
      ) : null}

      <Panel
        title="Manual telemetry draft"
        description="Generate a next-round proposal from an editable telemetry snapshot. The form starts blank unless live telemetry has already been read from the policy agent."
      >
        {latestTelemetry ? (
          <div className="mb-5 rounded-[18px] border border-slate-900/8 bg-white/70 px-4 py-3 text-sm text-slate-600">
            Latest stored telemetry is available for round {latestTelemetry.round_id}. You can load it into the draft or continue editing manually.
          </div>
        ) : !latestTelemetryError ? (
          <div className="mb-5 rounded-[18px] border border-dashed border-slate-900/10 bg-white/60 px-4 py-3 text-sm text-slate-500">
            No latest telemetry snapshot is stored yet, so this form avoids prefilled scenario values and starts from a blank draft.
          </div>
        ) : null}

        <form
          className="space-y-5"
          onSubmit={(event) => {
            event.preventDefault()
            proposeMutation.mutate(telemetryDraft)
          }}
        >
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
            {telemetrySections.map((section) => (
              <div key={section.title} className="field-cluster">
                <div className="mb-4">
                  <h3 className="field-cluster-title">{section.title}</h3>
                  <p className="field-cluster-copy">{section.description}</p>
                </div>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-1">
                  {section.fields.map(({ key, label, step }) => (
                    <label key={key} className="block">
                      <span className="mb-2 block text-sm font-medium text-slate-600">{label}</span>
                      <input
                        type="number"
                        step={step ?? '0.01'}
                        value={telemetryDraft[key] as number}
                        onChange={(event) => handleTelemetryChange(key, event.target.value)}
                        className="soft-input"
                      />
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-3">
            <button type="submit" disabled={proposeMutation.isPending} className="soft-button-primary">
              {proposeMutation.isPending ? 'Proposing...' : 'Generate proposal'}
            </button>
            <button
              type="button"
              onClick={loadLatestTelemetryIntoDraft}
              disabled={!latestTelemetry}
              className="soft-button-secondary"
            >
              Load latest telemetry
            </button>
            <button
              type="button"
              onClick={() => setTelemetryDraft(createBlankTelemetryDraft())}
              className="soft-button-secondary"
            >
              Clear draft
            </button>
          </div>

          {proposeMutation.isError ? (
            <p className="text-sm text-rose-700">Failed to generate proposal. Check policy-agent availability.</p>
          ) : null}
        </form>
      </Panel>

      <Panel
        title="Policy history"
        description="Recent frozen policies stored in Redis history for auditability and comparison."
      >
        {policyHistoryError ? (
          <StateCallout
            icon={<AlertTriangle size={18} />}
            title="Unable to load policy history"
            description="The policy history endpoint is unavailable. This table now reserves the empty state for a genuinely empty history."
            tone="red"
          />
        ) : historyRows.length > 0 ? (
          <div className="table-shell overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="table-header-row border-b border-slate-900/10">
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Round</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Version</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Theta Tol</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Theta Rare</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Theta Drift</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Honest Reward</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Slash</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Beneficial Rarity Reward</th>
                </tr>
              </thead>
              <tbody>
                {historyRows.map((policy) => (
                  <tr key={policy.round_id} className="border-b border-slate-900/8 last:border-b-0 hover:bg-white/60">
                    <td className="px-5 py-4 font-mono text-sm text-slate-900">{policy.round_id}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-600">{policy.policy_version}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-700">{formatPolicyNumber(policy.theta_tol)}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-700">{formatPolicyNumber(policy.theta_rare)}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-700">{formatPolicyNumber(policy.theta_drift)}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-700">{formatPolicyNumber(policy.honest_reward_multiplier)}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-700">{formatPolicyNumber(policy.slash_multiplier)}</td>
                    <td className="px-5 py-4 font-mono text-sm text-slate-700">{formatPolicyNumber(policy.rarity_reward_multiplier)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex min-h-[200px] flex-col items-center justify-center gap-4 rounded-[20px] border border-dashed border-slate-900/10 bg-white/60 p-6 text-center">
            <div className="metric-icon-wrap bg-slate-100/80 text-slate-500">
              <History size={18} />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900">No history yet</h3>
              <p className="max-w-md text-sm leading-6 text-slate-500">Once proposals are activated, this table will show how thresholds and incentives drift across rounds.</p>
            </div>
          </div>
        )}
      </Panel>

      <GLMDecisionLog />
    </div>
  )
}
