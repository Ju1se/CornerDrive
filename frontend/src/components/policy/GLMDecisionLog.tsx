import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, RefreshCcw } from 'lucide-react'

import { fetchGLMDecisions } from '../../services/api'
import type { GLMDecision } from '../../types/policy'
import { Panel, StatusBadge } from '../ui/primitives'

const PARAM_LABELS: Record<string, string> = {
  theta_tol: 'Theta Tol',
  theta_rare: 'Theta Rare',
  theta_drift: 'Theta Drift',
  recheck_probability: 'Recheck Probability',
  honest_reward_multiplier: 'Honest Reward',
  slash_multiplier: 'Slash Multiplier',
  rarity_reward_multiplier: 'Beneficial Rarity Reward',
  corner_weight: 'Corner Weight',
  cosine_filter_threshold: 'Cosine Filter',
}

function formatRelativeTime(timestamp: string) {
  const createdAt = new Date(timestamp)
  const diffMinutes = Math.floor((Date.now() - createdAt.getTime()) / 60000)

  if (diffMinutes < 1) return 'Just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffMinutes < 1440) return `${Math.floor(diffMinutes / 60)}h ago`
  return createdAt.toLocaleDateString()
}

function formatNumber(value: number) {
  return Math.abs(value) < 0.1 ? value.toFixed(4) : value.toFixed(2)
}

function ChangeChip({
  param,
  before,
  after,
}: {
  param: string
  before: number
  after: number
}) {
  const increased = after > before
  return (
    <span className="rounded-full bg-white/80 px-3 py-1.5 text-xs text-slate-700 ring-1 ring-slate-900/8">
      <span className="font-semibold">{PARAM_LABELS[param] || param}</span>{' '}
      <span className={increased ? 'text-rose-700' : 'text-emerald-700'}>
        {formatNumber(before)} {'->'} {formatNumber(after)}
      </span>
    </span>
  )
}

function DecisionItem({
  decision,
  expanded,
  onToggle,
}: {
  decision: GLMDecision
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <div className="rounded-[22px] border border-slate-900/8 bg-white/70">
      <button
        type="button"
        onClick={onToggle}
        className="w-full px-5 py-5 text-left"
      >
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-teal-700 text-sm font-semibold text-white">
                R{decision.round_id}
              </span>
              <StatusBadge label={decision.llm_used ? 'GLM' : 'Fallback'} tone={decision.llm_used ? 'teal' : 'slate'} />
              <StatusBadge label={decision.blocked ? 'Blocked' : 'Accepted'} tone={decision.blocked ? 'red' : 'blue'} />
              <StatusBadge label={`${decision.parameters_changed.length} changes`} tone="amber" />
            </div>

            {!expanded && decision.parameters_changed.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {decision.parameters_changed.slice(0, 3).map((change) => (
                  <ChangeChip key={`${decision.round_id}-${change.param}`} {...change} />
                ))}
              </div>
            ) : null}
          </div>

          <div className="flex items-center justify-between gap-4 lg:flex-col lg:items-end">
            <span className="text-sm text-slate-500">{formatRelativeTime(decision.timestamp)}</span>
            <ChevronDown
              size={18}
              className={`text-slate-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
            />
          </div>
        </div>
      </button>

      {expanded ? (
        <div className="space-y-5 border-t border-slate-900/8 bg-[#fffaf3] px-5 py-5">
          <div>
            <h4 className="mb-3 text-sm font-semibold text-slate-900">Parameter changes</h4>
            <div className="flex flex-wrap gap-2">
              {decision.parameters_changed.length > 0 ? (
                decision.parameters_changed.map((change) => (
                  <ChangeChip key={`${decision.round_id}-${change.param}`} {...change} />
                ))
              ) : (
                <span className="text-sm text-slate-500">No parameter changes in this round.</span>
              )}
            </div>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-slate-900">Reasons</h4>
            <div className="space-y-2">
              {decision.reasons.length > 0 ? (
                decision.reasons.map((reason) => (
                  <p key={reason} className="text-sm leading-6 text-slate-600">{reason}</p>
                ))
              ) : (
                <p className="text-sm text-slate-500">No explanation stored.</p>
              )}
            </div>
          </div>

          {decision.validator_messages.length > 0 ? (
            <div>
              <h4 className="mb-3 text-sm font-semibold text-slate-900">Validator messages</h4>
              <div className="space-y-2">
                {decision.validator_messages.map((message) => (
                  <p key={message} className="text-sm leading-6 text-amber-700">{message}</p>
                ))}
              </div>
            </div>
          ) : null}

          {decision.telemetry ? (
            <div>
              <h4 className="mb-3 text-sm font-semibold text-slate-900">Telemetry snapshot</h4>
              <div className="grid gap-3 md:grid-cols-5">
                {[
                  { label: 'Fraud Rate', value: `${(decision.telemetry.fraud_rate * 100).toFixed(2)}%` },
                  { label: 'Beneficial Rarity Rate', value: `${(decision.telemetry.rarity_rate * 100).toFixed(2)}%` },
                  { label: 'Honest Rate', value: `${(decision.telemetry.honest_rate * 100).toFixed(2)}%` },
                  { label: 'Main Accuracy', value: `${(decision.telemetry.main_accuracy * 100).toFixed(2)}%` },
                  { label: 'Corner Accuracy', value: `${(decision.telemetry.corner_accuracy * 100).toFixed(2)}%` },
                ].map((item) => (
                  <div key={item.label} className="rounded-[18px] border border-slate-900/8 bg-white/70 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-slate-500">{item.label}</div>
                    <div className="mt-2 text-sm font-semibold text-slate-900">{item.value}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export default function GLMDecisionLog() {
  const [expandedRound, setExpandedRound] = useState<number | null>(null)
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['glmDecisions'],
    queryFn: fetchGLMDecisions,
    refetchInterval: 30000,
  })

  return (
    <Panel
      title="GLM decision log"
      description="A trace of recent proposals, including whether they came from GLM reasoning or deterministic fallback."
      action={
        <button type="button" onClick={() => refetch()} className="soft-button-secondary">
          <RefreshCcw size={14} className="mr-2 inline-block" />
          Refresh
        </button>
      }
    >
      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <div key={index} className="h-28 animate-pulse rounded-[22px] bg-slate-200/60" />
          ))}
        </div>
      ) : isError ? (
        <div className="rounded-[20px] border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-700">
          Failed to load GLM decision history.
        </div>
      ) : data && data.data.length > 0 ? (
        <div className="space-y-3">
          {data.data.map((decision) => (
            <DecisionItem
              key={decision.round_id}
              decision={decision}
              expanded={expandedRound === decision.round_id}
              onToggle={() =>
                setExpandedRound((current) => (current === decision.round_id ? null : decision.round_id))
              }
            />
          ))}
        </div>
      ) : (
        <div className="rounded-[20px] border border-dashed border-slate-900/10 bg-white/60 px-5 py-10 text-center text-sm text-slate-500">
          No policy decisions have been recorded yet.
        </div>
      )}
    </Panel>
  )
}
