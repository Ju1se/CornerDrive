import { useDeferredValue, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, ClipboardCheck, Search, ShieldAlert, Sparkles, Waves } from 'lucide-react'

import { fetchCurrentPolicy, fetchRecentAudits } from '../services/api'
import { AuditDecisionMap } from '../components/audits/AuditDecisionMap'
import { ClassificationBadge, EmptyState, MetricTile, PageIntro, Panel, StateCallout } from '../components/ui/primitives'
import { formatClassificationLabel } from '../utils/classification'

type VerdictFilter = 'ALL' | 'FRAUD' | 'RARITY' | 'HONEST' | 'NOISE'

function classifyAgainstPolicy(
  deltaMain: number,
  deltaCorner: number,
  thetaTol: number,
  thetaRare: number,
) {
  if (deltaMain > thetaTol) return 'FRAUD'
  if (deltaCorner <= thetaRare && deltaMain <= 0) return 'RARITY'
  if (deltaMain < 0) return 'HONEST'
  return 'NOISE'
}

function formatVerdictOption(option: VerdictFilter) {
  if (option === 'ALL') return 'All verdicts'
  return formatClassificationLabel(option)
}

export default function Audits() {
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>('ALL')
  const [searchValue, setSearchValue] = useState('')
  const deferredSearch = useDeferredValue(searchValue.trim().toLowerCase())

  const { data: audits, isLoading, isError } = useQuery({
    queryKey: ['audits'],
    queryFn: () => fetchRecentAudits(50),
    refetchInterval: 5000,
  })
  const {
    data: currentPolicy,
    isError: currentPolicyError,
  } = useQuery({
    queryKey: ['policy', 'current'],
    queryFn: fetchCurrentPolicy,
    refetchInterval: 10000,
  })

  const overallSummary = useMemo(() => {
    const items = audits ?? []
    return items.reduce(
      (acc, audit) => {
        acc.total += 1
        acc[audit.classification] += 1
        acc.netSbt += audit.sbt_points
        return acc
      },
      {
        total: 0,
        FRAUD: 0,
        RARITY: 0,
        HONEST: 0,
        NOISE: 0,
        netSbt: 0,
      },
    )
  }, [audits])

  const filteredAudits = useMemo(() => {
    let items = audits ?? []

    if (verdictFilter !== 'ALL') {
      items = items.filter((audit) => audit.classification === verdictFilter)
    }

    if (deferredSearch) {
      items = items.filter((audit) => audit.vehicle_id.toLowerCase().includes(deferredSearch))
    }

    return items
  }, [audits, deferredSearch, verdictFilter])

  const summary = useMemo(() => {
    return filteredAudits.reduce(
      (acc, audit) => {
        acc.total += 1
        acc[audit.classification] += 1
        acc.netSbt += audit.sbt_points
        return acc
      },
      {
        total: 0,
        FRAUD: 0,
        RARITY: 0,
        HONEST: 0,
        NOISE: 0,
        netSbt: 0,
      },
    )
  }, [filteredAudits])

  const boundaryAgreement = useMemo(() => {
    if (!currentPolicy || filteredAudits.length === 0) return null

    const matches = filteredAudits.filter(
      (audit) =>
        classifyAgainstPolicy(
          audit.delta_loss_main,
          audit.delta_loss_corner,
          currentPolicy.theta_tol,
          currentPolicy.theta_rare,
        ) === audit.classification,
    ).length

    return matches / filteredAudits.length
  }, [currentPolicy, filteredAudits])

  return (
    <div className="page-shell">
      <PageIntro
        eyebrow="Inspection"
        title="Audit stream"
        description="The last 50 live L2 decisions, including delta-loss evidence and the immediate incentive effect on each contributor. This is where beneficial rarity should be visibly separated from fraud."
        actions={
          !isError ? (
            <>
              {verdictFilter === 'ALL' ? (
                <span className="rounded-full bg-white/70 px-3 py-1.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-900/8">
                  All verdicts
                </span>
              ) : (
                <ClassificationBadge type={verdictFilter} compact />
              )}
              <span className="rounded-full bg-white/70 px-3 py-1.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-900/8">
                {summary.total} visible / {overallSummary.total} total
              </span>
            </>
          ) : null
        }
      />

      {isError ? (
        <StateCallout
          icon={<AlertTriangle size={18} />}
          title="Audit stream unavailable"
          description="The frontend could not reach the L4 recent-audits endpoint, so this page now surfaces the outage instead of showing a misleading empty table."
          tone="red"
        />
      ) : null}

      <Panel
        title="Focus controls"
        description="Filter the verdict stream by classification and search specific contributors before presenting the decision map."
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,300px)_minmax(0,1fr)]">
          <label className="field-cluster">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <Search size={16} />
              Vehicle search
            </div>
            <p className="field-cluster-copy">Paste part of an address to narrow the visible window to one contributor.</p>
            <input
              value={searchValue}
              onChange={(event) => setSearchValue(event.target.value)}
              className="soft-input mt-4"
              placeholder="0xabc..."
            />
          </label>

          <div className="field-cluster">
            <p className="field-cluster-title">Verdict focus</p>
            <p className="field-cluster-copy">Useful during demos when you want to isolate beneficial-rarity discoveries or fraud penalties.</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {(['ALL', 'FRAUD', 'RARITY', 'HONEST', 'NOISE'] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setVerdictFilter(option)}
                  className={option === verdictFilter ? 'soft-button-primary' : 'soft-button-secondary'}
                >
                  {formatVerdictOption(option)}
                </button>
              ))}
            </div>
            <div className="section-note mt-4">
              Showing {summary.total} audit points out of {overallSummary.total} visible in the current recent-audit window.
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Recent verdicts"
          value={isError ? 'N/A' : `${summary.total}`}
          detail={isError ? 'L4 recent-audits endpoint unavailable' : 'Latest audits visible from L4 recent-audits'}
          icon={<ClipboardCheck size={22} />}
          tone={isError ? 'red' : 'blue'}
        />
        <MetricTile
          label="Fraud flagged"
          value={isError ? 'N/A' : `${summary.FRAUD}`}
          detail={isError ? 'Fraud counts unavailable while L4 is offline' : 'Updates that crossed the main-task harm threshold'}
          icon={<ShieldAlert size={22} />}
          tone="red"
        />
        <MetricTile
          label="Beneficial rarity"
          value={isError ? 'N/A' : `${summary.RARITY}`}
          detail={isError ? 'Beneficial-rarity counts unavailable while L4 is offline' : 'Updates that improved corner cases without harming the main task'}
          icon={<Sparkles size={22} />}
          tone={isError ? 'red' : 'teal'}
        />
        <MetricTile
          label="Net SBT impact"
          value={isError ? 'N/A' : `${summary.netSbt > 0 ? '+' : ''}${summary.netSbt}`}
          detail={isError ? 'Recent SBT deltas unavailable' : 'Positive means the recent window rewarded more than it slashed'}
          icon={<Waves size={22} />}
          tone={isError ? 'red' : 'amber'}
        />
      </div>

      {!isError && currentPolicy ? (
        <div className="grid gap-4 md:grid-cols-3">
          <MetricTile
            label="Overlay agreement"
            value={boundaryAgreement !== null ? `${(boundaryAgreement * 100).toFixed(1)}%` : 'N/A'}
            detail="How many visible points still match the current policy boundary overlay."
            icon={<ClipboardCheck size={22} />}
            tone={boundaryAgreement === null ? 'slate' : boundaryAgreement >= 0.8 ? 'teal' : 'amber'}
          />
          <MetricTile
            label="θ_tol"
            value={currentPolicy.theta_tol.toFixed(4)}
            detail="Current fraud boundary used in the overlay."
            icon={<ShieldAlert size={22} />}
            tone="red"
          />
          <MetricTile
            label="θ_rare"
            value={currentPolicy.theta_rare.toFixed(4)}
            detail="Current beneficial-rarity boundary used in the overlay."
            icon={<Sparkles size={22} />}
            tone="teal"
          />
        </div>
      ) : null}

      <Panel
        title="Fraud vs beneficial-rarity decision map"
        description="A live scatter view of recent audits. The point color is the stored L2 verdict; the threshold overlay comes from the current active policy."
      >
        {isLoading ? (
          <div className="grid gap-3">
            {Array.from({ length: 2 }).map((_, index) => (
              <div key={index} className="h-48 animate-pulse rounded-[24px] bg-slate-200/60" />
            ))}
          </div>
        ) : isError ? (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Decision map unavailable"
              description="The map needs the recent-audits feed from L4. Once that endpoint comes back, this panel will show how fraud, beneficial rarity, honest and noise decisions fall against the active thresholds."
              tone="red"
            />
          ) : filteredAudits.length > 0 ? (
            <AuditDecisionMap
              audits={filteredAudits}
              policy={currentPolicy ?? null}
              policyUnavailable={currentPolicyError || currentPolicy === null}
            />
          ) : (
            <EmptyState
              icon={<Search size={20} />}
              title="No audit points match the current focus"
              description="Clear the search box or switch the verdict filter back to All to repopulate the map."
            />
          )}
        </Panel>

      <Panel
        title="Recent L2 verdict table"
        description="Use this as the quickest truth source for whether beneficial rarity and fraud are both surfacing in the live demo."
      >
        {isLoading ? (
          <div className="grid gap-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="h-16 animate-pulse rounded-[18px] bg-slate-200/60" />
            ))}
          </div>
        ) : isError ? (
          <StateCallout
            icon={<AlertTriangle size={18} />}
            title="Unable to load recent verdicts"
            description="Check whether the L4 dashboard service is running and reachable. This panel no longer treats a transport failure as “no audit records yet.”"
            tone="red"
          />
        ) : filteredAudits.length > 0 ? (
          <div className="table-shell overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="table-header-row border-b border-slate-900/10">
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Vehicle</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Verdict</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">ΔL main</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">ΔL corner</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">SBT</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Observed</th>
                </tr>
              </thead>
              <tbody>
                {filteredAudits.map((audit, index) => (
                  <tr
                    key={`${audit.vehicle_id}-${audit.timestamp}-${index}`}
                    className="border-b border-slate-900/8 last:border-b-0 hover:bg-white/60"
                  >
                    <td className="px-5 py-4 align-top">
                      <div className="space-y-1">
                        <p className="break-all font-mono text-xs text-slate-900 md:text-sm">{audit.vehicle_id}</p>
                        <p className="text-xs text-slate-500">Gradient contribution record</p>
                      </div>
                    </td>
                    <td className="px-5 py-4 align-top">
                      <ClassificationBadge type={audit.classification} />
                    </td>
                    <td className="px-5 py-4 align-top font-mono text-sm text-slate-700">
                      {audit.delta_loss_main.toFixed(6)}
                    </td>
                    <td className="px-5 py-4 align-top font-mono text-sm text-slate-700">
                      {audit.delta_loss_corner.toFixed(6)}
                    </td>
                    <td
                      className={`px-5 py-4 align-top text-sm font-semibold ${
                        audit.sbt_points > 0
                          ? 'text-emerald-700'
                          : audit.sbt_points < 0
                            ? 'text-rose-700'
                            : 'text-slate-600'
                      }`}
                    >
                      {audit.sbt_points > 0 ? '+' : ''}
                      {audit.sbt_points}
                    </td>
                    <td className="px-5 py-4 align-top text-sm text-slate-500">
                      {new Date(audit.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={<Search size={20} />}
            title="No rows match the current focus"
            description="This table is filtered. Reset the verdict focus or clear the vehicle search to bring rows back."
          />
        )}
      </Panel>
    </div>
  )
}
