import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  AlertTriangle,
  Award,
  CarFront,
  Coins,
  Network,
  Shield,
  Sparkles,
} from 'lucide-react'

import { fetchL1Health, fetchL3Status, fetchL4Health, fetchPolicyHealth, fetchRecentAudits, fetchSystemStats, fetchTierDistribution } from '../services/api'
import { ClassificationBadge, MetricBar, MetricTile, PageIntro, Panel, StateCallout, StatusBadge } from '../components/ui/primitives'

function toneFromStatus(status?: string) {
  const normalized = normalizeStatus(status)
  if (!normalized) return 'red'
  if (normalized === 'healthy' || normalized === 'online') return 'teal'
  if (normalized === 'degraded') return 'amber'
  return 'red'
}

function formatStatus(status?: string) {
  const normalized = normalizeStatus(status)
  if (!normalized) return 'Offline'
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

function normalizeStatus(status?: string) {
  if (!status) return undefined
  if (status === 'ok') return 'healthy'
  return status
}

function toneFromL3Source(source?: string) {
  if (source === 'disk') return 'teal'
  if (source === 'placeholder') return 'amber'
  return 'red'
}

function labelFromL3Source(source?: string) {
  if (source === 'disk') return 'Disk-backed'
  if (source === 'placeholder') return 'Placeholder'
  if (source === 'invalid_artifacts') return 'Invalid dataset'
  return 'Unknown'
}

function TierDonut({
  data,
}: {
  data: Array<{ name: string; value: number; color: string }>
}) {
  const total = data.reduce((sum, item) => sum + item.value, 0)
  const safeTotal = total > 0 ? total : 1

  let start = 0
  const segments = data.map((item) => {
    const portion = item.value / safeTotal
    const end = start + portion * 360
    const segment = `${item.color} ${start.toFixed(2)}deg ${end.toFixed(2)}deg`
    start = end
    return segment
  })

  const background = total > 0
    ? `conic-gradient(${segments.join(', ')})`
    : 'conic-gradient(#cbd5e1 0deg 360deg)'

  return (
    <div className="relative mx-auto h-[220px] w-[220px] md:h-[236px] md:w-[236px]">
      <div
        className="absolute inset-0 rounded-full border border-slate-900/8 shadow-inner"
        style={{ background }}
      />
      <div className="absolute inset-[25%] flex flex-col items-center justify-center rounded-full border border-white/60 bg-[#fffaf3] px-3 text-center shadow-sm">
        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Tiers</span>
        <span className="mt-2 text-3xl font-bold text-slate-900">{total}</span>
        <span className="mt-1 text-xs leading-5 text-slate-500 md:text-sm">Visible vehicles</span>
      </div>
    </div>
  )
}

function formatCompactEth(value: number) {
  if (Math.abs(value) >= 1000) {
    return `${(value / 1000).toFixed(3)}k`
  }

  return value.toFixed(3)
}

function truncateMiddle(value: string, head = 18, tail = 8) {
  if (value.length <= head + tail + 1) {
    return value
  }

  return `${value.slice(0, head)}...${value.slice(-tail)}`
}

function formatAuditTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function SnapshotCard({
  label,
  value,
  detail,
  unit,
  title,
}: {
  label: string
  value: string
  detail: string
  unit?: string
  title?: string
}) {
  return (
    <div className="h-full rounded-[20px] border border-slate-900/8 bg-white/70 p-4 md:p-5">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <div className="mt-4 flex flex-wrap items-end gap-2">
        <p
          className="min-w-0 max-w-full break-words font-mono text-[clamp(1.7rem,2.5vw,2.6rem)] font-bold leading-none tracking-[-0.05em] text-slate-900"
          title={title}
        >
          {value}
        </p>
        {unit ? (
          <span className="pb-1 text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
            {unit}
          </span>
        ) : null}
      </div>
      <p className="mt-3 max-w-[24ch] text-sm leading-7 text-slate-600">{detail}</p>
    </div>
  )
}

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading, isError: statsError } = useQuery({
    queryKey: ['systemStats'],
    queryFn: fetchSystemStats,
    refetchInterval: 5000,
  })

  const { data: tiers, isError: tiersError } = useQuery({
    queryKey: ['tierDistribution'],
    queryFn: fetchTierDistribution,
    refetchInterval: 10000,
  })

  const { data: recentAudits, isError: recentAuditsError } = useQuery({
    queryKey: ['recentAudits'],
    queryFn: () => fetchRecentAudits(12),
    refetchInterval: 5000,
  })

  const { data: l1Health, isError: l1Error } = useQuery({
    queryKey: ['health', 'l1'],
    queryFn: fetchL1Health,
    refetchInterval: 15000,
    retry: 1,
  })

  const { data: l4Health, isError: l4Error } = useQuery({
    queryKey: ['health', 'l4'],
    queryFn: fetchL4Health,
    refetchInterval: 15000,
    retry: 1,
  })

  const { data: policyHealth, isError: policyError } = useQuery({
    queryKey: ['health', 'policy'],
    queryFn: fetchPolicyHealth,
    refetchInterval: 15000,
    retry: 1,
  })

  const { data: l3Status, isError: l3Error } = useQuery({
    queryKey: ['l3Status'],
    queryFn: fetchL3Status,
    refetchInterval: 15000,
    retry: 1,
  })

  const statsUnavailable = statsError && !stats
  const tiersUnavailable = tiersError && !tiers
  const recentAuditsUnavailable = recentAuditsError && !recentAudits

  const tierData = useMemo(
    () => [
      { name: 'Bronze', value: tiers?.bronze ?? 0, color: '#b98552' },
      { name: 'Silver', value: tiers?.silver ?? 0, color: '#94a3b8' },
      { name: 'Gold', value: tiers?.gold ?? 0, color: '#d9a441' },
      { name: 'Platinum', value: tiers?.platinum ?? 0, color: '#0f766e' },
    ],
    [tiers],
  )

  const auditMix = useMemo(() => {
    if (!stats || stats.total_audits === 0) {
      return [
        { label: 'Fraud', value: 0, colorClass: 'bg-rose-500' },
        { label: 'Beneficial rarity', value: 0, colorClass: 'bg-emerald-500' },
        { label: 'Honest', value: 0, colorClass: 'bg-sky-500' },
        { label: 'Noise', value: 0, colorClass: 'bg-slate-500' },
      ]
    }

    return [
      { label: 'Fraud', value: (stats.fraud_count / stats.total_audits) * 100, colorClass: 'bg-rose-500' },
      { label: 'Beneficial rarity', value: (stats.rare_count / stats.total_audits) * 100, colorClass: 'bg-emerald-500' },
      { label: 'Honest', value: (stats.honest_count / stats.total_audits) * 100, colorClass: 'bg-sky-500' },
      { label: 'Noise', value: (stats.noise_count / stats.total_audits) * 100, colorClass: 'bg-slate-500' },
    ]
  }, [stats])

  const layerCards = [
    {
      name: 'L1 Linear Defense',
      label: l1Error ? 'Offline' : formatStatus(l1Health?.status),
      tone: (l1Error ? 'red' : toneFromStatus(l1Health?.status)) as 'teal' | 'amber' | 'red' | 'blue' | 'slate',
      detail: 'Cosine deviation screening and suspect routing',
    },
    {
      name: 'L2 Dual Audit',
      label: statsUnavailable || recentAuditsUnavailable ? 'Not confirmed' : 'Observed via L4',
      tone: (statsUnavailable || recentAuditsUnavailable ? 'amber' : 'slate') as 'teal' | 'amber' | 'red' | 'blue' | 'slate',
      detail: 'Redis-backed audit workers separate fraud from beneficial rarity, but there is no standalone health endpoint yet.',
    },
    {
      name: 'L3 Gatekeeper',
      label: l3Error ? 'Unknown' : labelFromL3Source(l3Status?.dataset_source),
      tone: (l3Error ? 'red' : toneFromL3Source(l3Status?.dataset_source)) as 'teal' | 'amber' | 'red' | 'blue' | 'slate',
      detail: l3Error
        ? 'Unable to read live L3 dataset status from L4.'
        : l3Status
          ? `${l3Status.sample_count ?? 0} samples, θ_drift ${l3Status.drift_threshold.toFixed(4)}. ${l3Status.detail}`
          : 'Implemented as validation logic, not deployed as a standalone service.',
    },
    {
      name: 'L4 Settlement',
      label: l4Error ? 'Offline' : formatStatus(l4Health?.status),
      tone: (l4Error ? 'red' : toneFromStatus(l4Health?.status)) as 'teal' | 'amber' | 'red' | 'blue' | 'slate',
      detail: 'Dashboard + settlement API, with chain readiness reflected in health',
    },
    {
      name: 'Policy Agent',
      label: policyError ? 'Offline' : formatStatus(policyHealth?.status),
      tone: (policyError ? 'red' : toneFromStatus(policyHealth?.status)) as 'teal' | 'amber' | 'red' | 'blue' | 'slate',
      detail: 'Adaptive proposal engine with hybrid GLM routing',
    },
  ]

  if (statsLoading) {
    return (
      <div className="page-shell">
        <PageIntro
          eyebrow="Operations"
          title="FLPG live dashboard"
          description="Loading the current telemetry snapshot, audit stream and layer status."
        />
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="metric-tile animate-pulse">
              <div className="h-4 w-24 rounded bg-slate-200" />
              <div className="mt-4 h-10 w-28 rounded bg-slate-300" />
              <div className="mt-3 h-4 w-40 rounded bg-slate-200" />
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="page-shell">
      <PageIntro
        eyebrow="Operations"
        title="FLPG live dashboard"
        description="A real-time view across intake, audit, policy control and settlement. The goal here is to see whether beneficial rare gradients are preserved while fraud pressure stays bounded."
        actions={
          <>
            <StatusBadge
              label={
                statsUnavailable
                  ? 'L4 telemetry unavailable'
                  : stats && stats.total_audits > 0
                    ? `${stats.total_audits} audits observed`
                    : 'Awaiting audits'
              }
              tone={statsUnavailable ? 'red' : stats && stats.total_audits > 0 ? 'teal' : 'amber'}
              pulse
            />
            <StatusBadge
              label={
                policyError
                  ? 'Policy offline'
                  : normalizeStatus(policyHealth?.status) === 'healthy'
                    ? 'Adaptive policy live'
                    : 'Policy attention needed'
              }
              tone={(policyError ? 'red' : toneFromStatus(policyHealth?.status)) as 'teal' | 'amber' | 'red' | 'blue' | 'slate'}
            />
          </>
        }
      />

      {statsUnavailable || tiersUnavailable || recentAuditsUnavailable ? (
        <StateCallout
          icon={<AlertTriangle size={18} />}
          title="Some dashboard panels are disconnected from the backend"
          description="This page now shows backend outages directly instead of substituting zeroes or empty lists that look like valid telemetry."
          tone="red"
        />
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Participating vehicles"
          value={stats ? `${stats.total_vehicles}` : 'N/A'}
          detail={stats ? 'Registered or recently seen contributors' : 'L4 stats endpoint unavailable'}
          icon={<CarFront size={22} />}
          tone={stats ? 'blue' : 'red'}
        />
        <MetricTile
          label="Audits completed"
          value={stats ? `${stats.total_audits}` : 'N/A'}
          detail={stats ? 'L2 verdicts written to Redis and surfaced to L4' : 'Unable to read /api/v1/stats from L4'}
          icon={<Shield size={22} />}
          tone={stats ? 'teal' : 'red'}
        />
        <MetricTile
          label="Fraud detections"
          value={stats ? `${stats.fraud_count}` : 'N/A'}
          detail={stats ? `${(stats.fraud_rate * 100).toFixed(2)}% of observed audits` : 'Fraud telemetry unavailable'}
          icon={<AlertTriangle size={22} />}
          tone="red"
        />
        <MetricTile
          label="Beneficial rarity"
          value={stats ? `${stats.rare_count}` : 'N/A'}
          detail={stats ? 'Beneficial rare updates preserved by L2' : 'Beneficial-rarity telemetry unavailable'}
          icon={<Sparkles size={22} />}
          tone={stats ? 'amber' : 'red'}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.14fr)_minmax(320px,0.86fr)]">
        <Panel
          title="Audit composition"
          description="How the system is currently classifying incoming suspect gradients."
          className="h-full"
        >
          {stats ? (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,0.88fr)_minmax(380px,1.12fr)] xl:items-start">
              <div className="space-y-4">
                {auditMix.map((item) => (
                  <MetricBar key={item.label} label={item.label} value={item.value} colorClass={item.colorClass} />
                ))}
              </div>

              <div className="grid auto-rows-fr gap-3 md:grid-cols-2">
                <SnapshotCard
                  label="Reward flow"
                  value={formatCompactEth(stats.total_rewards_distributed)}
                  unit="ETH"
                  title={`${stats.total_rewards_distributed.toFixed(3)} ETH`}
                  detail="Aggregate rewards recorded by settlement."
                />
                <SnapshotCard
                  label="Slashing pressure"
                  value={formatCompactEth(stats.total_slashed)}
                  unit="ETH"
                  title={`${stats.total_slashed.toFixed(3)} ETH`}
                  detail="Penalty outflow attributed to fraud verdicts."
                />
                <SnapshotCard
                  label="Beneficial share"
                  value={stats.total_audits > 0 ? `${(((stats.honest_count + stats.rare_count) / stats.total_audits) * 100).toFixed(1)}%` : '0.0%'}
                  detail="Recent audits that helped the main task or preserved useful corner-case signal."
                />
                <SnapshotCard
                  label="Beneficial-rarity density"
                  value={stats.total_audits > 0 ? `${((stats.rare_count / stats.total_audits) * 100).toFixed(1)}%` : '0.0%'}
                  detail="Useful corner-case updates that reached a rewarded beneficial-rarity verdict."
                />
              </div>
            </div>
          ) : (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Unable to load audit composition"
              description="The L4 `/api/v1/stats` endpoint did not respond, so this chart is intentionally withheld instead of showing fake zeroes."
              tone="red"
            />
          )}
        </Panel>

        <Panel
          title="Tier distribution"
          description="SBT tiers currently visible from L4 vehicle state."
          className="h-full"
        >
          {tiers ? (
            <div className="flex h-full flex-col gap-6">
              <TierDonut data={tierData} />

              <div className="grid gap-3 sm:grid-cols-2">
                {tierData.map((tier) => (
                  <div key={tier.name} className="flex items-center justify-between gap-4 rounded-[18px] border border-slate-900/8 bg-white/70 px-4 py-3">
                    <div className="flex min-w-0 items-center gap-3">
                      <span className="h-3 w-3 rounded-full" style={{ backgroundColor: tier.color }} />
                      <span className="truncate font-medium text-slate-700">{tier.name}</span>
                    </div>
                    <span className="shrink-0 font-mono text-slate-900">{tier.value}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Unable to load tier distribution"
              description="The vehicle tier chart now waits for a real backend response instead of substituting an all-zero donut."
              tone="red"
            />
          )}
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Panel
          title="Recent audit stream"
          description="Latest verdicts arriving from the L2 worker queue."
        >
          <div className="space-y-3">
            {recentAuditsUnavailable ? (
              <StateCallout
                icon={<AlertTriangle size={18} />}
                title="Unable to load the recent audit stream"
                description="The L4 recent-audits endpoint is unavailable, so this page no longer disguises the outage as an empty verdict list."
                tone="red"
              />
            ) : recentAudits && recentAudits.length > 0 ? (
              recentAudits.map((audit, index) => (
                <div
                  key={`${audit.vehicle_id}-${audit.timestamp}-${index}`}
                  className="grid gap-3 rounded-[20px] border border-slate-900/8 bg-white/70 px-4 py-4 md:grid-cols-[minmax(0,1fr)_auto_auto] md:items-center md:gap-4"
                >
                  <div className="min-w-0 space-y-1">
                    <p className="truncate font-mono text-sm text-slate-900" title={audit.vehicle_id}>
                      {truncateMiddle(audit.vehicle_id)}
                    </p>
                    <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
                      <span>Δmain {audit.delta_loss_main.toFixed(6)}</span>
                      <span>Δcorner {audit.delta_loss_corner.toFixed(6)}</span>
                      <span>SBT {audit.sbt_points > 0 ? '+' : ''}{audit.sbt_points}</span>
                    </div>
                  </div>
                  <span className="shrink-0 whitespace-nowrap text-sm text-slate-500 md:text-right">
                    {formatAuditTime(audit.timestamp)}
                  </span>
                  <div className="shrink-0">
                    <ClassificationBadge type={audit.classification} />
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-[20px] border border-dashed border-slate-900/10 bg-white/60 px-5 py-10 text-center text-slate-500">
                Waiting for live gradients to reach L2 and appear in the L4 recent-audits stream.
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title="Layer readiness"
          description="A practical view of what is actually running versus what exists only as code."
        >
          <div className="space-y-3">
            {layerCards.map((layer) => (
              <div key={layer.name} className="rounded-[20px] border border-slate-900/8 bg-white/70 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="font-semibold text-slate-900">{layer.name}</p>
                    <p className="mt-1 text-sm leading-6 text-slate-500">{layer.detail}</p>
                  </div>
                  <StatusBadge
                    label={layer.label}
                    tone={layer.tone}
                  />
                </div>
              </div>
            ))}

            <div className="rounded-[20px] border border-dashed border-slate-900/10 bg-[#fff9ef] p-4">
              <div className="flex items-start gap-3">
                <div className="metric-icon-wrap bg-amber-100 text-amber-700">
                  <Network size={18} />
                </div>
                <div className="space-y-2">
                  <p className="font-semibold text-slate-900">Interpretation</p>
                  <p className="text-sm leading-6 text-slate-600">
                    This project already behaves like a working research console, but L3 and the full chain-backed incentive story are still not as operational as L1, L2 and the policy control plane.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <MetricTile
          label="Adaptive policy status"
          value={policyError ? 'Offline' : normalizeStatus(policyHealth?.status) === 'healthy' ? 'Hybrid live' : formatStatus(policyHealth?.status)}
          detail="Simple scenarios use deterministic rules; complex overlap can route into GLM."
          icon={<Activity size={22} />}
          tone={policyError ? 'red' : 'teal'}
        />
        <MetricTile
          label="Settlement visibility"
          value={l4Error ? 'Offline' : l4Health?.status === 'degraded' ? 'Partial' : formatStatus(l4Health?.status)}
          detail="L4 health can be degraded even when dashboard reads continue to work."
          icon={<Coins size={22} />}
          tone={l4Error ? 'red' : 'amber'}
        />
        <MetricTile
          label="Beneficial rarity"
          value={stats ? (stats.total_audits > 0 && stats.rare_count > 0 ? 'Observed' : 'Not yet visible') : 'Unknown'}
          detail="The target is to reward rare gradients only when they improve corner cases without harming the main task."
          icon={<Award size={22} />}
          tone={stats ? 'blue' : 'red'}
        />
      </div>
    </div>
  )
}
