import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Award, Coins, Link2, ShieldAlert, Wallet } from 'lucide-react'

import { fetchL4Health, fetchSystemStats } from '../services/api'
import { MetricTile, PageIntro, Panel, StateCallout, StatusBadge } from '../components/ui/primitives'

export default function Settlement() {
  const { data: stats, isError: statsError } = useQuery({
    queryKey: ['systemStats'],
    queryFn: fetchSystemStats,
    refetchInterval: 5000,
  })

  const { data: health, isError: healthError } = useQuery({
    queryKey: ['health', 'l4'],
    queryFn: fetchL4Health,
    refetchInterval: 15000,
    retry: 1,
  })

  return (
    <div className="page-shell">
      <PageIntro
        eyebrow="Settlement"
        title="FLPG settlement layer"
        description="This page only shows live reads from the L4 stats and health endpoints: rewards, slashing pressure and runtime readiness."
        actions={
          <StatusBadge
            label={
              healthError
                ? 'Settlement offline'
                : health?.status === 'healthy'
                  ? 'Settlement healthy'
                  : health?.status === 'degraded'
                    ? 'Chain config incomplete'
                    : 'Settlement offline'
            }
            tone={healthError ? 'red' : health?.status === 'healthy' ? 'teal' : health?.status === 'degraded' ? 'amber' : 'red'}
            pulse
          />
        }
      />

      {statsError || healthError ? (
        <StateCallout
          icon={<AlertTriangle size={18} />}
          title="Settlement telemetry is partially unavailable"
          description="This page now distinguishes a backend outage from a legitimately quiet chain. Live metrics stay blank when the L4 service cannot be reached."
          tone="red"
        />
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Total rewards"
          value={stats ? `${stats.total_rewards_distributed.toFixed(4)} ETH` : 'N/A'}
          detail={stats ? 'Reward volume attributed to honest and rare contributors' : 'L4 stats endpoint unavailable'}
          icon={<Coins size={22} />}
          tone={stats ? 'teal' : 'red'}
        />
        <MetricTile
          label="Total slashed"
          value={stats ? `${stats.total_slashed.toFixed(4)} ETH` : 'N/A'}
          detail={stats ? 'Cumulative penalty pressure from fraud verdicts' : 'Slashing totals unavailable'}
          icon={<ShieldAlert size={22} />}
          tone="red"
        />
        <MetricTile
          label="Beneficial-rarity jackpots"
          value={stats ? `${stats.rare_count}` : 'N/A'}
          detail={stats ? 'Rare updates that received enhanced incentive treatment' : 'Beneficial-rarity counts unavailable'}
          icon={<Award size={22} />}
          tone={stats ? 'amber' : 'red'}
        />
        <MetricTile
          label="Rewarded audits"
          value={stats ? `${stats.honest_count + stats.rare_count}` : 'N/A'}
          detail={stats ? 'Honest plus beneficial-rarity verdicts surfaced by L4' : 'Reward-side counts unavailable'}
          icon={<Wallet size={22} />}
          tone={stats ? 'blue' : 'red'}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel
          title="Runtime checks"
          description="Direct health information returned by the L4 service, without frontend-side placeholder values."
        >
          {healthError ? (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Unable to load L4 health checks"
              description="The settlement service did not respond, so this page avoids showing contract or chain readiness from stale frontend config."
              tone="red"
            />
          ) : health ? (
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(health.checks).map(([key, value]) => (
                <div key={key} className="rounded-[22px] border border-slate-900/8 bg-white/70 p-5">
                  <p className="text-sm text-slate-500">{key}</p>
                  <p className="mt-3 break-all font-mono text-lg font-semibold text-slate-900">{value}</p>
                </div>
              ))}
            </div>
          ) : null}
        </Panel>

        <Panel
          title="Settlement mix"
          description="Live settlement-facing outcome counts derived from the current L4 stats payload."
        >
          {statsError ? (
            <StateCallout
              icon={<AlertTriangle size={18} />}
              title="Unable to load settlement mix"
              description="The L4 stats endpoint is unavailable, so reward and slash composition is intentionally withheld."
              tone="red"
            />
          ) : stats ? (
            <div className="grid gap-4 md:grid-cols-2">
              {[
                { label: 'Honest rewarded', value: stats.honest_count, icon: Link2 },
                { label: 'Beneficial rarity rewarded', value: stats.rare_count, icon: Award },
                { label: 'Fraud penalized', value: stats.fraud_count, icon: ShieldAlert },
                { label: 'Noise ignored', value: stats.noise_count, icon: Coins },
              ].map((item) => {
                const Icon = item.icon
                return (
                  <div key={item.label} className="rounded-[20px] border border-slate-900/8 bg-white/70 p-5">
                    <div className="flex items-center gap-3">
                      <div className="metric-icon-wrap bg-slate-100/80 text-slate-500">
                        <Icon size={18} />
                      </div>
                      <div>
                        <p className="text-sm text-slate-500">{item.label}</p>
                        <p className="mt-2 text-2xl font-bold text-slate-900">{item.value}</p>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : null}
        </Panel>
      </div>
    </div>
  )
}
