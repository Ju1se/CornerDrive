import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Car, Gem, Trophy, Users } from 'lucide-react'

import { fetchVehicles } from '../services/api'
import { EmptyState, MetricTile, PageIntro, Panel, StateCallout, StatusBadge } from '../components/ui/primitives'

interface VehicleRow {
  address: string
  reputation: number
  tier: string
  contributions: number
}

function getTierTone(tier: string) {
  switch (tier) {
    case 'PLATINUM':
      return 'teal'
    case 'GOLD':
      return 'amber'
    case 'SILVER':
      return 'blue'
    default:
      return 'slate'
  }
}

export default function Vehicles() {
  const page = 1

  const { data, isLoading, isError } = useQuery({
    queryKey: ['vehicles', page],
    queryFn: () => fetchVehicles(page, 10),
    refetchInterval: 10000,
  })

  const rows = useMemo(() => (data?.data ?? []) as VehicleRow[], [data])

  const summary = useMemo(() => {
    const tierCounts = rows.reduce<Record<string, number>>((acc, row) => {
      acc[row.tier] = (acc[row.tier] || 0) + 1
      return acc
    }, {})

    return {
      count: rows.length,
      avgReputation: rows.length > 0
        ? rows.reduce((sum, row) => sum + row.reputation, 0) / rows.length
        : 0,
      topTier: Object.entries(tierCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'BRONZE',
      topContributor: rows.slice().sort((a, b) => b.contributions - a.contributions)[0],
    }
  }, [rows])

  return (
    <div className="page-shell">
      <PageIntro
        eyebrow="Participants"
        title="Vehicles and reputation state"
        description="A registry-style view of contributors that have accumulated reputation, tier status and contribution history inside the settlement layer."
      />

      {isError ? (
        <StateCallout
          icon={<AlertTriangle size={18} />}
          title="Vehicle registry unavailable"
          description="The frontend could not reach the L4 vehicles endpoint, so this page now reports the outage instead of pretending that no vehicles exist."
          tone="red"
        />
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Visible vehicles"
          value={isError ? 'N/A' : `${summary.count}`}
          detail={isError ? 'L4 vehicles endpoint unavailable' : 'Rows currently returned by the L4 vehicle endpoint'}
          icon={<Users size={22} />}
          tone={isError ? 'red' : 'blue'}
        />
        <MetricTile
          label="Average reputation"
          value={isError ? 'N/A' : summary.avgReputation.toFixed(1)}
          detail={isError ? 'Reputation telemetry unavailable' : 'Quick read on whether contributor quality is improving'}
          icon={<Gem size={22} />}
          tone={isError ? 'red' : 'teal'}
        />
        <MetricTile
          label="Most common tier"
          value={isError ? 'N/A' : summary.topTier}
          detail={isError ? 'Tier mix unavailable' : 'The dominant tier band in the current vehicle window'}
          icon={<Trophy size={22} />}
          tone={isError ? 'red' : 'amber'}
        />
        <MetricTile
          label="Top contributor"
          value={isError ? 'N/A' : summary.topContributor ? summary.topContributor.address.slice(0, 10) : 'N/A'}
          detail={
            isError
              ? 'Contributor leaderboard unavailable'
              : summary.topContributor
                ? `${summary.topContributor.contributions} contributions`
                : 'Waiting for vehicles'
          }
          icon={<Car size={22} />}
          tone={isError ? 'red' : 'slate'}
        />
      </div>

      <Panel
        title="Vehicle table"
        description="This page is strongest when the demo has been running long enough to show tier separation and reputation accumulation."
      >
        {isLoading ? (
          <div className="grid gap-3">
            {Array.from({ length: 5 }).map((_, index) => (
              <div key={index} className="h-16 animate-pulse rounded-[18px] bg-slate-200/60" />
            ))}
          </div>
        ) : isError ? (
          <StateCallout
            icon={<AlertTriangle size={18} />}
            title="Unable to load vehicles"
            description="Check whether the L4 dashboard service is up. This table no longer collapses a transport failure into the empty-state message."
            tone="red"
          />
        ) : rows.length > 0 ? (
          <div className="table-shell overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="table-header-row border-b border-slate-900/10">
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Vehicle</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Reputation</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Tier</th>
                  <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Contributions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((vehicle) => (
                  <tr key={vehicle.address} className="border-b border-slate-900/8 last:border-b-0 hover:bg-white/60">
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-3">
                        <div className="metric-icon-wrap h-10 w-10 bg-slate-100/80 text-slate-500">
                          <Car size={18} />
                        </div>
                        <div>
                          <p className="break-all font-mono text-sm text-slate-900">{vehicle.address}</p>
                          <p className="mt-1 text-xs text-slate-500">Contributor address</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-5 py-4 text-sm font-semibold text-slate-900">{vehicle.reputation}</td>
                    <td className="px-5 py-4">
                      <StatusBadge
                        label={vehicle.tier}
                        tone={getTierTone(vehicle.tier) as 'teal' | 'amber' | 'red' | 'blue' | 'slate'}
                      />
                    </td>
                    <td className="px-5 py-4 text-sm text-slate-600">{vehicle.contributions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={<Car size={20} />}
            title="No vehicles registered yet"
            description="Once the demo starts writing vehicle state into Redis and L4, contributor rows will appear here."
          />
        )}
      </Panel>
    </div>
  )
}
