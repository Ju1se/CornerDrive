import { useQuery } from '@tanstack/react-query'
import { Car, CheckCircle, Coins, Eye, Shield } from 'lucide-react'

import { fetchL3Status } from '../services/api'
import { PageIntro, Panel, StatusBadge } from '../components/ui/primitives'

const rules = [
  {
    trigger: 'ΔL_main > θ_tol',
    diagnosis: 'FRAUD',
    contract: 'Slash or penalize contributor',
    sbt: '-50',
    style: 'bg-rose-100 text-rose-800',
  },
  {
    trigger: 'ΔL_corner ≤ θ_rare and ΔL_main ≤ 0',
    diagnosis: 'RARITY',
    contract: 'Reward beneficial rare signal',
    sbt: '+10',
    style: 'bg-emerald-100 text-emerald-800',
  },
  {
    trigger: 'ΔL_main < 0',
    diagnosis: 'HONEST',
    contract: 'Keep contribution and reward',
    sbt: '+1',
    style: 'bg-sky-100 text-sky-800',
  },
  {
    trigger: 'Otherwise',
    diagnosis: 'NOISE',
    contract: 'Discard without incentive',
    sbt: '0',
    style: 'bg-slate-200 text-slate-700',
  },
]

export default function Architecture() {
  const { data: l3Status, isError: l3Error } = useQuery({
    queryKey: ['l3Status'],
    queryFn: fetchL3Status,
    refetchInterval: 15000,
    retry: 1,
  })

  const l3Lifecycle = l3Error
    ? 'Status unknown'
    : l3Status?.dataset_source === 'disk'
      ? 'Library only / disk-backed'
      : l3Status?.dataset_source === 'placeholder'
        ? 'Library only / placeholder'
        : 'Library only / attention needed'

  const l3BadgeTone = l3Error
    ? 'red'
    : l3Status?.dataset_source === 'disk'
      ? 'teal'
      : l3Status?.dataset_source === 'placeholder'
        ? 'amber'
        : 'red'

  const l3Description = l3Error
    ? 'Golden-dataset validation exists in code, but the frontend could not confirm its current dataset source from L4.'
    : l3Status
      ? `${l3Status.detail} Sample count: ${l3Status.sample_count ?? 0}.`
      : 'Golden-dataset validation exists in code, but is not deployed as an active service in the default stack.'

  const layers = [
    {
      id: 'l0',
      name: 'L0 Client Compliance',
      lifecycle: 'Out of scope',
      description: 'Vehicle-side privacy and integrity controls. Important to the research story, but not implemented in this repository.',
      icon: Car,
      accent: 'bg-slate-200 text-slate-700',
      badgeTone: 'slate',
      components: ['Norm clipping', 'Local DP', 'Top-k sparsification', 'Digital signature'],
    },
    {
      id: 'l1',
      name: 'L1 Linear Defense',
      lifecycle: 'Deployed',
      description: 'Fast intake screening using geometric median aggregation and cosine deviation.',
      icon: Shield,
      accent: 'bg-sky-100 text-sky-700',
      badgeTone: 'blue',
      components: ['Geometric median', 'Weiszfeld iteration', 'Cosine suspect filter'],
    },
    {
      id: 'l2',
      name: 'L2 Dual-Purpose Audit',
      lifecycle: 'Deployed',
      description: 'The key classifier: distinguishes fraud from beneficial rarity with main-task and corner-task loss signals.',
      icon: Eye,
      accent: 'bg-emerald-100 text-emerald-700',
      badgeTone: 'teal',
      components: ['Delta main analysis', 'Delta corner analysis', 'Beneficial-rarity threshold'],
    },
    {
      id: 'l3',
      name: 'L3 Gatekeeper',
      lifecycle: l3Lifecycle,
      description: l3Description,
      icon: CheckCircle,
      accent: 'bg-amber-100 text-amber-700',
      badgeTone: l3BadgeTone,
      components: ['Golden dataset', 'Global drift score', 'Approve / reject'],
    },
    {
      id: 'l4',
      name: 'L4 Settlement',
      lifecycle: 'Deployed',
      description: 'Dashboard, vehicle state and settlement API. On-chain execution is simplified relative to the full adaptive story.',
      icon: Coins,
      accent: 'bg-orange-100 text-orange-700',
      badgeTone: 'amber',
      components: ['Audit dashboard', 'Vehicle registry', 'Settlement contract calls'],
    },
  ]

  return (
    <div className="page-shell">
      <PageIntro
        eyebrow="Architecture"
        title="What is implemented versus what is aspirational"
        description="This page is meant to be honest about system boundaries. The repository already runs as a multi-layer research console, but not every conceptual layer is equally operational today."
      />

      <Panel
        title="Layer map"
        description="The runtime path is strongest at L1, L2, L4 and the policy control plane. L3 is present as code, and this page now shows whether its golden dataset is disk-backed or still placeholder."
      >
        <div className="space-y-4">
          {layers.map((layer, index) => {
            const Icon = layer.icon
            return (
              <div key={layer.id}>
                <div className="glass-panel-strong p-5">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="flex items-start gap-4">
                      <div className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-[20px] ${layer.accent}`}>
                        <Icon size={24} />
                      </div>
                      <div className="space-y-2">
                        <div className="flex flex-wrap items-center gap-3">
                          <h2 className="text-xl font-semibold text-slate-900">{layer.name}</h2>
                          <StatusBadge
                            label={layer.lifecycle}
                            tone={layer.badgeTone as 'teal' | 'amber' | 'red' | 'blue' | 'slate'}
                          />
                        </div>
                        <p className="max-w-3xl text-sm leading-7 text-slate-600">{layer.description}</p>
                        <div className="flex flex-wrap gap-2 pt-1">
                          {layer.components.map((component) => (
                            <span
                              key={component}
                              className="rounded-full bg-slate-900/5 px-3 py-1.5 text-xs font-medium text-slate-600"
                            >
                              {component}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                {index < layers.length - 1 ? (
                  <div className="mx-auto h-8 w-px bg-slate-900/10" />
                ) : null}
              </div>
            )
          })}
        </div>
      </Panel>

      <Panel
        title="Mechanism rules"
        description="The L2/L4 incentive story centers on a simple piecewise mechanism: harm the main task and you are fraud; strongly help the corner task without hurting the main task and you become beneficial rarity."
      >
        <div className="table-shell overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="table-header-row border-b border-slate-900/10">
                <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Trigger</th>
                <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Diagnosis</th>
                <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Settlement action</th>
                <th className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">SBT</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((rule) => (
                <tr key={rule.trigger} className="border-b border-slate-900/8 last:border-b-0">
                  <td className="px-5 py-4">
                    <code className="rounded-xl bg-slate-900/5 px-3 py-2 text-sm text-slate-700">{rule.trigger}</code>
                  </td>
                  <td className="px-5 py-4">
                    <span className={`inline-flex rounded-full px-3 py-1.5 text-xs font-semibold ${rule.style}`}>
                      {rule.diagnosis}
                    </span>
                  </td>
                  <td className="px-5 py-4 text-sm text-slate-600">{rule.contract}</td>
                  <td className="px-5 py-4 font-mono text-sm font-semibold text-slate-900">{rule.sbt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5 rounded-[20px] border border-teal-800/10 bg-teal-50/80 p-5">
          <p className="text-sm leading-7 text-teal-900">
            <strong>The important caveat:</strong> the conceptual policy story is richer than the current chain contract. The adaptive controller can tune thresholds and incentives off-chain, but the on-chain layer still uses simplified fixed reward and slashing logic.
          </p>
        </div>
      </Panel>
    </div>
  )
}
