import { useMemo, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart3,
  Car,
  ChevronRight,
  ClipboardCheck,
  Coins,
  LayoutDashboard,
  Menu,
  Network,
  ShieldCheck,
  SlidersHorizontal,
  X,
} from 'lucide-react'

import { fetchL1Health, fetchL4Health, fetchPolicyHealth } from '../../services/api'
import { StatusBadge } from '../ui/primitives'

const navItems = [
  {
    path: '/',
    label: 'Dashboard',
    blurb: 'Live system pulse and audit mix',
    icon: LayoutDashboard,
  },
  {
    path: '/architecture',
    label: 'Architecture',
    blurb: 'Layer map and operating boundaries',
    icon: Network,
  },
  {
    path: '/vehicles',
    label: 'Vehicles',
    blurb: 'Reputation, tiers and contributors',
    icon: Car,
  },
  {
    path: '/audits',
    label: 'Audits',
    blurb: 'Recent L2 verdict stream',
    icon: ClipboardCheck,
  },
  {
    path: '/analysis',
    label: 'Data Analysis',
    blurb: 'Live history and backend baseline evidence',
    icon: BarChart3,
  },
  {
    path: '/settlement',
    label: 'Settlement',
    blurb: 'Rewards, slashing and chain state',
    icon: Coins,
  },
  {
    path: '/policy',
    label: 'Policy',
    blurb: 'Adaptive controls and GLM decisions',
    icon: SlidersHorizontal,
  },
]

function toneFromStatus(status?: string) {
  const normalized = normalizeStatus(status)
  if (!normalized) return 'slate'
  if (normalized === 'healthy' || normalized === 'online') return 'teal'
  if (normalized === 'degraded') return 'amber'
  return 'red'
}

function normalizeStatus(status?: string) {
  if (!status) return undefined
  if (status === 'ok') return 'healthy'
  return status
}

function formatStatus(status?: string) {
  const normalized = normalizeStatus(status)
  if (!normalized) return 'Offline'
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(false)

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

  const currentPage = navItems.find((item) => item.path === location.pathname) ?? navItems[0]

  const services = useMemo(
    () => [
      {
        name: 'L1 Screening',
        status: l1Error ? 'offline' : normalizeStatus(l1Health?.status),
        detail: l1Health?.service ?? 'Gradient intake',
      },
      {
        name: 'L4 Dashboard',
        status: l4Error ? 'offline' : normalizeStatus(l4Health?.status),
        detail: l4Health?.service ?? 'Audit and settlement view',
      },
      {
        name: 'Policy Agent',
        status: policyError ? 'offline' : normalizeStatus(policyHealth?.status),
        detail: policyHealth?.service ?? 'Adaptive controller',
      },
    ],
    [l1Error, l1Health, l4Error, l4Health, policyError, policyHealth],
  )

  const overallTone = services.some((service) => service.status === 'offline')
    ? 'red'
    : services.some((service) => service.status === 'degraded')
      ? 'amber'
      : 'teal'

  const onlineCount = services.filter((service) => service.status && service.status !== 'offline').length

  const shell = (
    <aside className="flex h-full flex-col gap-6 overflow-y-auto px-4 py-5 md:px-5">
      <div className="glass-panel-strong p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="section-label">Research Control Surface</p>
            <div className="mt-3 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-700 text-lg font-bold text-white shadow-lg shadow-teal-950/20">
                FL
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">FLPG</h1>
                <p className="text-sm leading-6 text-slate-600">Federated learning security console</p>
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="rounded-2xl bg-slate-900/5 p-2 text-slate-500 hover:bg-slate-900/10 lg:hidden"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      <nav className="glass-panel p-3">
        <div className="space-y-1.5">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path
            const Icon = item.icon

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`group flex items-start gap-3 rounded-[22px] px-4 py-3.5 transition-all ${
                  isActive
                    ? 'bg-teal-700 text-white shadow-lg shadow-teal-950/15'
                    : 'text-slate-700 hover:bg-white/70'
                }`}
              >
                <div
                  className={`mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl ${
                    isActive ? 'bg-white/15 text-white' : 'bg-slate-900/5 text-slate-500'
                  }`}
                >
                  <Icon size={18} />
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{item.label}</span>
                    {isActive ? <ChevronRight size={14} className="shrink-0" /> : null}
                  </div>
                  <p
                    className={`mt-1 text-sm leading-5 ${
                      isActive ? 'text-teal-50/90' : 'text-slate-500'
                    }`}
                  >
                    {item.blurb}
                  </p>
                </div>
              </Link>
            )
          })}
        </div>
      </nav>

      <div className="glass-panel p-4">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="section-label">Service Pulse</p>
            <h2 className="mt-2 text-lg font-semibold text-slate-900">Runtime health</h2>
          </div>
          <StatusBadge label={`${onlineCount}/3 online`} tone={overallTone} pulse />
        </div>
        <div className="space-y-3">
          {services.map((service) => (
            <div key={service.name} className="rounded-[20px] border border-slate-900/6 bg-white/60 p-4">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold text-slate-900">{service.name}</p>
                  <StatusBadge
                    label={formatStatus(service.status)}
                    tone={toneFromStatus(service.status) as 'teal' | 'amber' | 'red' | 'blue' | 'slate'}
                  />
                </div>
                <p className="break-words text-sm leading-6 text-slate-500">{service.detail}</p>
              </div>
            </div>
          ))}
          <div className="rounded-[20px] border border-dashed border-slate-900/10 bg-slate-50/60 p-4">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-sm font-semibold text-slate-900">L3 Gatekeeper</p>
                <StatusBadge label="Library only" tone="slate" />
              </div>
              <p className="break-words text-sm leading-6 text-slate-500">
                Implemented as library logic, not a live service
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel p-4">
        <p className="section-label">Operator Notes</p>
        <div className="mt-3 space-y-3 text-sm leading-6 text-slate-600">
          <p>Dashboard, Audits, Vehicles, Settlement and Policy now stay tied to live backend reads instead of presentation-only snapshots.</p>
          <p>Architecture remains the conceptual view of what is deployed versus what is still only library code.</p>
        </div>
      </div>
    </aside>
  )

  return (
    <div className="min-h-screen overflow-x-hidden">
      <div className="lg:grid lg:min-h-screen lg:grid-cols-[312px_minmax(0,1fr)]">
        <div className="fixed left-0 top-0 z-40 hidden h-screen w-[312px] lg:block">{shell}</div>

        {sidebarOpen ? (
          <div className="fixed inset-0 z-50 bg-slate-950/35 backdrop-blur-sm lg:hidden">
            <button
              type="button"
              aria-label="Close navigation"
              onClick={() => setSidebarOpen(false)}
              className="absolute inset-0"
            />
            <div className="absolute inset-y-0 left-0 z-10 w-[88vw] max-w-[312px]">{shell}</div>
          </div>
        ) : null}

        <div className="lg:col-start-2">
          <header className="sticky top-0 z-30 border-b border-slate-900/8 bg-[#f6f2e8]/85 backdrop-blur-xl">
            <div className="flex items-center justify-between gap-4 px-4 py-4 md:px-8 md:py-5">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setSidebarOpen(true)}
                  className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-900/10 bg-white/70 text-slate-600 lg:hidden"
                >
                  <Menu size={18} />
                </button>
                <div className="min-w-0">
                  <p className="section-label">{currentPage.label}</p>
                  <h2 className="mt-1 text-xl font-semibold text-slate-900">{currentPage.label}</h2>
                  <p className="mt-1 hidden max-w-xl text-sm leading-6 text-slate-500 md:block">
                    {currentPage.blurb}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <StatusBadge
                  label={overallTone === 'teal' ? 'Stack healthy' : overallTone === 'amber' ? 'Needs attention' : 'Service outage'}
                  tone={overallTone}
                  pulse
                />
                <div className="hidden rounded-2xl border border-slate-900/8 bg-white/65 px-4 py-2 text-sm text-slate-500 md:block">
                  <div className="flex items-center gap-2">
                    <ShieldCheck size={16} className="text-teal-700" />
                    <span>Live FLPG control plane with policy telemetry</span>
                  </div>
                </div>
              </div>
            </div>
          </header>

          <main className="px-4 py-6 md:px-8 md:py-8">
            <div className="mx-auto max-w-[1380px]">{children}</div>
          </main>
        </div>
      </div>

    </div>
  )
}
