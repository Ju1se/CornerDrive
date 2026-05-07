import { useMemo, useState } from 'react'
import { Crosshair, Radar } from 'lucide-react'

import type { RecentAudit } from '../../types'
import type { Policy } from '../../types/policy'
import { ClassificationBadge, StateCallout, StatusBadge } from '../ui/primitives'

const WIDTH = 920
const HEIGHT = 500
const PADDING = {
  top: 28,
  right: 28,
  bottom: 64,
  left: 84,
}

type OverlayVerdict = 'FRAUD' | 'RARITY' | 'HONEST' | 'NOISE'

function formatMetric(value: number) {
  return value.toFixed(4)
}

function formatAxisTick(value: number) {
  if (Math.abs(value) >= 0.1) {
    return value.toFixed(2)
  }
  return value.toFixed(3)
}

function classifyAgainstPolicy(audit: RecentAudit, policy: Policy): OverlayVerdict {
  if (audit.delta_loss_main > policy.theta_tol) {
    return 'FRAUD'
  }

  if (audit.delta_loss_corner <= policy.theta_rare && audit.delta_loss_main <= 0) {
    return 'RARITY'
  }

  if (audit.delta_loss_main < 0) {
    return 'HONEST'
  }

  return 'NOISE'
}

function explainOverlayVerdict(audit: RecentAudit, policy: Policy) {
  const fraudTriggered = audit.delta_loss_main > policy.theta_tol
  const rarityTriggered = audit.delta_loss_corner <= policy.theta_rare && audit.delta_loss_main <= 0
  const blockedRarity = audit.delta_loss_corner <= policy.theta_rare && audit.delta_loss_main > 0

  if (fraudTriggered) {
    return `Fraud line crossed because ΔL_main ${formatMetric(audit.delta_loss_main)} > θ_tol ${formatMetric(policy.theta_tol)}.`
  }

  if (rarityTriggered) {
    return `Beneficial-rarity line crossed because ΔL_corner ${formatMetric(audit.delta_loss_corner)} ≤ θ_rare ${formatMetric(policy.theta_rare)} and ΔL_main ${formatMetric(audit.delta_loss_main)} does not harm the main task.`
  }

  if (blockedRarity) {
    return `Corner help is visible, but ΔL_main ${formatMetric(audit.delta_loss_main)} is still above zero, so the point stays out of the beneficial-rarity band.`
  }

  if (audit.delta_loss_main < 0) {
    return `No fraud or beneficial-rarity trigger; ΔL_main ${formatMetric(audit.delta_loss_main)} is still helping the main task.`
  }

  return 'The point stays between the active fraud and beneficial-rarity boundaries, so it lands in the noise band.'
}

function getPointClasses(type: string) {
  switch (type) {
    case 'FRAUD':
      return {
        fill: '#e11d48',
        stroke: '#881337',
      }
    case 'RARITY':
      return {
        fill: '#059669',
        stroke: '#065f46',
      }
    case 'HONEST':
      return {
        fill: '#0284c7',
        stroke: '#075985',
      }
    default:
      return {
        fill: '#64748b',
        stroke: '#334155',
      }
  }
}

function buildTicks(min: number, max: number, count = 5) {
  if (min === max) {
    return [min]
  }

  const step = (max - min) / (count - 1)
  return Array.from({ length: count }, (_, index) => min + step * index)
}

export function AuditDecisionMap({
  audits,
  policy,
  policyUnavailable,
}: {
  audits: RecentAudit[]
  policy: Policy | null
  policyUnavailable: boolean
}) {
  const [activeIndex, setActiveIndex] = useState(0)

  const activeAudit = audits[activeIndex] ?? audits[0] ?? null

  const plot = useMemo(() => {
    const xCandidates = audits.map((audit) => audit.delta_loss_main)
    const yCandidates = audits.map((audit) => audit.delta_loss_corner)

    xCandidates.push(0)
    yCandidates.push(0)

    if (policy) {
      xCandidates.push(policy.theta_tol)
      yCandidates.push(policy.theta_rare)
    }

    const rawXMin = Math.min(...xCandidates, -0.02)
    const rawXMax = Math.max(...xCandidates, 0.08)
    const rawYMin = Math.min(...yCandidates, -0.05)
    const rawYMax = Math.max(...yCandidates, 0.05)

    const xPadding = Math.max((rawXMax - rawXMin) * 0.14, 0.01)
    const yPadding = Math.max((rawYMax - rawYMin) * 0.14, 0.01)

    const xMin = rawXMin - xPadding
    const xMax = rawXMax + xPadding
    const yMin = rawYMin - yPadding
    const yMax = rawYMax + yPadding

    const plotLeft = PADDING.left
    const plotRight = WIDTH - PADDING.right
    const plotTop = PADDING.top
    const plotBottom = HEIGHT - PADDING.bottom
    const plotWidth = plotRight - plotLeft
    const plotHeight = plotBottom - plotTop

    const scaleX = (value: number) => plotLeft + ((value - xMin) / (xMax - xMin)) * plotWidth
    const scaleY = (value: number) => plotBottom - ((value - yMin) / (yMax - yMin)) * plotHeight

    return {
      xMin,
      xMax,
      yMin,
      yMax,
      plotLeft,
      plotRight,
      plotTop,
      plotBottom,
      scaleX,
      scaleY,
      xTicks: buildTicks(xMin, xMax),
      yTicks: buildTicks(yMin, yMax),
    }
  }, [audits, policy])

  const zeroX = plot.scaleX(0)
  const zeroY = plot.scaleY(0)
  const fraudX = policy ? plot.scaleX(policy.theta_tol) : null
  const rarityY = policy ? plot.scaleY(policy.theta_rare) : null

  const regionLabels = policy
    ? [
        {
          name: 'Honest',
          x: (plot.plotLeft + zeroX) / 2,
          y: (plot.plotTop + (rarityY ?? plot.plotBottom)) / 2,
          color: 'fill-sky-800',
        },
        {
          name: 'Noise',
          x: (zeroX + (fraudX ?? plot.plotRight)) / 2,
          y: (plot.plotTop + plot.plotBottom) / 2,
          color: 'fill-slate-700',
        },
        {
          name: 'Beneficial rarity',
          x: (plot.plotLeft + zeroX) / 2,
          y: ((rarityY ?? plot.plotBottom) + plot.plotBottom) / 2,
          color: 'fill-emerald-800',
        },
        {
          name: 'Fraud',
          x: (((fraudX ?? plot.plotRight) + plot.plotRight) / 2),
          y: (plot.plotTop + plot.plotBottom) / 2,
          color: 'fill-rose-800',
        },
      ]
    : []

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        {policy ? (
          <>
            <StatusBadge label={`θ_tol ${formatMetric(policy.theta_tol)}`} tone="red" />
            <StatusBadge label={`θ_rare ${formatMetric(policy.theta_rare)}`} tone="teal" />
            <StatusBadge label="Stored L2 verdicts" tone="blue" />
          </>
        ) : policyUnavailable ? (
          <StatusBadge label="Threshold overlay unavailable" tone="amber" />
        ) : (
          <StatusBadge label="Waiting for active policy" tone="amber" />
        )}
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
        <div className="overflow-hidden rounded-[24px] border border-slate-900/10 bg-white/80 p-4 shadow-[0_20px_60px_-42px_rgba(15,23,42,0.35)]">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-slate-900">Decision boundary view</p>
              <p className="mt-1 text-sm leading-6 text-slate-500">
                Each point is a recent audit. X is `ΔL_main`; Y is `ΔL_corner`.
              </p>
            </div>
            <div className="flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600">
              <Radar size={14} />
              {audits.length} points
            </div>
          </div>

          <div className="relative">
            <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full">
              {policy && fraudX !== null && rarityY !== null ? (
                <>
                  <rect
                    x={plot.plotLeft}
                    y={plot.plotTop}
                    width={Math.max(zeroX - plot.plotLeft, 0)}
                    height={Math.max(rarityY - plot.plotTop, 0)}
                    fill="#e0f2fe"
                    opacity="0.7"
                  />
                  <rect
                    x={zeroX}
                    y={plot.plotTop}
                    width={Math.max(fraudX - zeroX, 0)}
                    height={Math.max(rarityY - plot.plotTop, 0)}
                    fill="#e2e8f0"
                    opacity="0.65"
                  />
                  <rect
                    x={plot.plotLeft}
                    y={rarityY}
                    width={Math.max(zeroX - plot.plotLeft, 0)}
                    height={Math.max(plot.plotBottom - rarityY, 0)}
                    fill="#d1fae5"
                    opacity="0.72"
                  />
                  <rect
                    x={zeroX}
                    y={rarityY}
                    width={Math.max(fraudX - zeroX, 0)}
                    height={Math.max(plot.plotBottom - rarityY, 0)}
                    fill="#e2e8f0"
                    opacity="0.65"
                  />
                  <rect
                    x={fraudX}
                    y={plot.plotTop}
                    width={Math.max(plot.plotRight - fraudX, 0)}
                    height={Math.max(plot.plotBottom - plot.plotTop, 0)}
                    fill="#ffe4e6"
                    opacity="0.82"
                  />
                </>
              ) : null}

              {plot.yTicks.map((tick) => {
                const y = plot.scaleY(tick)
                return (
                  <g key={`y-${tick}`}>
                    <line x1={plot.plotLeft} y1={y} x2={plot.plotRight} y2={y} stroke="#cbd5e1" strokeDasharray="4 8" />
                    <text x={plot.plotLeft - 14} y={y + 4} textAnchor="end" className="fill-slate-500 text-[12px]">
                      {formatAxisTick(tick)}
                    </text>
                  </g>
                )
              })}

              {plot.xTicks.map((tick) => {
                const x = plot.scaleX(tick)
                return (
                  <g key={`x-${tick}`}>
                    <line x1={x} y1={plot.plotTop} x2={x} y2={plot.plotBottom} stroke="#cbd5e1" strokeDasharray="4 8" />
                    <text x={x} y={plot.plotBottom + 22} textAnchor="middle" className="fill-slate-500 text-[12px]">
                      {formatAxisTick(tick)}
                    </text>
                  </g>
                )
              })}

              <line x1={plot.plotLeft} y1={zeroY} x2={plot.plotRight} y2={zeroY} stroke="#64748b" strokeWidth="1.5" />
              <line x1={zeroX} y1={plot.plotTop} x2={zeroX} y2={plot.plotBottom} stroke="#64748b" strokeWidth="1.5" />

              {policy && fraudX !== null ? (
                <>
                  <line
                    x1={fraudX}
                    y1={plot.plotTop}
                    x2={fraudX}
                    y2={plot.plotBottom}
                    stroke="#e11d48"
                    strokeWidth="2.5"
                    strokeDasharray="8 8"
                  />
                  <text x={fraudX + 8} y={plot.plotTop + 18} className="fill-rose-700 text-[12px] font-semibold">
                    θ_tol
                  </text>
                </>
              ) : null}

              {policy && rarityY !== null ? (
                <>
                  <line
                    x1={plot.plotLeft}
                    y1={rarityY}
                    x2={plot.plotRight}
                    y2={rarityY}
                    stroke="#059669"
                    strokeWidth="2.5"
                    strokeDasharray="8 8"
                  />
                  <text x={plot.plotLeft + 8} y={rarityY - 10} className="fill-emerald-700 text-[12px] font-semibold">
                    θ_rare
                  </text>
                </>
              ) : null}

              {regionLabels.map((region) => (
                <text
                  key={region.name}
                  x={region.x}
                  y={region.y}
                  textAnchor="middle"
                  className={`${region.color} text-[14px] font-semibold uppercase tracking-[0.18em]`}
                  opacity="0.75"
                >
                  {region.name}
                </text>
              ))}

              {audits.map((audit, index) => {
                const point = getPointClasses(audit.classification)
                const isActive = activeAudit?.vehicle_id === audit.vehicle_id && activeAudit?.timestamp === audit.timestamp
                return (
                  <circle
                    key={`${audit.vehicle_id}-${audit.timestamp}-${index}`}
                    cx={plot.scaleX(audit.delta_loss_main)}
                    cy={plot.scaleY(audit.delta_loss_corner)}
                    r={isActive ? 8.5 : 6.5}
                    fill={point.fill}
                    stroke={isActive ? '#0f172a' : point.stroke}
                    strokeWidth={isActive ? 2.4 : 1.4}
                    opacity={isActive ? 1 : 0.88}
                    className="cursor-pointer transition-all duration-150"
                    onMouseEnter={() => setActiveIndex(index)}
                    onFocus={() => setActiveIndex(index)}
                    onClick={() => setActiveIndex(index)}
                    tabIndex={0}
                    aria-label={`${audit.classification} at ${formatMetric(audit.delta_loss_main)}, ${formatMetric(audit.delta_loss_corner)}`}
                  />
                )
              })}

              <text
                x={(plot.plotLeft + plot.plotRight) / 2}
                y={HEIGHT - 14}
                textAnchor="middle"
                className="fill-slate-700 text-[13px] font-medium"
              >
                ΔL_main → main-task harm increases to the right
              </text>
              <text
                x={18}
                y={(plot.plotTop + plot.plotBottom) / 2}
                transform={`rotate(-90 18 ${(plot.plotTop + plot.plotBottom) / 2})`}
                textAnchor="middle"
                className="fill-slate-700 text-[13px] font-medium"
              >
                ΔL_corner → beneficial rare help gets lower
              </text>
            </svg>
          </div>
        </div>

        <div className="space-y-4">
          {activeAudit ? (
            <div className="rounded-[24px] border border-slate-900/10 bg-white/80 p-5 shadow-[0_20px_60px_-42px_rgba(15,23,42,0.35)]">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-2">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">Selected audit</p>
                  <ClassificationBadge type={activeAudit.classification} />
                </div>
                <div className="flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600">
                  <Crosshair size={14} />
                  Hover or tap a point
                </div>
              </div>

              <div className="mt-4 space-y-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Vehicle</p>
                  <p className="mt-1 break-all font-mono text-sm text-slate-900">{activeAudit.vehicle_id}</p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-[18px] border border-slate-900/8 bg-slate-50/80 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">ΔL main</p>
                    <p className="mt-2 font-mono text-2xl font-semibold text-slate-900">
                      {formatMetric(activeAudit.delta_loss_main)}
                    </p>
                  </div>
                  <div className="rounded-[18px] border border-slate-900/8 bg-slate-50/80 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">ΔL corner</p>
                    <p className="mt-2 font-mono text-2xl font-semibold text-slate-900">
                      {formatMetric(activeAudit.delta_loss_corner)}
                    </p>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-[18px] border border-slate-900/8 bg-slate-50/80 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">SBT delta</p>
                    <p
                      className={`mt-2 text-2xl font-semibold ${
                        activeAudit.sbt_points > 0
                          ? 'text-emerald-700'
                          : activeAudit.sbt_points < 0
                            ? 'text-rose-700'
                            : 'text-slate-700'
                      }`}
                    >
                      {activeAudit.sbt_points > 0 ? '+' : ''}
                      {activeAudit.sbt_points}
                    </p>
                  </div>
                  <div className="rounded-[18px] border border-slate-900/8 bg-slate-50/80 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Observed</p>
                    <p className="mt-2 text-sm font-medium text-slate-900">
                      {new Date(activeAudit.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>

                {policy ? (
                  <div className="rounded-[18px] border border-slate-900/8 bg-slate-50/80 p-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Current boundary reading</p>
                      <ClassificationBadge type={classifyAgainstPolicy(activeAudit, policy)} compact />
                    </div>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      {explainOverlayVerdict(activeAudit, policy)}
                    </p>
                    {classifyAgainstPolicy(activeAudit, policy) !== activeAudit.classification ? (
                      <p className="mt-2 text-sm leading-6 text-amber-700">
                        Stored L2 verdict differs from the current overlay. That can happen when recent points were judged under an earlier round policy.
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <div className="rounded-[18px] border border-slate-900/8 bg-slate-50/80 p-4">
                    <p className="text-sm leading-6 text-slate-600">
                      Active policy thresholds are not available, so this side panel is showing the stored verdict only.
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : null}

          <StateCallout
            title="How to read this view"
            description="Points are colored by stored L2 verdicts. The dashed threshold lines come from the current active policy, so they act as a live reference overlay rather than a guaranteed historical replay."
            tone="blue"
          />
        </div>
      </div>
    </div>
  )
}
