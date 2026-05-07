import type { ReactNode } from 'react'

import { formatClassificationLabel } from '../../utils/classification'

type Tone = 'teal' | 'amber' | 'red' | 'blue' | 'slate'

const toneMap: Record<Tone, string> = {
  teal: 'bg-teal-100/80 text-teal-800',
  amber: 'bg-amber-100/80 text-amber-800',
  red: 'bg-rose-100/80 text-rose-800',
  blue: 'bg-blue-100/80 text-blue-800',
  slate: 'bg-slate-200/70 text-slate-700',
}

const toneSurfaceMap: Record<Tone, string> = {
  teal: 'border-teal-200 bg-teal-50/80',
  amber: 'border-amber-200 bg-amber-50/80',
  red: 'border-rose-200 bg-rose-50/80',
  blue: 'border-blue-200 bg-blue-50/80',
  slate: 'border-slate-200 bg-slate-50/80',
}

function joinClasses(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

export function PageIntro({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <section className="page-hero">
      <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
        <div className="max-w-3xl space-y-3">
          {eyebrow ? <p className="section-label">{eyebrow}</p> : null}
          <div className="space-y-3">
            <h1 className="max-w-4xl text-[clamp(2.1rem,4vw,2.95rem)] font-bold leading-[1.02] text-slate-900">
              {title}
            </h1>
            <p className="max-w-2xl text-sm leading-7 text-slate-600 md:text-[1.02rem]">
              {description}
            </p>
          </div>
        </div>
        {actions ? (
          <div className="flex w-full flex-wrap items-center gap-3 xl:w-auto xl:max-w-[32rem] xl:justify-end xl:pt-1">
            {actions}
          </div>
        ) : null}
      </div>
    </section>
  )
}

export function Panel({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: string
  description?: string
  action?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={joinClasses('glass-panel p-5 md:p-6', className)}>
      {title || description || action ? (
        <div className="mb-5 flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1.5">
            {title ? <h2 className="text-[1.28rem] font-semibold leading-tight text-slate-900">{title}</h2> : null}
            {description ? <p className="max-w-2xl text-sm leading-6 text-slate-600">{description}</p> : null}
          </div>
          {action ? <div className="flex w-full flex-wrap items-center gap-2 xl:w-auto xl:justify-end">{action}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  )
}

export function MetricTile({
  label,
  value,
  detail,
  icon,
  tone = 'slate',
}: {
  label: string
  value: string
  detail?: string
  icon?: ReactNode
  tone?: Tone
}) {
  return (
    <article className="metric-tile h-full">
      <div className="flex h-full items-start justify-between gap-4">
        <div className="min-w-0 space-y-3">
          <p className="text-sm font-medium tracking-[0.01em] text-slate-500">{label}</p>
          <div className="min-w-0 space-y-1">
            <p className="max-w-full break-words text-[clamp(1.7rem,3.6vw,2.05rem)] font-bold leading-[0.98] tracking-tight text-slate-900">
              {value}
            </p>
            {detail ? <p className="max-w-none text-sm leading-6 text-slate-600 md:max-w-[32ch]">{detail}</p> : null}
          </div>
        </div>
        {icon ? <div className={joinClasses('metric-icon-wrap shrink-0', toneMap[tone])}>{icon}</div> : null}
      </div>
    </article>
  )
}

export function StatusBadge({
  label,
  tone,
  pulse = false,
}: {
  label: string
  tone: Tone
  pulse?: boolean
}) {
  return (
    <span className={joinClasses('pill', toneMap[tone])}>
      <span
        className={joinClasses(
          'h-2.5 w-2.5 rounded-full',
          tone === 'teal' && 'bg-teal-600',
          tone === 'amber' && 'bg-amber-600',
          tone === 'red' && 'bg-rose-600',
          tone === 'blue' && 'bg-blue-600',
          tone === 'slate' && 'bg-slate-500',
          pulse && 'animate-pulse',
        )}
      />
      {label}
    </span>
  )
}

export function ClassificationBadge({
  type,
  compact = false,
}: {
  type: string
  compact?: boolean
}) {
  const map: Record<string, string> = {
    FRAUD: 'bg-rose-100 text-rose-800',
    RARITY: 'bg-emerald-100 text-emerald-800',
    HONEST: 'bg-sky-100 text-sky-800',
    NOISE: 'bg-slate-200 text-slate-700',
  }

  return (
    <span
      className={joinClasses(
        'inline-flex items-center rounded-full font-semibold',
        compact ? 'px-2.5 py-1 text-[11px]' : 'px-3 py-1.5 text-xs',
        map[type] || map.NOISE,
      )}
    >
      {formatClassificationLabel(type, compact)}
    </span>
  )
}

export function EmptyState({
  icon,
  title,
  description,
}: {
  icon: ReactNode
  title: string
  description: string
}) {
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-4 px-6 py-10 text-center">
      <div className="metric-icon-wrap bg-slate-100/80 text-slate-500">{icon}</div>
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
        <p className="max-w-md text-sm leading-6 text-slate-500">{description}</p>
      </div>
    </div>
  )
}

export function StateCallout({
  icon,
  title,
  description,
  tone = 'slate',
}: {
  icon?: ReactNode
  title: string
  description: string
  tone?: Tone
}) {
  return (
    <div className={joinClasses('rounded-[20px] border px-5 py-4', toneSurfaceMap[tone])}>
      <div className="flex items-start gap-3">
        {icon ? <div className={joinClasses('metric-icon-wrap', toneMap[tone])}>{icon}</div> : null}
        <div className="space-y-1.5">
          <p className="font-semibold text-slate-900">{title}</p>
          <p className="text-sm leading-6 text-slate-600">{description}</p>
        </div>
      </div>
    </div>
  )
}

export function MetricBar({
  label,
  value,
  colorClass,
}: {
  label: string
  value: number
  colorClass: string
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-slate-600">{label}</span>
        <span className="font-mono text-slate-900">{value.toFixed(1)}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-slate-200/70">
        <div
          className={joinClasses('h-full rounded-full transition-all', colorClass)}
          style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }}
        />
      </div>
    </div>
  )
}
