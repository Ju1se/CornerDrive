import { EmptyState } from '../ui/primitives'

function formatTick(value: number, digits = 1) {
  return Number.isInteger(value) ? String(value) : value.toFixed(digits)
}

function chartPath(values: number[], xForIndex: (index: number) => number, yForValue: (value: number) => number) {
  return values
    .map((value, index) => `${index === 0 ? 'M' : 'L'} ${xForIndex(index)} ${yForValue(value)}`)
    .join(' ')
}

function domainFromValues(values: number[], fallbackMin = 0, fallbackMax = 1) {
  if (!values.length) {
    return { min: fallbackMin, max: fallbackMax }
  }

  let min = Math.min(...values)
  let max = Math.max(...values)

  if (min === max) {
    const padding = min === 0 ? 1 : Math.abs(min) * 0.15
    min -= padding
    max += padding
  }

  return { min, max }
}

function valueToY(value: number, min: number, max: number, top: number, height: number) {
  const ratio = (value - min) / (max - min || 1)
  return top + height - ratio * height
}

export interface LineChartSeries {
  label: string
  color: string
  values: number[]
}

export function LineChart({
  labels,
  series,
  yAxisLabel,
  valueFormatter = (value: number) => formatTick(value),
  min,
  max,
}: {
  labels: string[]
  series: LineChartSeries[]
  yAxisLabel?: string
  valueFormatter?: (value: number) => string
  min?: number
  max?: number
}) {
  const width = 720
  const height = 280
  const padding = { top: 22, right: 18, bottom: 46, left: 56 }
  const innerWidth = width - padding.left - padding.right
  const innerHeight = height - padding.top - padding.bottom

  const flattened = series.flatMap((entry) => entry.values).filter((value) => Number.isFinite(value))
  if (!labels.length || !flattened.length) {
    return (
      <EmptyState
        icon={<div className="h-4 w-4 rounded-full bg-slate-400" />}
        title="No chart data yet"
        description="This chart will populate when the backend has enough live history to plot."
      />
    )
  }

  const domain = domainFromValues(flattened)
  const yMin = min ?? domain.min
  const yMax = max ?? domain.max
  const xForIndex = (index: number) => (
    padding.left + (labels.length === 1 ? innerWidth / 2 : (index / (labels.length - 1)) * innerWidth)
  )
  const yForValue = (value: number) => valueToY(value, yMin, yMax, padding.top, innerHeight)

  const tickValues = Array.from({ length: 4 }, (_, index) => yMin + ((yMax - yMin) / 3) * index)
  const step = labels.length > 12 ? Math.ceil(labels.length / 6) : 1

  return (
    <div className="space-y-4">
      {yAxisLabel ? <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{yAxisLabel}</p> : null}
      <div className="flex flex-wrap gap-3">
        {series.map((entry) => (
          <div key={entry.label} className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
            {entry.label}
          </div>
        ))}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full overflow-visible">
        {tickValues.map((tick) => {
          const y = yForValue(tick)
          return (
            <g key={tick}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#cbd5e1" strokeDasharray="4 6" />
              <text x={padding.left - 10} y={y + 4} textAnchor="end" fontSize="11" fill="#64748b">
                {valueFormatter(tick)}
              </text>
            </g>
          )
        })}
        <line x1={padding.left} y1={padding.top + innerHeight} x2={width - padding.right} y2={padding.top + innerHeight} stroke="#94a3b8" />
        {labels.map((label, index) => {
          if (index % step !== 0 && index !== labels.length - 1) return null
          return (
            <text
              key={`${label}-${index}`}
              x={xForIndex(index)}
              y={height - 14}
              textAnchor="middle"
              fontSize="11"
              fill="#64748b"
            >
              {label}
            </text>
          )
        })}
        {series.map((entry) => (
          <g key={entry.label}>
            <path
              d={chartPath(entry.values, xForIndex, yForValue)}
              fill="none"
              stroke={entry.color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {entry.values.map((value, index) => (
              <circle key={`${entry.label}-${index}`} cx={xForIndex(index)} cy={yForValue(value)} r="3.5" fill={entry.color} />
            ))}
          </g>
        ))}
      </svg>
    </div>
  )
}

export function GroupedBarChart({
  groups,
  max = 100,
  valueFormatter = (value: number) => `${value.toFixed(1)}%`,
}: {
  groups: Array<{
    label: string
    values: Array<{ series: string; color: string; value: number }>
  }>
  max?: number
  valueFormatter?: (value: number) => string
}) {
  const width = 720
  const height = 300
  const padding = { top: 20, right: 18, bottom: 58, left: 52 }
  const innerWidth = width - padding.left - padding.right
  const innerHeight = height - padding.top - padding.bottom

  if (!groups.length || groups.every((group) => group.values.length === 0)) {
    return (
      <EmptyState
        icon={<div className="h-4 w-4 rounded-full bg-slate-400" />}
        title="No grouped data yet"
        description="This comparison chart appears once the backend returns evaluation results."
      />
    )
  }

  const uniqueSeries = Array.from(new Set(groups.flatMap((group) => group.values.map((item) => item.series))))
  const groupWidth = innerWidth / Math.max(groups.length, 1)
  const barWidth = Math.min(24, (groupWidth - 18) / Math.max(uniqueSeries.length, 1))

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        {uniqueSeries.map((seriesLabel) => {
          const sample = groups.flatMap((group) => group.values).find((item) => item.series === seriesLabel)
          return (
            <div key={seriesLabel} className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: sample?.color ?? '#64748b' }} />
              {seriesLabel}
            </div>
          )
        })}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full overflow-visible">
        {[0, max * 0.25, max * 0.5, max * 0.75, max].map((tick) => {
          const y = valueToY(tick, 0, max, padding.top, innerHeight)
          return (
            <g key={tick}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#cbd5e1" strokeDasharray="4 6" />
              <text x={padding.left - 10} y={y + 4} textAnchor="end" fontSize="11" fill="#64748b">
                {formatTick(tick)}
              </text>
            </g>
          )
        })}
        {groups.map((group, groupIndex) => {
          const xStart = padding.left + groupIndex * groupWidth + (groupWidth - barWidth * group.values.length) / 2
          return (
            <g key={group.label}>
              {group.values.map((entry, barIndex) => {
                const barHeight = (Math.max(entry.value, 0) / max) * innerHeight
                const x = xStart + barIndex * barWidth
                const y = padding.top + innerHeight - barHeight
                return (
                  <g key={`${group.label}-${entry.series}`}>
                    <rect x={x} y={y} width={barWidth - 4} height={barHeight} rx="6" fill={entry.color} />
                    <text x={x + (barWidth - 4) / 2} y={y - 8} textAnchor="middle" fontSize="10" fill="#334155">
                      {valueFormatter(entry.value)}
                    </text>
                  </g>
                )
              })}
              <text x={padding.left + groupIndex * groupWidth + groupWidth / 2} y={height - 14} textAnchor="middle" fontSize="11" fill="#64748b">
                {group.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export function ScatterPlot({
  points,
  xLabel,
  yLabel,
  xFormatter = (value: number) => formatTick(value),
  yFormatter = (value: number) => formatTick(value),
}: {
  points: Array<{ label: string; color: string; x: number; y: number }>
  xLabel: string
  yLabel: string
  xFormatter?: (value: number) => string
  yFormatter?: (value: number) => string
}) {
  const width = 720
  const height = 300
  const padding = { top: 22, right: 22, bottom: 52, left: 56 }
  const innerWidth = width - padding.left - padding.right
  const innerHeight = height - padding.top - padding.bottom

  if (!points.length) {
    return (
      <EmptyState
        icon={<div className="h-4 w-4 rounded-full bg-slate-400" />}
        title="No scatter data yet"
        description="This plot appears when the backend has comparison points to map."
      />
    )
  }

  const xDomain = domainFromValues(points.map((point) => point.x))
  const yDomain = domainFromValues(points.map((point) => point.y))
  const xForValue = (value: number) => padding.left + ((value - xDomain.min) / (xDomain.max - xDomain.min || 1)) * innerWidth
  const yForValue = (value: number) => valueToY(value, yDomain.min, yDomain.max, padding.top, innerHeight)

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        {points.map((point) => (
          <div key={point.label} className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: point.color }} />
            {point.label}
          </div>
        ))}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((step) => {
          const x = padding.left + step * innerWidth
          const y = padding.top + step * innerHeight
          return (
            <g key={step}>
              <line x1={x} y1={padding.top} x2={x} y2={height - padding.bottom} stroke="#e2e8f0" />
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#e2e8f0" />
            </g>
          )
        })}
        <line x1={padding.left} y1={padding.top + innerHeight} x2={width - padding.right} y2={padding.top + innerHeight} stroke="#94a3b8" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + innerHeight} stroke="#94a3b8" />
        <text x={width / 2} y={height - 12} textAnchor="middle" fontSize="12" fill="#475569">
          {xLabel}
        </text>
        <text x={20} y={height / 2} textAnchor="middle" fontSize="12" fill="#475569" transform={`rotate(-90 20 ${height / 2})`}>
          {yLabel}
        </text>
        <text x={padding.left} y={height - 30} fontSize="11" fill="#64748b">{xFormatter(xDomain.min)}</text>
        <text x={width - padding.right} y={height - 30} textAnchor="end" fontSize="11" fill="#64748b">{xFormatter(xDomain.max)}</text>
        <text x={padding.left - 10} y={padding.top + innerHeight + 4} textAnchor="end" fontSize="11" fill="#64748b">{yFormatter(yDomain.min)}</text>
        <text x={padding.left - 10} y={padding.top + 4} textAnchor="end" fontSize="11" fill="#64748b">{yFormatter(yDomain.max)}</text>
        {points.map((point) => {
          const x = xForValue(point.x)
          const y = yForValue(point.y)
          return (
            <g key={point.label}>
              <circle cx={x} cy={y} r="7" fill={point.color} />
              <text x={x + 10} y={y - 10} fontSize="11" fill="#334155">
                {point.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export function StackedBarChart({
  bars,
}: {
  bars: Array<{
    label: string
    segments: Array<{ label: string; color: string; value: number }>
  }>
}) {
  if (!bars.length) {
    return (
      <EmptyState
        icon={<div className="h-4 w-4 rounded-full bg-slate-400" />}
        title="No round mix history yet"
        description="Run a few telemetry rounds and this mix chart will fill in from backend history."
      />
    )
  }

  const legend = Array.from(
    new Map(
      bars.flatMap((bar) => bar.segments).map((segment) => [segment.label, segment]),
    ).values(),
  )

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        {legend.map((segment) => (
          <div key={segment.label} className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: segment.color }} />
            {segment.label}
          </div>
        ))}
      </div>
      <div className="overflow-x-auto pb-2">
        <div className="flex min-w-max items-end gap-3">
          {bars.map((bar) => (
            <div key={bar.label} className="w-16 shrink-0 space-y-2">
              <div className="flex h-52 flex-col overflow-hidden rounded-[18px] border border-slate-200 bg-slate-100/70">
                {bar.segments.map((segment) => (
                  <div
                    key={`${bar.label}-${segment.label}`}
                    className="w-full transition-all"
                    style={{
                      height: `${Math.max(segment.value, 0) * 100}%`,
                      backgroundColor: segment.color,
                    }}
                    title={`${segment.label}: ${(segment.value * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <p className="text-center text-xs font-medium text-slate-600">{bar.label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
