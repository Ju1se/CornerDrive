import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  BarChart3,
  Compass,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react'

import { GroupedBarChart, LineChart, ScatterPlot, StackedBarChart } from '../components/analysis/Charts'
import { MetricTile, PageIntro, Panel, StateCallout, StatusBadge } from '../components/ui/primitives'
import {
  fetchBaselineAnalysis,
  fetchCurrentPolicy,
  fetchPolicyHistory,
  fetchTelemetryHistory,
} from '../services/api'

const baselineColors: Record<string, string> = {
  fedavg: '#e11d48',
  l1_only: '#f59e0b',
  static_l2: '#2563eb',
  adaptive: '#0f766e',
}

function formatPercent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`
}

function formatPolicyValue(value: number) {
  return value.toFixed(3)
}

function baselineColor(id: string) {
  return baselineColors[id] ?? '#64748b'
}

export default function DataAnalysis() {
  const {
    data: telemetryHistory,
    isLoading: telemetryLoading,
    isError: telemetryError,
  } = useQuery({
    queryKey: ['telemetryHistory', 24],
    queryFn: () => fetchTelemetryHistory(24),
    refetchInterval: 15000,
    staleTime: 5000,
    placeholderData: (previousData) => previousData,
  })

  const {
    data: policyHistory,
    isLoading: policyHistoryLoading,
    isError: policyHistoryError,
  } = useQuery({
    queryKey: ['policyHistory', 24],
    queryFn: () => fetchPolicyHistory(24),
    refetchInterval: 15000,
    staleTime: 5000,
    placeholderData: (previousData) => previousData,
  })

  const {
    data: currentPolicy,
    isLoading: currentPolicyLoading,
    isError: currentPolicyError,
  } = useQuery({
    queryKey: ['currentPolicy'],
    queryFn: fetchCurrentPolicy,
    refetchInterval: 10000,
    staleTime: 5000,
    placeholderData: (previousData) => previousData,
  })

  const baselineEnabled =
    Boolean(telemetryHistory?.length || policyHistory?.length || currentPolicy) ||
    (!telemetryLoading && !policyHistoryLoading && !currentPolicyLoading)

  const {
    data: baselineAnalysis,
    isLoading: baselineLoading,
    isError: baselineError,
  } = useQuery({
    queryKey: ['baselineAnalysis', 12],
    queryFn: () => fetchBaselineAnalysis(12),
    enabled: baselineEnabled,
    refetchInterval: 120000,
    staleTime: 60000,
    retry: 0,
    placeholderData: (previousData) => previousData,
  })

  const telemetry = useMemo(
    () => [...(telemetryHistory ?? [])].sort((left, right) => left.round_id - right.round_id),
    [telemetryHistory],
  )

  const policies = useMemo(() => {
    const merged = [...(policyHistory ?? [])]
    if (currentPolicy && !merged.some((policy) => policy.round_id === currentPolicy.round_id)) {
      merged.unshift(currentPolicy)
    }
    return merged.sort((left, right) => left.round_id - right.round_id)
  }, [currentPolicy, policyHistory])

  const baselines = useMemo(() => baselineAnalysis?.baselines ?? [], [baselineAnalysis])
  const baselineRoundLabels = baselines[0]?.rounds.map((entry) => `R${entry.round_id}`) ?? []
  const phaseLabels = baselines[0]?.rounds.map((entry) => entry.phase) ?? []

  const mainSeries = useMemo(
    () =>
      baselines.map((baseline) => ({
        label: baseline.label,
        color: baselineColor(baseline.id),
        values: baseline.rounds.map((entry) => entry.main_accuracy * 100),
      })),
    [baselines],
  )

  const cornerSeries = useMemo(
    () =>
      baselines.map((baseline) => ({
        label: baseline.label,
        color: baselineColor(baseline.id),
        values: baseline.rounds.map((entry) => entry.corner_accuracy * 100),
      })),
    [baselines],
  )

  const groupedMetrics = useMemo(
    () => [
      {
        label: 'RARITY Precision',
        values: baselines.map((baseline) => ({
          series: baseline.label,
          color: baselineColor(baseline.id),
          value: baseline.summary.rarity_precision * 100,
        })),
      },
      {
        label: 'RARITY Recall',
        values: baselines.map((baseline) => ({
          series: baseline.label,
          color: baselineColor(baseline.id),
          value: baseline.summary.rarity_recall * 100,
        })),
      },
      {
        label: 'FRAUD Precision',
        values: baselines.map((baseline) => ({
          series: baseline.label,
          color: baselineColor(baseline.id),
          value: baseline.summary.fraud_precision * 100,
        })),
      },
      {
        label: 'FRAUD Recall',
        values: baselines.map((baseline) => ({
          series: baseline.label,
          color: baselineColor(baseline.id),
          value: baseline.summary.fraud_recall * 100,
        })),
      },
    ],
    [baselines],
  )

  const paretoPoints = useMemo(
    () =>
      baselines.map((baseline) => ({
        label: baseline.label,
        color: baselineColor(baseline.id),
        x: baseline.summary.main_accuracy_avg * 100,
        y: baseline.summary.corner_accuracy_avg * 100,
      })),
    [baselines],
  )

  const retentionPoints = useMemo(
    () =>
      baselines.map((baseline) => ({
        label: baseline.label,
        color: baselineColor(baseline.id),
        x: baseline.summary.false_slash_estimate_avg * 100,
        y: baseline.summary.rarity_retention_rate_avg * 100,
      })),
    [baselines],
  )

  const classificationMixBars = useMemo(
    () =>
      telemetry.map((entry) => ({
        label: `R${entry.round_id}`,
        segments: [
          { label: 'Fraud', color: '#e11d48', value: entry.fraud_rate },
          { label: 'Beneficial rarity', color: '#0f766e', value: entry.rarity_rate },
          { label: 'Honest', color: '#2563eb', value: entry.honest_rate },
          { label: 'Noise', color: '#64748b', value: entry.noise_rate },
        ],
      })),
    [telemetry],
  )

  const policyRoundLabels = policies.map((policy) => `R${policy.round_id}`)
  const latestTelemetry = telemetry[telemetry.length - 1]
  const latestPolicy = policies[policies.length - 1]
  const liveHistoryLoading = telemetryLoading || policyHistoryLoading || currentPolicyLoading
  const liveHistoryError = telemetryError || policyHistoryError || currentPolicyError

  function metricValue({
    value,
    loading,
    error,
    emptyLabel,
  }: {
    value: string | null
    loading: boolean
    error: boolean
    emptyLabel: string
  }) {
    if (value !== null) {
      return value
    }
    if (loading) {
      return 'Loading...'
    }
    if (error) {
      return 'Request failed'
    }
    return emptyLabel
  }

  return (
    <div className="space-y-6">
      <PageIntro
        eyebrow="Backend-driven Evidence"
        title="Data Analysis"
        description="This page separates live backend history from backend-computed baseline evaluation so the defense can show both operating telemetry and comparative evidence without relying on presentation-only snapshots."
        actions={
          <>
            <StatusBadge
              label={
                baselineAnalysis
                  ? `Scenario: ${baselineAnalysis.scenario_policy_source.replace('_', ' ')}`
                  : baselineLoading
                    ? 'Baseline loading'
                    : 'Baseline idle'
              }
              tone={baselineAnalysis ? 'teal' : baselineLoading ? 'blue' : 'amber'}
            />
            <StatusBadge
              label={liveHistoryLoading && telemetry.length === 0 ? 'Syncing live history' : `${telemetry.length} live rounds`}
              tone={telemetry.length > 0 ? 'blue' : liveHistoryLoading ? 'amber' : 'slate'}
            />
          </>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Latest Main Accuracy"
          value={metricValue({
            value: latestTelemetry ? formatPercent(latestTelemetry.main_accuracy) : null,
            loading: telemetryLoading,
            error: telemetryError,
            emptyLabel: 'No telemetry yet',
          })}
          detail="Most recent live telemetry snapshot from the policy backend."
          icon={<Activity size={18} />}
          tone="teal"
        />
        <MetricTile
          label="Latest Corner Accuracy"
          value={metricValue({
            value: latestTelemetry ? formatPercent(latestTelemetry.corner_accuracy) : null,
            loading: telemetryLoading,
            error: telemetryError,
            emptyLabel: 'No telemetry yet',
          })}
          detail="Used to show whether beneficial rarity is helping the difficult tail cases."
          icon={<Sparkles size={18} />}
          tone="blue"
        />
        <MetricTile
          label="Current theta_rare"
          value={metricValue({
            value: latestPolicy ? latestPolicy.theta_rare.toFixed(3) : null,
            loading: currentPolicyLoading || policyHistoryLoading,
            error: currentPolicyError || policyHistoryError,
            emptyLabel: 'No policy yet',
          })}
          detail="Active beneficial-rarity threshold currently loaded by the backend."
          icon={<SlidersHorizontal size={18} />}
          tone="amber"
        />
        <MetricTile
          label="Baseline Variants"
          value={metricValue({
            value: baselines.length > 0 ? String(baselines.length) : null,
            loading: baselineLoading,
            error: baselineError,
            emptyLabel: 'No analysis yet',
          })}
          detail="On-demand backend comparison runs used for defense slides and ablations."
          icon={<BarChart3 size={18} />}
          tone="slate"
        />
      </div>

      {liveHistoryLoading && telemetry.length === 0 && policies.length === 0 ? (
        <StateCallout
          icon={<Activity size={18} />}
          title="Live history is loading"
          description="The policy backend is responding, but these history queries can take a little while when the demo has just started or the backend is under load."
          tone="blue"
        />
      ) : null}

      {baselineError ? (
        <StateCallout
          icon={<ShieldAlert size={18} />}
          title="Baseline comparison request failed"
          description="The live history charts can still render from backend telemetry, but the on-demand baseline evaluation endpoint timed out or returned an error."
          tone="amber"
        />
      ) : null}

      {liveHistoryError ? (
        <StateCallout
          icon={<ShieldAlert size={18} />}
          title="Some live history requests failed"
          description="One or more backend history endpoints did not finish successfully, so this page is showing only the sections with confirmed live data."
          tone="amber"
        />
      ) : null}

      {!liveHistoryLoading && !liveHistoryError && telemetry.length === 0 ? (
        <StateCallout
          icon={<Compass size={18} />}
          title="No live telemetry yet"
          description="The page is connected, but the backend has not stored enough telemetry rounds yet. Let the demo run for a short while and then refresh this page."
          tone="slate"
        />
      ) : null}

      <Panel
        title="Main Task And Corner Task Trends"
        description="Backend-evaluated baselines plotted over the same round schedule. The main chart answers whether FLPG protects primary accuracy, and the corner chart answers whether it recovers beneficial rarity."
        action={
          baselineAnalysis ? (
            <div className="flex flex-wrap gap-2">
              {phaseLabels.slice(0, 6).map((phase, index) => (
                <span key={`${phase}-${index}`} className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                  {phase}
                </span>
              ))}
            </div>
          ) : null
        }
      >
        {baselineLoading && !baselineAnalysis ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="h-[320px] animate-pulse rounded-[24px] bg-slate-100/80" />
            <div className="h-[320px] animate-pulse rounded-[24px] bg-slate-100/80" />
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <LineChart
              labels={baselineRoundLabels}
              series={mainSeries}
              yAxisLabel="Main Accuracy"
              valueFormatter={(value) => `${value.toFixed(1)}%`}
              min={40}
              max={100}
            />
            <LineChart
              labels={baselineRoundLabels}
              series={cornerSeries}
              yAxisLabel="Corner Accuracy"
              valueFormatter={(value) => `${value.toFixed(1)}%`}
              min={15}
              max={100}
            />
          </div>
        )}
      </Panel>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel
          title="RARITY And FRAUD Recognition"
          description="Grouped backend evaluation metrics showing whether each baseline really separates beneficial rarity from fraud."
        >
          <GroupedBarChart groups={groupedMetrics} />
        </Panel>

        <Panel
          title="Pareto Frontier"
          description="Average main versus corner accuracy for each backend-evaluated baseline. The ideal system sits toward the top-right."
        >
          <ScatterPlot
            points={paretoPoints}
            xLabel="Average Main Accuracy"
            yLabel="Average Corner Accuracy"
            xFormatter={(value) => `${value.toFixed(1)}%`}
            yFormatter={(value) => `${value.toFixed(1)}%`}
          />
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel
          title="Rarity Retention Versus False Slash"
          description="This chart shows whether a baseline preserves beneficial rarity while keeping mistaken punishment low."
        >
          <ScatterPlot
            points={retentionPoints}
            xLabel="False Slash Estimate"
            yLabel="Beneficial-Rarity Retention"
            xFormatter={(value) => `${value.toFixed(1)}%`}
            yFormatter={(value) => `${value.toFixed(1)}%`}
          />
        </Panel>

        <Panel
          title="Live Classification Mix"
          description="Stacked round history from backend telemetry. This is the live operational view, not a separate analysis snapshot."
        >
          <StackedBarChart bars={classificationMixBars} />
        </Panel>
      </div>

      <Panel
        title="Policy Parameter Motion"
        description="Round-by-round policy history from the backend, showing how beneficial-rarity thresholds and incentives move over time."
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <LineChart
            labels={policyRoundLabels}
            series={[
              {
                label: 'theta_rare',
                color: '#0f766e',
                values: policies.map((policy) => policy.theta_rare),
              },
            ]}
            yAxisLabel="Beneficial-Rarity Threshold"
            valueFormatter={formatPolicyValue}
          />
          <LineChart
            labels={policyRoundLabels}
            series={[
              {
                label: 'rarity_reward_multiplier',
                color: '#2563eb',
                values: policies.map((policy) => policy.rarity_reward_multiplier),
              },
              {
                label: 'slash_multiplier',
                color: '#e11d48',
                values: policies.map((policy) => policy.slash_multiplier),
              },
              {
                label: 'corner_weight',
                color: '#f59e0b',
                values: policies.map((policy) => policy.corner_weight),
              },
            ]}
            yAxisLabel="Incentive And Weight Multipliers"
            valueFormatter={formatPolicyValue}
            min={0.4}
            max={2.05}
          />
        </div>
      </Panel>

      <Panel
        title="How To Read This Page"
        description="Use these panels as evidence slides: live history proves what the running system observed, while the baseline section proves why the full FLPG pipeline is more useful than simpler alternatives."
      >
        <div className="grid gap-4 md:grid-cols-3">
          <StateCallout
            icon={<Compass size={18} />}
            title="Live backend history"
            description="Telemetry and policy charts are pulled directly from the backend's persisted round history."
            tone="blue"
          />
          <StateCallout
            icon={<BarChart3 size={18} />}
            title="Backend-evaluated baselines"
            description="Baseline curves and comparison metrics are generated on demand by the backend evaluation engine."
            tone="teal"
          />
          <StateCallout
            icon={<ShieldAlert size={18} />}
            title="Interpretation boundary"
            description="These comparisons are backend-generated and simulation-backed, so present them as controlled evaluation rather than live fleet deployment."
            tone="amber"
          />
        </div>
      </Panel>
    </div>
  )
}
