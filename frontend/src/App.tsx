import { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/layout/Layout'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Architecture = lazy(() => import('./pages/Architecture'))
const Vehicles = lazy(() => import('./pages/Vehicles'))
const Audits = lazy(() => import('./pages/Audits'))
const Settlement = lazy(() => import('./pages/Settlement'))
const PolicyDashboard = lazy(() => import('./pages/PolicyDashboard'))
const DataAnalysis = lazy(() => import('./pages/DataAnalysis'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: true,
      retry: 1,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Suspense fallback={<RouteLoadingState />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/architecture" element={<Architecture />} />
              <Route path="/vehicles" element={<Vehicles />} />
              <Route path="/audits" element={<Audits />} />
              <Route path="/analysis" element={<DataAnalysis />} />
              <Route path="/settlement" element={<Settlement />} />
              <Route path="/policy" element={<PolicyDashboard />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

function RouteLoadingState() {
  return (
    <div className="page-shell">
      <section className="page-hero">
        <p className="section-label">Loading</p>
        <div className="mt-4 max-w-xl space-y-3">
          <div className="h-10 w-64 animate-pulse rounded-2xl bg-slate-200/70" />
          <div className="h-5 w-full animate-pulse rounded-2xl bg-slate-200/60" />
          <div className="h-5 w-5/6 animate-pulse rounded-2xl bg-slate-200/60" />
        </div>
      </section>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="metric-tile animate-pulse">
            <div className="h-4 w-24 rounded bg-slate-200/70" />
            <div className="mt-4 h-10 w-28 rounded bg-slate-300/70" />
            <div className="mt-3 h-4 w-36 rounded bg-slate-200/60" />
          </div>
        ))}
      </div>
    </div>
  )
}
