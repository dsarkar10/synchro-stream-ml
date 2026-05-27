import { useState } from 'react'
import FileUpload from './components/FileUpload'
import ConflictHeatmap from './components/ConflictHeatmap'
import RecommendationPanel from './components/RecommendationPanel'
import StrategySimulator from './components/StrategySimulator'
import ModelArchitecture from './components/ModelArchitecture'
import { BarChart3, GitBranch } from 'lucide-react'

const API = 'http://localhost:8000'

function generateSyntheticHeatmap(npsScore) {
  const features = 10
  const layers = 4
  return Array.from({ length: features }, () =>
    Array.from({ length: layers }, () =>
      Math.min(1, Math.max(0, (Math.random() * 0.6 + (npsScore) * 0.4)))
    )
  )
}

export default function App() {
  const [profileResult, setProfileResult] = useState(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [simMetrics, setSimMetrics] = useState(null)
  const [simLoading, setSimLoading] = useState(false)

  const handleProfile = async (params) => {
    setProfileLoading(true)
    try {
      const res = await fetch(`${API}/profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      const data = await res.json()
      setProfileResult(data)
    } catch (err) {
      console.error('Profile failed, using fallback:', err)
      const fallback = {
        nps_score: 0.65,
        layer_disturbance: [0.4, 0.65, 0.55, 0.3],
        layer_names: ['Input', 'Hidden 1', 'Hidden 2', 'Output'],
        recommended_strategy: {
          strategy: 'Interleaved Mini-Batch',
          safety: 'medium',
          description: 'Interleaved old/new data in small batches. Balanced approach.',
          plasticity: 0.55,
          stability: 0.65,
          throughput: 0.60,
        },
      }
      setProfileResult(fallback)
    }
    setProfileLoading(false)
  }

  const handleSimulate = async (strategy) => {
    setSimLoading(true)
    try {
      const res = await fetch(`${API}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy }),
      })
      const data = await res.json()
      setSimMetrics(data)
    } catch (err) {
      console.error('Simulate failed, using fallback:', err)
      const fallbacks = {
        linear: { plasticity: 0.25, stability: 0.95, throughput: 0.3, strategy: 'Buffered Linear Ingestion', safety: 'high', description: 'Sequential processing with large memory buffer.' },
        interleaved: { plasticity: 0.55, stability: 0.65, throughput: 0.6, strategy: 'Interleaved Mini-Batch', safety: 'medium', description: 'Interleaved old/new data in small batches.' },
        parallel: { plasticity: 0.9, stability: 0.25, throughput: 0.95, strategy: 'High-Speed Parallel', safety: 'low', description: 'Fully parallel batch ingestion.' },
      }
      setSimMetrics(fallbacks[strategy] || fallbacks.interleaved)
    }
    setSimLoading(false)
  }

  const npsScore = profileResult?.nps_score ?? 0.5
  const layerDisturbance = profileResult?.layer_disturbance ?? [0.5, 0.5, 0.5, 0.5]
  const layerNames = profileResult?.layer_names ?? ['Input', 'Hidden 1', 'Hidden 2', 'Output']
  const strategy = profileResult?.recommended_strategy
  const heatmapData = generateSyntheticHeatmap(npsScore)

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">SynchroStream-ML</h1>
              <p className="text-[10px] text-slate-500">Decision Support System for MLOps</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-600">Data-to-Model Conflict Analyzer</span>
            <a href="#" className="text-slate-600 hover:text-slate-400 transition-colors">
              <GitBranch className="w-4 h-4" />
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-4">
          {/* Left sidebar */}
          <div className="col-span-12 lg:col-span-3 space-y-4">
            <FileUpload onProfile={handleProfile} loading={profileLoading} />
            <RecommendationPanel npsScore={npsScore} strategy={strategy} />
          </div>

          {/* Main content */}
          <div className="col-span-12 lg:col-span-6 space-y-4">
            <ConflictHeatmap data={heatmapData} layerNames={layerNames} />
            <ModelArchitecture disturbance={layerDisturbance} />
          </div>

          {/* Right sidebar */}
          <div className="col-span-12 lg:col-span-3 space-y-4">
            <StrategySimulator
              onSimulate={handleSimulate}
              metrics={simMetrics}
              loading={simLoading}
            />

            {/* NPS History / Info Panel */}
            <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                About NPS
              </h3>
              <div className="space-y-2 text-xs text-slate-400">
                <p>
                  <strong className="text-slate-300">Neural Perturbation Score (NPS)</strong>{' '}
                  measures gradient conflict between old and new data.
                </p>
                <ul className="space-y-1 list-disc list-inside">
                  <li><span className="text-green-400">&lt; 0.3</span> — Low conflict → Parallel</li>
                  <li><span className="text-yellow-400">0.3 – 0.7</span> — Moderate → Interleaved</li>
                  <li><span className="text-red-400">&gt; 0.7</span> — High conflict → Buffered Linear</li>
                </ul>
                <p className="text-slate-600 pt-1 border-t border-slate-800 mt-2">
                  Research tool for studying catastrophic forgetting in recurring data streams.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
