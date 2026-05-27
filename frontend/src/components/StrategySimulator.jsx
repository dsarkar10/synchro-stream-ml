import { useState } from 'react'
import LiveGauge from './LiveGauge'
import { Play, RotateCcw } from 'lucide-react'

export default function StrategySimulator({ onSimulate, metrics, loading }) {
  const [strategy, setStrategy] = useState('interleaved')

  const strategies = [
    { key: 'linear', label: 'Linear + EWC', desc: 'Maximum stability, low speed' },
    { key: 'interleaved', label: 'Interleaved', desc: 'Balanced trade-off' },
    { key: 'parallel', label: 'High-Speed Parallel', desc: 'Maximum throughput' },
  ]

  const handleSimulate = () => {
    onSimulate(strategy)
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-amber-400" />
        Strategy Simulator
      </h3>

      <div className="flex gap-2 mb-4">
        {strategies.map((s) => (
          <button
            key={s.key}
            onClick={() => setStrategy(s.key)}
            className={`flex-1 p-2 rounded-lg text-xs font-medium border transition-all ${
              strategy === s.key
                ? 'bg-blue-600/30 border-blue-500/50 text-blue-300'
                : 'bg-slate-800 border-slate-600 text-slate-400 hover:border-slate-500'
            }`}
          >
            <div className="font-semibold">{s.label}</div>
            <div className="text-[10px] opacity-60 mt-0.5">{s.desc}</div>
          </button>
        ))}
      </div>

      <button
        onClick={handleSimulate}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-sm font-medium text-slate-200 transition-all mb-4"
      >
        {loading ? (
          <RotateCcw className="w-4 h-4 animate-spin" />
        ) : (
          <Play className="w-4 h-4" />
        )}
        {loading ? 'Simulating...' : 'Run Simulation'}
      </button>

      {metrics && (
        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-slate-700">
          <LiveGauge label="Plasticity" value={metrics.plasticity} color="#22c55e" />
          <LiveGauge label="Stability" value={metrics.stability} color="#3b82f6" />
          <LiveGauge label="Throughput" value={metrics.throughput} color="#f59e0b" />
        </div>
      )}
    </div>
  )
}
