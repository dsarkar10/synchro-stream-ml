import { Shield, AlertTriangle, Zap, BrainCircuit } from 'lucide-react'

const strategyConfig = {
  'Buffered Linear Ingestion': { color: 'red', icon: Shield, label: 'High Safety' },
  'Interleaved Mini-Batch': { color: 'yellow', icon: AlertTriangle, label: 'Balanced' },
  'High-Speed Parallel': { color: 'green', icon: Zap, label: 'Maximum Speed' },
}

export default function RecommendationPanel({ npsScore, strategy }) {
  const config = strategyConfig[strategy?.strategy] || { color: 'gray', icon: BrainCircuit, label: 'Unknown' }
  const Icon = config.icon

  const glowColor =
    config.color === 'red' ? 'shadow-red-500/40' :
    config.color === 'yellow' ? 'shadow-yellow-500/40' :
    config.color === 'green' ? 'shadow-green-500/40' :
    'shadow-slate-500/40'

  const borderColor =
    config.color === 'red' ? 'border-red-500/50' :
    config.color === 'yellow' ? 'border-yellow-500/50' :
    config.color === 'green' ? 'border-green-500/50' :
    'border-slate-500/50'

  const bgColor =
    config.color === 'red' ? 'from-red-900/40 to-red-950/40' :
    config.color === 'yellow' ? 'from-yellow-900/40 to-yellow-950/40' :
    config.color === 'green' ? 'from-green-900/40 to-green-950/40' :
    'from-slate-900/40 to-slate-950/40'

  return (
    <div className={`bg-gradient-to-br ${bgColor} rounded-xl border ${borderColor} p-5 shadow-lg ${glowColor} transition-all duration-500`}>
      <div className="flex items-start gap-4">
        <div className={`p-3 rounded-lg ${
          config.color === 'red' ? 'bg-red-500/20 text-red-400' :
          config.color === 'yellow' ? 'bg-yellow-500/20 text-yellow-400' :
          'bg-green-500/20 text-green-400'
        }`}>
          <Icon className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400 mb-1">Recommended Strategy</p>
          <h2 className={`text-xl font-bold ${
            config.color === 'red' ? 'text-red-300' :
            config.color === 'yellow' ? 'text-yellow-300' :
            'text-green-300'
          }`}>
            {strategy?.strategy || 'Analyzing...'}
          </h2>
          <p className="text-sm text-slate-400 mt-1">{strategy?.description}</p>
        </div>
      </div>

      <div className="mt-4 flex gap-4">
        <div className="flex-1 bg-slate-900/60 rounded-lg p-3">
          <p className="text-xs text-slate-500 mb-1">NPS Score</p>
          <p className={`text-2xl font-bold ${
            npsScore > 0.7 ? 'text-red-400' :
            npsScore > 0.3 ? 'text-yellow-400' :
            'text-green-400'
          }`}>
            {npsScore.toFixed(2)}
          </p>
        </div>
        <div className="flex-1 bg-slate-900/60 rounded-lg p-3">
          <p className="text-xs text-slate-500 mb-1">Safety Level</p>
          <p className="text-lg font-semibold text-slate-300 capitalize">{strategy?.safety || '—'}</p>
        </div>
      </div>

      {strategy?.buffer_resize && (
        <div className="mt-3 bg-blue-900/30 border border-blue-500/30 rounded-lg p-3 text-sm text-blue-300">
          <p className="font-medium mb-1">🧠 Dynamic Buffer Resizing</p>
          <p>{strategy.buffer_resize.message}</p>
        </div>
      )}
    </div>
  )
}
