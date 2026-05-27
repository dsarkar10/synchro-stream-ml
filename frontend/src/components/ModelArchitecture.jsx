import { Cpu, Layers, ArrowRight, CircuitBoard } from 'lucide-react'

const layers = [
  { name: 'Input', neurons: 24, icon: Cpu, color: 'text-cyan-400', bg: 'bg-cyan-500/10 border-cyan-500/30' },
  { name: 'Hidden 1', neurons: 16, icon: Layers, color: 'text-violet-400', bg: 'bg-violet-500/10 border-violet-500/30' },
  { name: 'Hidden 2', neurons: 8, icon: Layers, color: 'text-fuchsia-400', bg: 'bg-fuchsia-500/10 border-fuchsia-500/30' },
  { name: 'Output', neurons: 2, icon: CircuitBoard, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30' },
]

export default function ModelArchitecture({ disturbance }) {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-violet-400" />
        Model Architecture
      </h3>
      <div className="flex items-center justify-center gap-1 overflow-x-auto py-2">
        {layers.map((layer, i) => {
          const Icon = layer.icon
          const dist = disturbance?.[i] ?? 0
          const distColor =
            dist > 0.7 ? 'text-red-400' :
            dist > 0.3 ? 'text-yellow-400' :
            'text-green-400'

          return (
            <div key={layer.name} className="flex items-center gap-1">
              <div className={`flex flex-col items-center p-3 rounded-lg border ${layer.bg} min-w-[80px]`}>
                <Icon className={`w-5 h-5 ${layer.color} mb-1`} />
                <span className="text-xs font-medium text-slate-300">{layer.name}</span>
                <span className="text-[10px] text-slate-500">{layer.neurons}n</span>
                <span className={`text-[10px] font-bold ${distColor}`}>Δ {dist.toFixed(2)}</span>
              </div>
              {i < layers.length - 1 && <ArrowRight className="w-4 h-4 text-slate-600 shrink-0" />}
            </div>
          )
        })}
      </div>
    </div>
  )
}
