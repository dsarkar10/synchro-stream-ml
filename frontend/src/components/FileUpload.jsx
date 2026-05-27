import { useState } from 'react'
import { Upload, Database, Loader2 } from 'lucide-react'

export default function FileUpload({ onProfile, loading }) {
  const [numFeatures, setNumFeatures] = useState(10)
  const [numSamples, setNumSamples] = useState(32)

  const handleProfile = () => {
    onProfile({ num_features: numFeatures, num_samples: numSamples, memory_samples: 32 })
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-green-400" />
        Ingestion Profiler
      </h3>

      <div className="space-y-3">
        <div>
          <label className="text-xs text-slate-400 block mb-1">Number of Features</label>
          <input
            type="range"
            min="4"
            max="20"
            value={numFeatures}
            onChange={(e) => setNumFeatures(Number(e.target.value))}
            className="w-full accent-cyan-500"
          />
          <span className="text-xs text-slate-500">{numFeatures} features</span>
        </div>

        <div>
          <label className="text-xs text-slate-400 block mb-1">Batch Size (Samples)</label>
          <input
            type="range"
            min="8"
            max="128"
            step="8"
            value={numSamples}
            onChange={(e) => setNumSamples(Number(e.target.value))}
            className="w-full accent-cyan-500"
          />
          <span className="text-xs text-slate-500">{numSamples} samples</span>
        </div>

        <div className="flex items-center gap-2 p-2 bg-slate-800/50 rounded-lg border border-dashed border-slate-600">
          <Database className="w-4 h-4 text-slate-500" />
          <span className="text-xs text-slate-500">Using synthetic data generator</span>
        </div>

        <button
          onClick={handleProfile}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-sm font-semibold text-white transition-all"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Upload className="w-4 h-4" />
          )}
          {loading ? 'Analyzing...' : 'Profile Ingestion'}
        </button>
      </div>
    </div>
  )
}
