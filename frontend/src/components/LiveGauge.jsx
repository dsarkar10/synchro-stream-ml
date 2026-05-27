export default function LiveGauge({ label, value, color, suffix = '%' }) {
  const pct = Math.round(value * 100)
  const dashArray = 251.2
  const dashOffset = dashArray - (pct / 100) * dashArray

  return (
    <div className="flex flex-col items-center">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" fill="none" stroke="#1e293b" strokeWidth="8" />
        <circle
          cx="50" cy="50" r="40"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={dashArray}
          strokeDashoffset={dashOffset}
          transform="rotate(-90 50 50)"
          className="transition-all duration-700 ease-out"
        />
        <text x="50" y="50" textAnchor="middle" dy="5" className="text-xl font-bold" fill="#e2e8f0">
          {pct}
        </text>
      </svg>
      <p className="text-xs font-medium text-slate-400 mt-1">{label}</p>
    </div>
  )
}
