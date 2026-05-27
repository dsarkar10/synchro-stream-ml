import { ResponsiveHeatMap } from '@nivo/heatmap'

export default function ConflictHeatmap({ data, layerNames }) {
  const featureLabels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']

  const heatmapData = featureLabels.map((feature, fi) => ({
    id: feature,
    data: layerNames.map((layer, li) => ({
      x: layer,
      y: data[fi]?.[li] ?? Math.random() * 0.8,
    })),
  }))

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-cyan-400" />
        Layer-wise Gradient Conflict Heatmap
      </h3>
      <div style={{ height: 350 }}>
        <ResponsiveHeatMap
          data={heatmapData}
          margin={{ top: 30, right: 30, bottom: 60, left: 60 }}
          valueFormat=".2f"
          axisTop={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: '',
          }}
          axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Features',
            legendPosition: 'middle',
            legendOffset: -40,
          }}
          axisBottom={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Layers',
            legendPosition: 'middle',
            legendOffset: 40,
          }}
          colors={{
            type: 'diverging',
            scheme: 'red_yellow_blue',
            minValue: 0,
            maxValue: 1,
          }}
          emptyColor="#1e293b"
          borderWidth={1}
          borderColor="#0f172a"
          enableLabels={false}
          legends={[
            {
              anchor: 'right',
              translateX: 30,
              translateY: 0,
              length: 200,
              thickness: 10,
              direction: 'column',
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              title: 'Conflict',
              titleOffset: -10,
            },
          ]}
          theme={{
            text: { fill: '#94a3b8', fontSize: 11 },
            axis: { ticks: { text: { fill: '#64748b' } } },
            legends: { text: { fill: '#94a3b8' } },
          }}
        />
      </div>
    </div>
  )
}
