import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-xl p-3 text-xs"
      style={{
        background: 'rgba(8,14,28,0.97)',
        border: '1px solid rgba(255,255,255,0.1)',
        minWidth: 140,
      }}
    >
      <p className="font-semibold text-white mb-2">{label}</p>
      {payload
        .filter(e => e.value != null)
        .sort((a, b) => b.value - a.value)
        .map(entry => (
          <div key={entry.dataKey} className="flex items-center justify-between gap-4 mb-1">
            <span style={{ color: entry.color }}>{entry.name}</span>
            <span className="font-bold text-white">{Math.round(entry.value)}</span>
          </div>
        ))}
    </div>
  );
};

export default function SkillTrendChart({ periods, selectedSkill, skillConfig }) {
  const chartData = periods.map(p => {
    const row = { label: p.label };
    for (const s of skillConfig) {
      row[s.key] = p.scores[s.key] != null ? Math.round(p.scores[s.key]) : null;
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }}
          axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        {skillConfig.map(s => {
          const active = !selectedSkill || selectedSkill === s.key;
          return (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={active ? 2 : 0.8}
              strokeOpacity={active ? 1 : 0.18}
              dot={{ fill: s.color, r: active ? 3 : 1.5, strokeWidth: 0 }}
              activeDot={{ r: 5, strokeWidth: 0 }}
              connectNulls={false}
            />
          );
        })}
      </LineChart>
    </ResponsiveContainer>
  );
}
