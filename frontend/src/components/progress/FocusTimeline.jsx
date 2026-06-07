export default function FocusTimeline({ periods, skillConfig }) {
  const colorBySkill = Object.fromEntries(skillConfig.map(s => [s.key, s.color]));
  const labelBySkill = Object.fromEntries(skillConfig.map(s => [s.key, s.label]));

  if (!periods.length) return null;

  return (
    <div>
      <h3 className="text-xs font-semibold mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>
        Session Focus History
      </h3>

      {/* Scrollable dot timeline */}
      <div className="overflow-x-auto">
        <div className="flex items-start gap-0 min-w-max">
          {periods.map((period, i) => {
            const color = period.focus_skill
              ? colorBySkill[period.focus_skill] ?? 'rgba(148,163,184,0.3)'
              : 'rgba(148,163,184,0.2)';
            const label = period.focus_skill ? labelBySkill[period.focus_skill] : null;

            return (
              <div key={period.period_key} className="flex items-center">
                {/* Dot + label stacked */}
                <div className="flex flex-col items-center" style={{ width: 36 }}>
                  <div
                    className="w-3 h-3 rounded-full transition-all"
                    style={{
                      background: color,
                      boxShadow: period.focus_skill ? `0 0 7px ${color}70` : 'none',
                    }}
                    title={label ? `Focus: ${label}` : 'Free session'}
                  />
                  <span
                    className="text-center mt-1.5 leading-none"
                    style={{ color: 'rgba(148,163,184,0.35)', fontSize: 10, width: 32 }}
                  >
                    {i + 1}
                  </span>
                </div>
                {/* Connector line to next dot */}
                {i < periods.length - 1 && (
                  <div
                    style={{
                      width: 12,
                      height: 1,
                      background: 'rgba(255,255,255,0.07)',
                      marginBottom: 14,
                      flexShrink: 0,
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-3">
        {skillConfig
          .filter(s => periods.some(p => p.focus_skill === s.key))
          .map(s => (
            <div key={s.key} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: s.color }} />
              <span className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>{s.label}</span>
            </div>
          ))}
        {periods.some(p => !p.focus_skill) && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(148,163,184,0.2)' }} />
            <span className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>Free session</span>
          </div>
        )}
      </div>
    </div>
  );
}
