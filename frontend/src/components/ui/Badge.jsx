/* Badge — status/label badge (Amethyst Precision palette) */

const VARIANTS = {
  active:    { color: '#a5d6a7',  bg: 'rgba(165,214,167,0.1)',  border: 'rgba(165,214,167,0.22)' },
  inactive:  { color: '#988d9d',  bg: 'rgba(152,141,157,0.1)',  border: 'rgba(152,141,157,0.2)'  },
  trial:     { color: '#deb7ff',  bg: 'rgba(222,183,255,0.12)', border: 'rgba(222,183,255,0.25)' },
  suspended: { color: '#ffb4ab',  bg: 'rgba(255,180,171,0.1)',  border: 'rgba(255,180,171,0.22)' },
  open:      { color: '#e9c46a',  bg: 'rgba(233,196,106,0.1)',  border: 'rgba(233,196,106,0.22)' },
  reviewed:  { color: '#deb7ff',  bg: 'rgba(222,183,255,0.1)',  border: 'rgba(222,183,255,0.22)' },
  dismissed: { color: '#988d9d',  bg: 'rgba(152,141,157,0.1)',  border: 'rgba(152,141,157,0.2)'  },
  easy:      { color: '#a5d6a7',  bg: 'rgba(165,214,167,0.1)',  border: 'rgba(165,214,167,0.22)' },
  medium:    { color: '#e9c46a',  bg: 'rgba(233,196,106,0.1)',  border: 'rgba(233,196,106,0.22)' },
  hard:      { color: '#ffb4ab',  bg: 'rgba(255,180,171,0.1)',  border: 'rgba(255,180,171,0.22)' },
  live:      { color: '#a5d6a7',  bg: 'rgba(165,214,167,0.1)',  border: 'rgba(165,214,167,0.22)' },
};

export default function Badge({ variant = 'active', label, dot = false, style: extraStyle }) {
  const v = VARIANTS[variant] ?? VARIANTS.active;

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: '3px 9px',
        borderRadius: 'var(--radius-badge)',
        fontSize: 11.5,
        fontWeight: 600,
        color: v.color,
        background: v.bg,
        border: `1px solid ${v.border}`,
        whiteSpace: 'nowrap',
        ...extraStyle,
      }}
    >
      {dot && (
        <span
          style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: v.color,
            flexShrink: 0,
          }}
        />
      )}
      {label}
    </span>
  );
}
