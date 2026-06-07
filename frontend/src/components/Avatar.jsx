import { useState } from 'react';

/**
 * Renders a persona avatar with fallback.
 * If `src` is provided and loads successfully, shows the image.
 * Otherwise renders a colored circle with the first letter of `name`.
 *
 * Props:
 *   src         optional image URL (persona.avatar_url)
 *   name        used for the fallback initial + alt text
 *   size        pixel size (default 40)
 *   active      when true, uses the brand-blue accent palette
 *   className   extra classes for the container
 */
export default function Avatar({ src, name = '?', size = 40, active = false, className = '' }) {
  const [errored, setErrored] = useState(false);
  const initial = (name || '?').trim().charAt(0).toUpperCase() || '?';
  const showImage = src && !errored;

  const px = `${size}px`;
  const rounded = size >= 56 ? 'rounded-2xl' : 'rounded-xl';

  return (
    <div
      className={`${rounded} overflow-hidden flex-shrink-0 flex items-center justify-center font-bold ${className}`}
      style={{
        width: px,
        height: px,
        fontSize: Math.max(12, Math.round(size * 0.36)),
        background: active ? 'rgba(222,183,255,0.18)' : 'rgba(255,255,255,0.06)',
        color: active ? '#deb7ff' : 'var(--text-muted)',
        border: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      {showImage ? (
        <img
          src={src}
          alt={name}
          onError={() => setErrored(true)}
          className="w-full h-full object-cover"
          loading="lazy"
        />
      ) : (
        initial
      )}
    </div>
  );
}
