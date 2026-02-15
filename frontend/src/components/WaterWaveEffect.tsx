import { useEffect, useRef } from 'react';

const RIPPLE_DURATION_MS = 2200;
const RIPPLE_MAX_RADIUS = 48;
const RIPPLE_STROKE_WIDTH = 1;
const THROTTLE_MS = 220;
const RINGS_PER_WAVE = 2;
const RING_DELAY_MS = 180;

interface Ripple {
  x: number;
  y: number;
  createdAt: number;
}

export function WaterWaveEffect() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ripplesRef = useRef<Ripple[]>([]);
  const lastSpawnRef = useRef(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const spawnRipple = (x: number, y: number) => {
      const now = Date.now();
      if (now - lastSpawnRef.current < THROTTLE_MS) return;
      lastSpawnRef.current = now;
      ripplesRef.current.push({ x, y, createdAt: now });
    };

    const handleMove = (e: MouseEvent) => {
      spawnRipple(e.clientX, e.clientY);
    };

    const draw = () => {
      const now = Date.now();
      const ripples = ripplesRef.current;
      ripplesRef.current = ripples.filter(
        (r) => now - r.createdAt < RIPPLE_DURATION_MS + RINGS_PER_WAVE * RING_DELAY_MS
      );

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const isDark = document.documentElement.classList.contains('dark');

      ripples.forEach((r) => {
        for (let ring = 0; ring < RINGS_PER_WAVE; ring++) {
          const ringStart = r.createdAt + ring * RING_DELAY_MS;
          const age = now - ringStart;
          if (age < 0) continue;
          const t = Math.min(1, age / RIPPLE_DURATION_MS);
          const radius = t * RIPPLE_MAX_RADIUS;
          const opacity = 1 - t;
          const alpha = opacity * 0.12 * (1 - ring * 0.25);

          ctx.beginPath();
          ctx.arc(r.x, r.y, radius, 0, Math.PI * 2);
          ctx.strokeStyle = isDark
            ? `rgba(16, 185, 129, ${alpha})`
            : `rgba(30, 58, 138, ${alpha})`;
          ctx.lineWidth = RIPPLE_STROKE_WIDTH * (1 - t * 0.6);
          ctx.stroke();
        }
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    window.addEventListener('mousemove', handleMove, { passive: true });

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMove);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none z-0"
      aria-hidden
    />
  );
}
