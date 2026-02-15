/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'sans-serif'],
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
      },
      colors: {
        medical: {
          blue: '#1E3A8A',
          teal: '#0D9488',
          yellow: '#B45309',
          red: '#E57373', // Light and warm red (Material Red 300) - matches background warmth
          blueMuted: '#334155',
          bg: '#F8FAFC',
          surface: '#FFFFFF',
        },
      },
      boxShadow: {
        'soft': '0 2px 8px rgba(30, 58, 138, 0.06)',
        'soft-lg': '0 4px 20px rgba(30, 58, 138, 0.08)',
        'glass': '0 8px 32px rgba(30, 58, 138, 0.1)',
        'elevation': '0 1px 3px rgba(0,0,0,0.04), 0 6px 16px rgba(30, 58, 138, 0.06)',
        'card': '0 4px 6px -1px rgba(0,0,0,0.06), 0 10px 28px -4px rgba(30, 58, 138, 0.12)',
        'card-hover': '0 12px 24px -4px rgba(0,0,0,0.08), 0 20px 40px -8px rgba(30, 58, 138, 0.15)',
        'venn': '0 0 0 1px rgba(255,255,255,0.5), 0 8px 32px rgba(30, 58, 138, 0.2), 0 24px 48px -12px rgba(0,0,0,0.15)',
        'venn-inner': 'inset 0 2px 12px rgba(255,255,255,0.4)',
      },
      dropShadow: {
        'venn': '0 8px 32px rgba(30, 58, 138, 0.18), 0 24px 48px -12px rgba(0,0,0,0.12)',
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'marquee': 'marquee 20s linear infinite',
        'blob': 'blob 8s ease-in-out infinite',
      },
      keyframes: {
        marquee: { '0%': { transform: 'translateX(0)' }, '100%': { transform: 'translateX(-50%)' } },
        blob: {
          '0%, 100%': { transform: 'translate(0, 0) scale(1)' },
          '33%': { transform: 'translate(20px, -30px) scale(1.05)' },
          '66%': { transform: 'translate(-20px, 20px) scale(0.95)' },
        },
      },
    },
  },
  plugins: [],
}
