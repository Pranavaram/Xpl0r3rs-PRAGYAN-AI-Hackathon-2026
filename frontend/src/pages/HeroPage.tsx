import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useLanguage } from '../context/LanguageContext';

export function HeroPage() {
  const { t } = useLanguage();

  return (
    <div className="w-full font-sans text-gray-900 dark:text-gray-100 pb-20">
      {/* 
        HERO SECTION: Premium look — background image shows through from Layout.
        Vibrant color combo: deep blue, teal, coral accent.
      */}
      <section className="relative w-full min-h-[550px] lg:min-h-[650px] flex items-center text-gray-900 dark:text-gray-100 overflow-hidden">
        {/* Subtle gradient + soft blobs so layout BG image stays visible */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-br from-sky-50/50 via-white/30 to-teal-50/40 dark:from-slate-900/40 dark:via-transparent dark:to-medical-blue/20" />
          <div className="absolute -top-24 -right-24 w-[28rem] h-[28rem] rounded-full bg-blue-200/50 dark:bg-medical-blue/30 mix-blend-multiply dark:mix-blend-screen blur-3xl animate-blob" />
          <div className="absolute bottom-0 left-1/4 w-72 h-72 rounded-full bg-teal-200/50 dark:bg-teal-500/20 mix-blend-multiply dark:mix-blend-screen blur-2xl animate-blob [animation-delay:2s]" />
          <div className="absolute top-1/2 right-1/3 w-64 h-64 rounded-full bg-amber-200/40 dark:bg-amber-500/15 mix-blend-multiply dark:mix-blend-screen blur-2xl animate-blob [animation-delay:4s]" />
        </div>

        <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
            className="text-center lg:text-left"
          >
            <h1 className="font-serif text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight mb-6 text-gray-900 dark:text-white drop-shadow-sm">
              Smart Triage <br />
              <span className="italic bg-gradient-to-r from-medical-blue via-teal-600 to-emerald-600 dark:from-sky-300 dark:via-teal-400 dark:to-emerald-400 bg-clip-text text-transparent">Simplified.</span>
            </h1>
            <p className="text-lg sm:text-xl lg:text-2xl font-light mb-10 max-w-xl mx-auto lg:mx-0 leading-relaxed text-gray-700 dark:text-gray-200">
              AI-powered precision for modern healthcare. Experience faster prioritization and seamless patient flow.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link
                to="/triage"
                className="inline-flex items-center justify-center px-8 py-4 text-base font-bold text-white bg-gradient-to-r from-medical-blue to-blue-700 hover:from-blue-700 hover:to-medical-blue rounded-full transition-all hover:scale-105 shadow-lg shadow-blue-900/25"
              >
                {t('quickTriage')}
              </Link>
              <Link
                to="/dashboard"
                className="inline-flex items-center justify-center px-8 py-4 text-base font-bold text-medical-blue dark:text-teal-300 border-2 border-medical-blue/50 dark:border-teal-400/50 rounded-full hover:bg-medical-blue/10 dark:hover:bg-teal-500/10 transition-colors"
              >
                {t('liveDashboard')}
              </Link>
            </div>
          </motion.div>

          {/* Hero Image / Graphic */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="hidden lg:block relative"
          >
            <div className="relative aspect-square max-w-md mx-auto">
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-medical-blue/20 to-teal-500/20 dark:from-medical-blue/30 dark:to-teal-500/30 backdrop-blur-sm border-2 border-white/40 dark:border-white/20 shadow-2xl" />
              <div className="absolute inset-4 overflow-hidden rounded-full border-4 border-white/40 dark:border-white/20 shadow-2xl ring-2 ring-medical-blue/20">
                <img
                  src="https://images.unsplash.com/photo-1631815589968-fdb09a223b1e?auto=format&fit=crop&q=80&w=800"
                  alt="Medical Team"
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-700"
                />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Bottom curve separator — vibrant tint */}
        <div className="absolute bottom-0 left-0 w-full overflow-hidden leading-none z-20">
          <svg className="relative block w-[120%] h-12 lg:h-20 -left-[10%] text-slate-50 dark:text-slate-900" viewBox="0 0 1200 120" preserveAspectRatio="none">
            <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" fill="currentColor" opacity=".3"></path>
            <path d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.15c59.54-19.85,113.91-58.74,179-81.82C485.6,3.06,583.56,26.54,659.66,61.94c84.66,39.38,181.79,48.25,270.47,15.19C1021.72,48.17,1101.43,18.06,1200,60V0Z" fill="currentColor" opacity=".5"></path>
            <path d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z" fill="currentColor"></path>
          </svg>
        </div>
      </section>

      {/* FEATURE GRID: vibrant accents, premium cards */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-24">
        <div className="text-center mb-16">
          <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Discover Smart Triage<span className="text-red-500 dark:text-red-400">.</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Explore the features that make our platform the leader in AI-assisted patient prioritization.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Feature 1 — red accent */}
          <Link to="/triage" className="group block h-full">
            <div className="relative h-64 overflow-hidden rounded-t-2xl ring-2 ring-transparent group-hover:ring-red-400/50 transition-all duration-300">
              <img
                src="https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?auto=format&fit=crop&q=80&w=800"
                alt="Diagnosis"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent group-hover:from-black/50 transition-opacity" />
              <div className="absolute bottom-4 left-4">
                <span className="px-3 py-1.5 bg-red-500 text-white text-xs font-bold tracking-wider uppercase rounded shadow-lg">AI Powered</span>
              </div>
            </div>
            <div className="p-6 bg-white/95 dark:bg-gray-800/95 backdrop-blur border-x border-b border-gray-200 dark:border-gray-600 rounded-b-2xl shadow-soft-lg group-hover:shadow-card-hover transition-shadow border-t-2 border-t-red-500/20">
              <h3 className="font-serif text-2xl font-bold mb-3 text-gray-900 dark:text-white group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">Start Triage</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">
                Begin a new patient assessment using our advanced AI engine. Get instant risk stratification and department recommendations.
              </p>
              <div className="text-sm font-bold uppercase tracking-wider text-red-600 dark:text-red-400 group-hover:underline">Start Now &rsaquo;</div>
            </div>
          </Link>

          {/* Feature 2 — blue accent */}
          <Link to="/dashboard" className="group block h-full">
            <div className="relative h-64 overflow-hidden rounded-t-2xl ring-2 ring-transparent group-hover:ring-medical-blue/50 transition-all duration-300">
              <img
                src="https://images.unsplash.com/photo-1551076805-e1869033e561?auto=format&fit=crop&q=80&w=800"
                alt="Dashboard"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent group-hover:from-black/50 transition-opacity" />
              <div className="absolute bottom-4 left-4">
                <span className="px-3 py-1.5 bg-medical-blue text-white text-xs font-bold tracking-wider uppercase rounded shadow-lg">Real-time</span>
              </div>
            </div>
            <div className="p-6 bg-white/95 dark:bg-gray-800/95 backdrop-blur border-x border-b border-gray-200 dark:border-gray-600 rounded-b-2xl shadow-soft-lg group-hover:shadow-card-hover transition-shadow border-t-2 border-t-medical-blue/30">
              <h3 className="font-serif text-2xl font-bold mb-3 text-gray-900 dark:text-white group-hover:text-medical-blue transition-colors">Live Dashboard</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">
                Monitor the waiting room in real-time. Track patient flow, department occupancy, and critical alerts instantly.
              </p>
              <div className="text-sm font-bold uppercase tracking-wider text-medical-blue group-hover:underline">View Dashboard &rsaquo;</div>
            </div>
          </Link>

          {/* Feature 3 — teal accent */}
          <Link to="/about" className="group block h-full">
            <div className="relative h-64 overflow-hidden rounded-t-2xl ring-2 ring-transparent group-hover:ring-teal-400/50 transition-all duration-300">
              <img
                src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=800"
                alt="Technology"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent group-hover:from-black/50 transition-opacity" />
              <div className="absolute bottom-4 left-4">
                <span className="px-3 py-1.5 bg-teal-500 text-white text-xs font-bold tracking-wider uppercase rounded shadow-lg">Technology</span>
              </div>
            </div>
            <div className="p-6 bg-white/95 dark:bg-gray-800/95 backdrop-blur border-x border-b border-gray-200 dark:border-gray-600 rounded-b-2xl shadow-soft-lg group-hover:shadow-card-hover transition-shadow border-t-2 border-t-teal-500/30">
              <h3 className="font-serif text-2xl font-bold mb-3 text-gray-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">How it Works</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">
                Discover the architecture behind Smart Triage. Learn about our XGBoost models, fairness audits, and tech stack.
              </p>
              <div className="text-sm font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400 group-hover:underline">Learn More &rsaquo;</div>
            </div>
          </Link>
        </div>
      </section>

      {/* Stats: vibrant metrics strip */}
      <section className="relative py-16 lg:py-20 border-y border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-slate-50 via-blue-50/50 to-teal-50/50 dark:from-gray-800 dark:via-medical-blue/10 dark:to-gray-800" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <h2 className="font-serif text-2xl lg:text-3xl font-bold text-gray-800 dark:text-gray-200 mb-12">Trusted Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="rounded-2xl py-6 px-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur border border-blue-100 dark:border-medical-blue/30 shadow-soft">
              <div className="text-4xl lg:text-5xl font-bold bg-gradient-to-br from-medical-blue to-blue-600 bg-clip-text text-transparent mb-2">95%</div>
              <div className="text-sm font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">Accuracy</div>
            </div>
            <div className="rounded-2xl py-6 px-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur border border-teal-200 dark:border-teal-500/30 shadow-soft">
              <div className="text-4xl lg:text-5xl font-bold text-teal-600 dark:text-teal-400 mb-2">&lt;2%</div>
              <div className="text-sm font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">Bias Ratio</div>
            </div>
            <div className="rounded-2xl py-6 px-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur border border-amber-200 dark:border-amber-500/30 shadow-soft">
              <div className="text-4xl lg:text-5xl font-bold text-amber-600 dark:text-amber-400 mb-2">&lt;500ms</div>
              <div className="text-sm font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">Inference</div>
            </div>
            <div className="rounded-2xl py-6 px-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur border border-red-200 dark:border-red-500/30 shadow-soft">
              <div className="text-4xl lg:text-5xl font-bold text-red-600 dark:text-red-400 mb-2">10k+</div>
              <div className="text-sm font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">Simulations</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
