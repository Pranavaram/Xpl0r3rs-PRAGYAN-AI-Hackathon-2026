import { motion } from 'framer-motion';
import { useLanguage } from '../context/LanguageContext';

const FEATURES = [
  { titleKey: 'aboutFeatureVoiceInput', descKey: 'aboutFeatureVoiceInputDesc', icon: '🎤' },
  { titleKey: 'aboutFeatureMultiLang', descKey: 'aboutFeatureMultiLangDesc', icon: '🗣️' },
  { titleKey: 'aboutFeaturePWA', descKey: 'aboutFeaturePWADesc', icon: '📱' },
  { titleKey: 'aboutFeatureBiasAudit', descKey: 'aboutFeatureBiasAuditDesc', icon: '⚖️' },
  { titleKey: 'aboutFeatureExplainableAI', descKey: 'aboutFeatureExplainableAIDesc', icon: '🎯' },
  { titleKey: 'aboutFeatureSmartEHR', descKey: 'aboutFeatureSmartEHRDesc', icon: '📄' },
  { titleKey: 'aboutFeatureLiveDashboard', descKey: 'aboutFeatureLiveDashboardDesc', icon: '📊' },
  { titleKey: 'aboutFeaturePDFReports', descKey: 'aboutFeaturePDFReportsDesc', icon: '📑' },
];

const METRICS = [
  { metricKey: 'aboutMetricAccuracy', value: '95.2%', notesKey: 'aboutMetricAccuracyNotes' },
  { metricKey: 'aboutMetricF1', value: '0.92', notesKey: 'aboutMetricF1Notes' },
  { metricKey: 'aboutMetricInference', value: '<500ms', notesKey: 'aboutMetricInferenceNotes' },
  { metricKey: 'aboutMetricBias', value: '<2%', notesKey: 'aboutMetricBiasNotes' },
];

export function AboutPage() {
  const { t } = useLanguage();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <div className="w-full min-h-screen font-sans text-gray-900 dark:text-gray-100 pb-20 relative overflow-hidden selection:bg-medical-blue/30">

      {/* ARTISTIC BACKGROUND MESH */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-purple-200/40 dark:bg-purple-900/20 rounded-full blur-[120px] mix-blend-multiply dark:mix-blend-screen animate-blob" />
        <div className="absolute top-[0%] right-[-10%] w-[50%] h-[50%] bg-blue-200/40 dark:bg-blue-900/20 rounded-full blur-[120px] mix-blend-multiply dark:mix-blend-screen animate-blob animation-delay-2000" />
        <div className="absolute bottom-[-10%] left-[20%] w-[50%] h-[50%] bg-teal-200/40 dark:bg-teal-900/20 rounded-full blur-[120px] mix-blend-multiply dark:mix-blend-screen animate-blob animation-delay-4000" />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 dark:brightness-50" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-32 pt-24">

        {/* HEADER SECTION */}
        <section className="text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="inline-block mb-4 px-4 py-1.5 rounded-full border border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm text-sm font-medium text-gray-500 dark:text-gray-400"
          >
            ✨ &nbsp; {t('aboutBadge')}
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="font-sans text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white mb-8 tracking-tight"
          >
            {t('aboutTitle')}
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto font-light leading-relaxed"
          >
            {t('aboutSubtitle')}
          </motion.p>
        </section>

        {/* ARCHITECTURE DIAGRAM (Artistic Glass) */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="relative"
        >
          <div className="absolute inset-0 bg-gradient-to-b from-white/40 to-white/0 dark:from-gray-800/40 dark:to-gray-800/0 rounded-[3rem] -z-10 blur-xl" />

          <div className="bg-white/60 dark:bg-gray-900/60 backdrop-blur-xl rounded-[2.5rem] p-8 lg:p-16 border border-white/50 dark:border-gray-700 shadow-2xl relative overflow-hidden group">
            {/* Subtle border gradient on hover */}
            <div className="absolute inset-0 rounded-[2.5rem] md:rounded-[2.5rem] border-2 border-transparent bg-gradient-to-br from-blue-500/10 via-transparent to-teal-500/10 z-0 pointer-events-none" />

            <div className="relative z-10 flex flex-col items-center gap-12">
              <div className="text-center mb-4">
                <h2 className="font-sans text-3xl font-bold text-gray-900 dark:text-gray-100">{t('aboutArch')}</h2>
                <div className="w-12 h-1 bg-medical-blue mt-4 mx-auto rounded-full"></div>
              </div>

              {/* Inputs */}
              <div className="flex flex-wrap justify-center gap-6">
                {[
                  { key: 'aboutArchPatientInput' },
                  { key: 'aboutArchVoiceCommand' },
                  { key: 'aboutArchEHRUpload' },
                ].map(({ key }) => (
                  <motion.div
                    whileHover={{ scale: 1.05, borderColor: '#3B82F6' }}
                    key={key}
                    className="px-8 py-4 rounded-2xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-600 font-medium text-gray-600 dark:text-gray-300 shadow-lg shadow-gray-200/50 dark:shadow-none cursor-default relative overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-50 to-transparent dark:from-blue-900/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                    <span className="relative z-10">{t(key)}</span>
                  </motion.div>
                ))}
              </div>

              {/* Flow Line */}
              <div className="h-16 w-px bg-gradient-to-b from-gray-300 via-blue-400 to-gray-300 dark:from-gray-700 dark:via-blue-500 dark:to-gray-700 relative">
                <motion.div
                  animate={{ y: [0, 64, 0] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                />
              </div>

              {/* Core System */}
              <div className="w-full max-w-4xl p-1.5 rounded-3xl bg-gradient-to-br from-blue-100 via-indigo-100 to-purple-100 dark:from-blue-900/40 dark:via-indigo-900/40 dark:to-purple-900/40 relative">
                <div className="absolute inset-0 blur-xl bg-blue-200/30 dark:bg-blue-900/30 -z-10" />
                <div className="bg-white/80 dark:bg-gray-900/90 backdrop-blur-md rounded-[1.3rem] p-10 text-center border border-white/60 dark:border-gray-600 shadow-inner">
                  <h3 className="text-sm font-bold text-blue-600 dark:text-blue-400 mb-8 uppercase tracking-[0.2em]">{t('aboutTriageEngine')}</h3>
                  <div className="flex flex-col md:flex-row justify-center items-center gap-8">
                    <div className="p-6 rounded-2xl bg-gray-50 dark:bg-gray-800 w-full md:w-64 border border-gray-100 dark:border-gray-700 shadow-sm relative group/node">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-200 to-teal-200 opacity-0 group-hover/node:opacity-50 blur transition-opacity rounded-2xl" />
                      <div className="relative font-bold text-gray-800 dark:text-gray-200">{t('aboutNodeSymptoms')}</div>
                    </div>

                    <div className="text-2xl text-gray-300 dark:text-gray-600">→</div>

                    <div className="p-8 rounded-2xl bg-gradient-to-br from-white to-blue-50 dark:from-gray-800 dark:to-gray-800 border border-blue-100 dark:border-blue-900/50 w-full md:w-auto shadow-[0_10px_40px_-10px_rgba(59,130,246,0.2)] transform hover:scale-105 transition-all duration-300 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-20 h-20 bg-blue-400/10 rounded-full blur-2xl pointer-events-none" />
                      <div className="font-bold text-2xl text-gray-900 dark:text-white mb-2">{t('aboutNodeAIModel')}</div>
                      <div className="text-sm font-medium px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 inline-block">{t('aboutXGBoost')}</div>
                    </div>

                    <div className="text-2xl text-gray-300 dark:text-gray-600">→</div>

                    <div className="p-6 rounded-2xl bg-gray-50 dark:bg-gray-800 w-full md:w-64 border border-gray-100 dark:border-gray-700 shadow-sm relative group/node">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-teal-200 to-emerald-200 opacity-0 group-hover/node:opacity-50 blur transition-opacity rounded-2xl" />
                      <div className="relative font-bold text-gray-800 dark:text-gray-200">{t('aboutNodeRiskScore')}</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Flow Line */}
              <div className="h-16 w-px bg-gradient-to-b from-gray-300 via-emerald-400 to-gray-300 dark:from-gray-700 dark:via-emerald-500 dark:to-gray-700 relative">
                <motion.div
                  animate={{ y: [0, 64, 0] }}
                  transition={{ duration: 2, delay: 1, repeat: Infinity, ease: "linear" }}
                  className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.8)]"
                />
              </div>

              {/* Outputs */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-3xl">
                {[
                  { icon: '🏥', labelKey: 'aboutOutDepartment', color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-100' },
                  { icon: '⚠️', labelKey: 'aboutOutPriority', color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-100' },
                  { icon: '💡', labelKey: 'aboutOutExplanation', color: 'text-purple-600', bg: 'bg-purple-50', border: 'border-purple-100' }
                ].map((out) => (
                  <div key={out.labelKey} className={`p-6 rounded-2xl ${out.bg} dark:bg-gray-800 border ${out.border} dark:border-gray-700 text-center shadow-sm hover:translate-y-[-2px] transition-transform`}>
                    <div className="text-3xl mb-3 filter drop-shadow-sm">{out.icon}</div>
                    <div className={`font-bold ${out.color} dark:text-gray-200`}>{t(out.labelKey)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.section>

        {/* FEATURES GRID (Dynamic Layout) */}
        <motion.section
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          <div className="text-center mb-16">
            <h2 className="font-sans text-4xl font-bold text-gray-900 dark:text-white mb-4">{t('aboutFeatures')}</h2>
            <div className="w-24 h-1.5 bg-gradient-to-r from-blue-400 to-teal-400 mx-auto rounded-full"></div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
            {FEATURES.map((f, i) => (
              <motion.div
                key={f.titleKey}
                variants={itemVariants}
                className={`group relative p-8 rounded-[2rem] bg-white/70 dark:bg-gray-800/70 backdrop-blur-md border border-white/50 dark:border-gray-700 shadow-xl hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 overflow-hidden ${i === 0 || i === 3 ? 'md:col-span-2 lg:col-span-1' : ''}`}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-teal-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="absolute -right-16 -top-16 w-48 h-48 bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="w-16 h-16 mb-8 rounded-2xl bg-gradient-to-br from-white to-gray-50 dark:from-gray-700 dark:to-gray-800 shadow-md border border-gray-100 dark:border-gray-600 flex items-center justify-center text-3xl group-hover:scale-110 group-hover:rotate-3 transition-transform duration-300">
                    {f.icon}
                  </div>
                  <h3 className="font-sans text-xl font-bold text-gray-900 dark:text-white mb-3 tracking-tight">{t(f.titleKey)}</h3>
                  <p className="text-base text-gray-500 dark:text-gray-400 leading-relaxed font-light flex-grow">{t(f.descKey)}</p>

                  <div className="mt-6 flex items-center text-sm font-medium text-blue-600 dark:text-blue-400 opacity-0 group-hover:opacity-100 transition-all duration-300 translate-y-2 group-hover:translate-y-0">
                    {t('aboutLearnMore')} <span className="ml-2">→</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* METRICS GRID (Stats) */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="pb-20"
        >
          <div className="bg-medical-blue dark:bg-gray-800 rounded-[3rem] p-12 lg:p-20 text-white relative overflow-hidden shadow-2xl">
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-overlay"></div>
            {/* Decorative circles */}
            <div className="absolute top-0 left-0 w-64 h-64 bg-white/10 rounded-full -translate-x-1/2 -translate-y-1/2 blur-2xl"></div>
            <div className="absolute bottom-0 right-0 w-96 h-96 bg-teal-500/20 rounded-full translate-x-1/3 translate-y-1/3 blur-3xl"></div>

            <div className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-12 text-center">
              {METRICS.map((m) => (
                <div key={m.metricKey} className="space-y-2">
                  <div className="text-4xl lg:text-5xl font-bold tracking-tighter mb-2">{m.value}</div>
                  <div className="text-sm font-bold uppercase tracking-widest text-blue-200">{t(m.metricKey)}</div>
                  <div className="text-xs text-blue-100/60 font-medium">{t(m.notesKey)}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.section>

      </div>
    </div>
  );
}
