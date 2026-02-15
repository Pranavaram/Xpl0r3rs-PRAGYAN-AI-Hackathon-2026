import { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
import { useDemo } from '../context/DemoContext';
import { useLanguage } from '../context/LanguageContext';
import { RelatedDiseasesGraph } from '../components/RelatedDiseasesGraph';
import type { RiskLevel } from '../types';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

export function ResultsPage() {
  const { lastTriageResult, addToQueue } = useDemo();
  const { t } = useLanguage();
  const navigate = useNavigate();
  const reportRef = useRef<HTMLDivElement>(null);

  const result = lastTriageResult;

  if (!result) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <p className="text-gray-500 mb-4">{t('resultsNoResult')}</p>
        <button
          type="button"
          onClick={() => navigate('/triage')}
          className="px-4 py-2 rounded-lg bg-medical-teal text-white"
        >
          {t('resultsGoToTriage')}
        </button>
      </div>
    );
  }

  const risk = result.aiOutput.risk as RiskLevel;
  const riskBg = risk === 'high' ? 'bg-medical-red' : risk === 'medium' ? 'bg-medical-yellow' : 'bg-medical-teal';
  const riskLabelKey = risk === 'high' ? 'dashboardHigh' : risk === 'medium' ? 'dashboardMedium' : 'dashboardLow';

  const donutData = {
    labels: [t('resultsConfidence'), t('resultsUncertainty')],
    datasets: [{ data: [result.aiOutput.confidence * 100, (1 - result.aiOutput.confidence) * 100], backgroundColor: ['#10B981', '#E5E7EB'] }],
  };

  const factors = result.aiOutput.factors.slice(0, 10); // Show top 10 factors

  const exportPdf = async () => {
    if (!reportRef.current) return;
    const canvas = await html2canvas(reportRef.current, { scale: 2 });
    const img = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    pdf.addImage(img, 'PNG', 10, 10, 190, 0);
    pdf.save('triage-report.pdf');
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <motion.div
        ref={reportRef}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="space-y-8"
      >
        <div className="text-center">
          <motion.div
            initial={{ scale: 0.5 }}
            animate={{ scale: 1.2 }}
            transition={{ type: 'spring', stiffness: 200, damping: 10 }}
            className={`inline-block rounded-2xl px-8 py-4 ${riskBg} text-white text-2xl font-bold uppercase shadow-lg`}
          >
            {t(riskLabelKey)} {t('resultRisk')}
          </motion.div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
            <h3 className="font-semibold mb-2">{t('resultsConfidence')}</h3>
            <div className="w-40 h-40 mx-auto">
              <Doughnut data={donutData} options={{ responsive: true }} />
            </div>
            <p className="text-center font-bold">{(result.aiOutput.confidence * 100).toFixed(0)}%</p>
          </div>
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
            <h3 className="font-semibold mb-2">{t('resultsDepartment')}</h3>
            <p className="text-lg">
              {result.aiOutput.department}
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-6 bg-white dark:bg-gray-800 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <svg className="w-24 h-24" fill="currentColor" viewBox="0 0 24 24"><path d="M9.5 3A6.5 6.5 0 0 1 16 9.5c0 1.61-.59 3.09-1.56 4.23l.27.27h.79l5 5-1.5 1.5-5-5v-.79l-.27-.27A6.516 6.516 0 0 1 9.5 16 6.5 6.5 0 0 1 3 9.5 6.5 6.5 0 0 1 9.5 3m0 2C7 5 5 7 5 9.5S7 14 9.5 14 14 12 14 9.5 12 5 9.5 5z" /></svg>
          </div>
          <h3 className="font-semibold mb-4 text-lg">{t('resultsExplainableFactors')}</h3>

          {/* AI Explanation Text */}
          <div className="mb-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 text-gray-800 dark:text-gray-200">
            <p className="font-medium mb-1 text-medical-blue dark:text-blue-300">Analysis Summary:</p>
            <p className="italic">{result.aiOutput.reasoning}</p>
          </div>

          <div className="h-80">
            <Bar
              data={{
                labels: factors.map((f) => f.name),
                datasets: [{
                  label: 'Impact on Risk',
                  data: factors.map((f) => f.weight),
                  backgroundColor: factors.map((f) => f.weight >= 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)'), // Red for positive (risk increase), Green for negative (risk decrease)
                  borderColor: factors.map((f) => f.weight >= 0 ? 'rgb(239, 68, 68)' : 'rgb(16, 185, 129)'),
                  borderWidth: 1
                }]
              }}
              options={{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { display: false },
                  tooltip: {
                    callbacks: {
                      label: (ctx) => `Impact: ${ctx.formattedValue}`
                    }
                  }
                },
                scales: {
                  x: {
                    grid: { color: 'rgba(0,0,0,0.05)' },
                    title: { display: true, text: 'Contribution to Risk Score (SHAP value)' }
                  },
                  y: {
                    grid: { display: false }
                  }
                }
              }}
            />
          </div>
          <p className="text-xs text-center text-gray-500 mt-3">Positive values (Right/Red) increase risk. Negative values (Left/Green) decrease risk.</p>
        </div>



        <RelatedDiseasesGraph />

        <div className="flex flex-wrap gap-4">
          <button
            type="button"
            onClick={exportPdf}
            className="px-6 py-3 rounded-xl bg-medical-blue text-white font-semibold hover:bg-medical-blue/90"
          >
            {t('resultsPdfReport')}
          </button>
          <button
            type="button"
            onClick={() => {
              addToQueue(result);
              navigate('/dashboard');
            }}
            className="px-6 py-3 rounded-xl border border-medical-teal text-medical-teal font-semibold hover:bg-medical-teal/10"
          >
            {t('resultsAddToQueue')}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
