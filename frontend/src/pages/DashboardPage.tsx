import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { useDemo } from '../context/DemoContext';
import { useLanguage } from '../context/LanguageContext';
import { demoPatients } from '../data/dummyData';
import type { DemoPatient, RiskLevel } from '../types';

ChartJS.register(ArcElement, Tooltip, Legend);

const COLUMNS: { id: RiskLevel; labelKey: string; color: string }[] = [
  { id: 'low', labelKey: 'dashboardLow', color: 'bg-medical-teal' },
  { id: 'medium', labelKey: 'dashboardMedium', color: 'bg-medical-yellow' },
  { id: 'high', labelKey: 'dashboardHigh', color: 'bg-medical-red' },
];

const DEMO_CRITICAL = demoPatients.find((p) => p.aiOutput.risk === 'high') ?? demoPatients[0];
const DEMO_MODERATE = demoPatients.find((p) => p.aiOutput.risk === 'medium') ?? demoPatients[3];
const DEMO_MINOR = demoPatients.find((p) => p.aiOutput.risk === 'low') ?? demoPatients[1];

export function DashboardPage() {
  const { queue, addToQueue, removeFromQueue, liveMode, toggleLiveMode } = useDemo();
  const { t } = useLanguage();
  const [columns, setColumns] = useState<Record<RiskLevel, DemoPatient[]>>({
    low: [],
    medium: [],
    high: [],
  });
  const [speed, setSpeed] = useState(1);
  const [dragged, setDragged] = useState<{ id: number; risk: RiskLevel } | null>(null);
  const [demoRunning, setDemoRunning] = useState(false);
  const [demoLabel, setDemoLabel] = useState<string | null>(null);
  const demoTimeoutRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    if (demoRunning) return;
    const interval = setInterval(() => {
      const random = demoPatients[Math.floor(Math.random() * demoPatients.length)];
      setColumns((prev) => {
        const risk = random.aiOutput.risk as RiskLevel;
        const list = [...(prev[risk] || []), { ...random, id: Date.now() + Math.random() }];
        return { ...prev, [risk]: list.slice(-8) };
      });
    }, 5000 / speed);
    return () => clearInterval(interval);
  }, [speed, demoRunning]);

  const addRandom = () => {
    const p = demoPatients[Math.floor(Math.random() * demoPatients.length)];
    addToQueue(p);
    const risk = p.aiOutput.risk as RiskLevel;
    setColumns((prev) => ({
      ...prev,
      [risk]: [...(prev[risk] || []), { ...p, id: Date.now() }].slice(-10),
    }));
  };

  const runDemo = () => {
    demoTimeoutRef.current.forEach(clearTimeout);
    demoTimeoutRef.current = [];
    setDemoRunning(true);
    setColumns({ low: [], medium: [], high: [] });
    setDemoLabel(null);

    const schedule = (fn: () => void, ms: number) => {
      const id = setTimeout(fn, ms);
      demoTimeoutRef.current.push(id);
    };

    const oneCycle = (cycleIndex: number) => {
      setDemoLabel(t('demoCritical'));
      setColumns((prev) => ({
        ...prev,
        high: [{ ...DEMO_CRITICAL, id: Date.now() + 1 }],
      }));
      schedule(() => {
        setDemoLabel(t('demoModerate'));
        setColumns((prev) => ({
          ...prev,
          medium: [{ ...DEMO_MODERATE, id: Date.now() + 2 }],
        }));
      }, 1800);
      schedule(() => {
        setDemoLabel(t('demoMinor'));
        setColumns((prev) => ({
          ...prev,
          low: [{ ...DEMO_MINOR, id: Date.now() + 3 }],
        }));
      }, 3600);
      schedule(() => setDemoLabel(t('demoExplanation')), 5200);
      schedule(() => {
        setColumns({ low: [], medium: [], high: [] });
        setDemoLabel(null);
      }, 8500);
      schedule(() => oneCycle(cycleIndex + 1), 9000);
    };
    oneCycle(0);
  };

  const stopDemo = () => {
    demoTimeoutRef.current.forEach(clearTimeout);
    demoTimeoutRef.current = [];
    setDemoRunning(false);
    setDemoLabel(null);
    setColumns({ low: [], medium: [], high: [] });
  };

  const moveCard = (fromRisk: RiskLevel, fromIdx: number, toRisk: RiskLevel, toIdx: number) => {
    setColumns((prev) => {
      const fromList = [...(prev[fromRisk] || [])];
      const [item] = fromList.splice(fromIdx, 1);
      if (!item) return prev;
      const toList = [...(prev[toRisk] || [])];
      toList.splice(toIdx, 0, { ...item, aiOutput: { ...item.aiOutput, risk: toRisk } });
      return {
        ...prev,
        [fromRisk]: fromList,
        [toRisk]: toList,
      };
    });
  };

  const totalCards = Object.values(columns).reduce((s, arr) => s + arr.length, 0);
  const deptCount: Record<string, number> = {};
  Object.values(columns).flat().forEach((p) => {
    deptCount[p.aiOutput.department] = (deptCount[p.aiOutput.department] || 0) + 1;
  });
  const pieData = {
    labels: Object.keys(deptCount),
    datasets: [{ data: Object.values(deptCount), backgroundColor: ['#1E3A8A', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'] }],
  };

  return (
    <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="absolute inset-0 bg-medical-bg/30 dark:bg-gray-900/30 -z-10" />
      {demoLabel && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 rounded-xl bg-medical-blue/10 dark:bg-medical-teal/10 border border-medical-blue/20 dark:border-medical-teal/20 px-4 py-3 text-center text-sm font-medium text-medical-blue dark:text-medical-teal"
        >
          {demoLabel}
        </motion.div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">{t('dashboardTitle')}</h1>
        <div className="flex flex-wrap items-center gap-4">
          {!demoRunning ? (
            <button
              type="button"
              onClick={runDemo}
              className="px-5 py-2.5 rounded-xl bg-medical-blue text-white font-medium shadow-soft hover:shadow-soft-lg focus:ring-2 focus:ring-medical-teal focus:ring-offset-2"
            >
              {t('dashboardDemo')}
            </button>
          ) : (
            <button
              type="button"
              onClick={stopDemo}
              className="px-5 py-2.5 rounded-xl bg-medical-red text-white font-medium shadow-soft hover:bg-medical-red/90"
            >
              {t('stopDemo')}
            </button>
          )}
          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <input type="checkbox" checked={liveMode} onChange={toggleLiveMode} className="rounded accent-medical-teal" />
            <span>{t('dashboardAutoDemo')}</span>
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <span>{t('dashboardSpeed')}</span>
            <input
              type="range"
              min={1}
              max={5}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-24 accent-medical-teal"
              disabled={demoRunning}
            />
            <span>{speed}x</span>
          </label>
          <button
            type="button"
            onClick={addRandom}
            disabled={demoRunning}
            className="px-4 py-2 rounded-xl bg-medical-teal text-white font-medium shadow-soft disabled:opacity-50"
          >
            {t('dashboardAddRandom')}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {[
          { labelKey: 'dashboardTotal', value: totalCards + queue.length },
          { labelKey: 'dashboardHigh', value: (columns.high || []).length },
          { labelKey: 'dashboardMedium', value: (columns.medium || []).length },
          { labelKey: 'dashboardLow', value: (columns.low || []).length },
        ].map((k) => (
          <div key={k.labelKey} className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-white/95 dark:bg-gray-800/95 backdrop-blur shadow-soft">
            <div className="text-2xl font-bold text-medical-blue dark:text-medical-teal">{k.value}</div>
            <div className="text-sm text-gray-500">{t(k.labelKey)}</div>
          </div>
        ))}
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="w-full lg:w-64 shrink-0">
          <h2 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">{t('dashboardQueue')}</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {queue.map((p) => (
              <motion.div
                key={p.id}
                layout
                className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 bg-white dark:bg-gray-800 cursor-move"
                draggable
                onDragStart={() => setDragged({ id: p.id, risk: p.aiOutput.risk as RiskLevel })}
              >
                <p className="text-sm font-medium">#{p.id.toString().slice(-4)}</p>
                <p className="text-xs text-gray-500">{p.aiOutput.department} · {p.aiOutput.risk}</p>
                <button type="button" onClick={() => removeFromQueue(p.id)} className="text-xs text-medical-red mt-1">
                  {t('dashboardRemove')}
                </button>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-4">
          {COLUMNS.map((col) => (
            <div
              key={col.id}
              className={`rounded-xl border-2 border-dashed ${col.id === 'high' ? 'border-medical-red/50' : col.id === 'medium' ? 'border-medical-yellow/50' : 'border-medical-teal/50'} p-4 min-h-[300px]`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (dragged) {
                  const fromCol = columns[dragged.risk] || [];
                  const idx = fromCol.findIndex((c) => c.id === dragged.id);
                  if (idx >= 0) moveCard(dragged.risk, idx, col.id, (columns[col.id] || []).length);
                  setDragged(null);
                }
              }}
            >
              <h3 className={`font-semibold mb-3 ${col.id === 'high' ? 'text-medical-red' : col.id === 'medium' ? 'text-medical-yellow' : 'text-medical-teal'}`}>
                {t(col.labelKey)}
              </h3>
              <div className="space-y-2">
                <AnimatePresence>
                  {(columns[col.id] || []).map((p) => (
                    <motion.div
                      key={p.id}
                      layout
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 bg-white dark:bg-gray-800 cursor-move group"
                      draggable
                      onDragStart={() => setDragged({ id: p.id, risk: col.id })}
                    >
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">#{String(p.id).slice(-4)}</span>
                        <span className="text-xs text-gray-500">{p.age}y {p.gender}</span>
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">
                        {p.symptoms.slice(0, 2).join(', ')}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{p.aiOutput.department}</p>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          ))}
        </div>

        <div className="w-full lg:w-56 shrink-0">
          <h3 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">{t('dashboardByDept')}</h3>
          <div className="h-48">
            <Doughnut data={pieData} options={{ responsive: true }} />
          </div>
        </div>
      </div>
    </div>
  );
}
