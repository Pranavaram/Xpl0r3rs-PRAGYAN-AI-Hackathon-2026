import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { DemoPatient, TriageFormData } from '../types';
import { demoPatients, QUICK_PRESETS } from '../data/dummyData';

interface DemoState {
  currentPatient: DemoPatient | null;
  currentPatientId: number;
  liveMode: boolean;
  demoCycleSeconds: number;
  queue: DemoPatient[];
  lastTriageResult: DemoPatient | null;
}

interface DemoContextValue extends DemoState {
  loadPatient: (id: number) => void;
  nextDemo: () => void;
  toggleLiveMode: () => void;
  setDemoCycleSeconds: (s: number) => void;
  setLastTriageResult: (p: DemoPatient | null) => void;
  addToQueue: (p: DemoPatient) => void;
  removeFromQueue: (id: number) => void;
  reorderQueue: (from: number, to: number) => void;
  getPresets: () => typeof QUICK_PRESETS;
  formDataFromPatient: (p: DemoPatient) => TriageFormData;
}

const DemoContext = createContext<DemoContextValue | null>(null);

export function DemoProvider({ children }: { children: React.ReactNode }) {
  const [currentPatientId, setCurrentPatientId] = useState(1);
  const [liveMode, setLiveModeState] = useState(true);
  const [demoCycleSeconds, setDemoCycleSeconds] = useState(30);
  const [queue, setQueue] = useState<DemoPatient[]>([]);
  const [lastTriageResult, setLastTriageResult] = useState<DemoPatient | null>(null);

  const currentPatient = demoPatients.find((p) => p.id === currentPatientId) ?? demoPatients[0];

  const loadPatient = useCallback((id: number) => {
    const p = demoPatients.find((x) => x.id === id);
    if (p) setCurrentPatientId(p.id);
  }, []);

  const nextDemo = useCallback(() => {
    const idx = demoPatients.findIndex((p) => p.id === currentPatientId);
    const nextIdx = (idx + 1) % demoPatients.length;
    setCurrentPatientId(demoPatients[nextIdx].id);
  }, [currentPatientId]);

  const toggleLiveMode = useCallback(() => {
    setLiveModeState((v) => !v);
  }, []);

  const addToQueue = useCallback((p: DemoPatient) => {
    setQueue((q) => [...q, { ...p, id: Date.now() }]);
  }, []);

  const removeFromQueue = useCallback((id: number) => {
    setQueue((q) => q.filter((x) => x.id !== id));
  }, []);

  const reorderQueue = useCallback((from: number, to: number) => {
    setQueue((q) => {
      const copy = [...q];
      const [removed] = copy.splice(from, 1);
      copy.splice(to, 0, removed);
      return copy;
    });
  }, []);

  const getPresets = useCallback(() => QUICK_PRESETS, []);

  const formDataFromPatient = useCallback((p: DemoPatient): TriageFormData => ({
    age: p.age,
    gender: p.gender,
    symptoms: [...p.symptoms],
    vitals: { ...p.vitals },
    conditions: [...p.conditions],
    allergies: p.allergies ?? '',
    medications: p.medications ?? '',
    onsetDurationValue: 1,
    onsetDurationUnit: 'days',
  }), []);

  useEffect(() => {
    if (!liveMode) return;
    const interval = setInterval(nextDemo, demoCycleSeconds * 1000);
    return () => clearInterval(interval);
  }, [liveMode, demoCycleSeconds, nextDemo]);

  const value: DemoContextValue = {
    currentPatient,
    currentPatientId,
    liveMode,
    demoCycleSeconds,
    queue,
    lastTriageResult,
    loadPatient,
    nextDemo,
    toggleLiveMode,
    setDemoCycleSeconds,
    setLastTriageResult,
    addToQueue,
    removeFromQueue,
    reorderQueue,
    getPresets,
    formDataFromPatient,
  };

  return <DemoContext.Provider value={value}>{children}</DemoContext.Provider>;
}

export function useDemo() {
  const ctx = useContext(DemoContext);
  if (!ctx) throw new Error('useDemo must be used within DemoProvider');
  return ctx;
}
