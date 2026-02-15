import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion, AnimatePresence } from 'framer-motion';
import Fuse from 'fuse.js';
import toast from 'react-hot-toast';
import { useDemo } from '../context/DemoContext';
import { useLanguage } from '../context/LanguageContext';
import { getSymptomKey } from '../i18n/symptomKeys';
import { startTriageSessionApi, answerTriageQuestionApi, type SessionStartRequest, type QuestionResponse, fahrenheitToCelsius, ensureNumber, parseSystolicBp } from '../api/client';
import { SYMPTOM_OPTIONS, demoPatients } from '../data/dummyData';
import type { TriageFormData, DemoPatient, RiskLevel } from '../types';
import { extractTextFromPdf } from '../utils/extractPdfText';
import { parseEhrText, mergeParsedIntoForm } from '../utils/parseEhrText';

const inputFocus =
  'focus:ring-2 focus:ring-medical-teal/50 focus:border-medical-teal dark:focus:ring-medical-teal/40 dark:focus:border-medical-teal transition-shadow';

const schema = z.object({
  age: z.number().min(0).max(110),
  gender: z.enum(['M', 'F', 'O']),
  symptoms: z.array(z.string()),
  vitals: z.object({
    bp: z.string().optional(),
    hr: z.number().optional(),
    temp: z.number().optional(),
    respiratoryRate: z.number().optional(),
    spo2: z.number().optional(),
  }).optional(),
  conditions: z.array(z.string()).optional(),
  onsetDurationValue: z.number().optional(),
  onsetDurationUnit: z.string().optional(),
  allergies: z.string().optional(),
  medications: z.string().optional(),
});

const STORAGE_KEY = 'triage-form-draft';

export function TriagePage() {
  const [step, setStep] = useState(0); // 0: Initial, 1: Questioning, 2: Result/Demo fallback
  const [analyzing, setAnalyzing] = useState(false);
  const [symptomSearch, setSymptomSearch] = useState('');
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string; ocrText?: string } | null>(null);
  const [ocrProcessing, setOcrProcessing] = useState(false);

  // Active Questioning State
  const [session, setSession] = useState<QuestionResponse | null>(null);
  const [currentAnswer, setCurrentAnswer] = useState<string | number>('');
  const [answerError, setAnswerError] = useState('');

  const navigate = useNavigate();
  const { currentPatient, formDataFromPatient, setLastTriageResult } = useDemo();
  const { t } = useLanguage();

  const defaultValues: TriageFormData = currentPatient
    ? formDataFromPatient(currentPatient)
    : {
      age: 35,
      gender: 'M',
      symptoms: [],
      vitals: { bp: '120/80', hr: 72, temp: 98.6, respiratoryRate: 18 },
      onsetDurationValue: 1,
      onsetDurationUnit: 'days',
      conditions: [],
      allergies: '',
      medications: '',
    };

  const { register, setValue, watch } = useForm<TriageFormData>({
    defaultValues,
    resolver: zodResolver(schema),
  });

  const formValues = watch();

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        // Only restore step 0 fields
        if (parsed.age !== undefined) setValue('age', parsed.age);
        if (parsed.gender) setValue('gender', parsed.gender);
        if (parsed.symptoms?.length) setValue('symptoms', parsed.symptoms);
        // We restore others too in case user goes back, but they are optional now
        if (parsed.vitals) setValue('vitals', parsed.vitals);
        if (parsed.onsetDurationValue) setValue('onsetDurationValue', parsed.onsetDurationValue);
        if (parsed.onsetDurationUnit) setValue('onsetDurationUnit', parsed.onsetDurationUnit);
        if (parsed.conditions?.length) setValue('conditions', parsed.conditions);
        if (parsed.allergies !== undefined) setValue('allergies', parsed.allergies);
        if (parsed.medications !== undefined) setValue('medications', parsed.medications);
      }
    } catch (_) { }
  }, [setValue]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(formValues));
  }, [formValues]);

  const fuse = new Fuse(SYMPTOM_OPTIONS, { threshold: 0.3 });
  const symptomSuggestions = symptomSearch
    ? fuse.search(symptomSearch).map((r) => r.item).slice(0, 12)
    : SYMPTOM_OPTIONS.slice(0, 15);

  const toggleSymptom = (s: string) => {
    const current = formValues.symptoms || [];
    if (current.includes(s)) setValue('symptoms', current.filter((x) => x !== s));
    else setValue('symptoms', [...current, s]);
  };

  const startVoice = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setVoiceTranscript('[Demo mode: Voice not available in this browser]');
      return;
    }
    const Win = window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognition; SpeechRecognition?: new () => SpeechRecognition };
    const SR = Win.webkitSpeechRecognition || Win.SpeechRecognition;
    if (!SR) {
      setVoiceTranscript('[Demo: use text search]');
      return;
    }
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-IN';
    rec.onresult = (e: SpeechRecognitionEvent) => {
      const t = Array.from(e.results)
        .map((r: SpeechRecognitionResult) => r[0].transcript)
        .join(' ');
      setVoiceTranscript(t);
      const words = t.toLowerCase().split(/\s+/);
      const matched = SYMPTOM_OPTIONS.filter((sym) => words.some((w) => sym.includes(w) || w.includes(sym)));
      matched.forEach((s) => {
        const cur = formValues.symptoms || [];
        if (!cur.includes(s)) setValue('symptoms', [...cur, s]);
      });
    };
    rec.start();
    setIsListening(true);
    rec.onend = () => setIsListening(false);
  }, [formValues.symptoms, setValue]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setOcrProcessing(true);
    setUploadedFile(null);
    try {
      let fullText = '';
      const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
      if (isPdf) {
        const arrayBuffer = await file.arrayBuffer();
        fullText = await extractTextFromPdf(arrayBuffer);
      } else {
        fullText = await file.text();
      }
      if (!fullText.trim()) {
        toast.error(t('triageNoTextExtracted'));
        setOcrProcessing(false);
        return;
      }
      const parsed = parseEhrText(fullText);
      const merged = mergeParsedIntoForm(parsed, formValues);
      if (merged.age != null) setValue('age', merged.age);
      if (merged.gender != null) setValue('gender', merged.gender);
      if (merged.symptoms?.length) setValue('symptoms', merged.symptoms);
      // We still capture parsed vitals/history even if not shown in step 0, 
      // as we'll send them to the backend to "skip" questions
      if (merged.vitals) setValue('vitals', { ...formValues.vitals, ...merged.vitals });
      if (merged.conditions?.length) setValue('conditions', merged.conditions);
      setValue('allergies', merged.allergies ?? '');
      setValue('medications', merged.medications ?? '');

      setUploadedFile({ name: file.name, ocrText: fullText.trim() });
      toast.success(t('triageDocParsed'));
    } catch (err) {
      console.error('PDF/text extraction failed:', err);
      toast.error(t('triageDocParseError'));
    } finally {
      setOcrProcessing(false);
    }
    e.target.value = '';
  };

  /**
   * START SESSION
   * Sends initial data to backend to start active questioning.
   */
  const startAssessment = async () => {
    setAnalyzing(true);
    try {
      const vitals = formValues.vitals || {};
      const bpStr = typeof vitals.bp === 'string' ? vitals.bp : '120/80';
      const tempC = vitals.temp ? fahrenheitToCelsius(ensureNumber(vitals.temp, 98.6)) : undefined;

      const payload: SessionStartRequest = {
        Age: formValues.age ?? 35,
        Gender: formValues.gender ?? 'M',
        Chief_Complaint: (formValues.symptoms?.[0]) ?? 'General checkup',
        Symptoms: (formValues.symptoms ?? []).join(', '),
        // Map vitals if present
        Systolic_BP: vitals.bp ? parseSystolicBp(bpStr) : undefined,
        Heart_Rate: vitals.hr,
        Body_Temperature_C: tempC,
        Respiratory_Rate: vitals.respiratoryRate,
        SpO2: vitals.spo2,
        // Map history/meds
        Allergies: formValues.allergies,
        Current_Medications: formValues.medications,
        Pre_Existing_Conditions: (formValues.conditions ?? []).join(', '),
        // Onset
        Onset_Duration_Value: formValues.onsetDurationValue,
        Onset_Duration_Unit: formValues.onsetDurationUnit,
      };

      // Ideally update startSession to accept more initial knowns or update immediately after.
      // For this implementation, we'll start minimal.

      const res = await startTriageSessionApi(payload);
      setSession(res);
      setStep(1); // Move to questioning
      setCurrentAnswer('');

      // If we have data from EHR that matches the first question, we could auto-answer here
      // But for simplicity, we'll let the user confirm.
    } catch (err) {
      console.error('Session start failed', err);
      toast.error('Connection failed. Starting standard fallback demo.');
      // Fallback to legacy flow
      onAnalyzeLegacy();
    } finally {
      setAnalyzing(false);
    }
  };

  /**
   * ANSWER QUESTION
   */
  const submitAnswer = async () => {
    if (!session?.session_id || !session.question_feature) return;

    // Basic validation
    if (session.question_type === 'numeric' && (currentAnswer === '' || isNaN(Number(currentAnswer)))) {
      setAnswerError('Please enter a valid number');
      return;
    }
    if (session.question_type === 'text' && String(currentAnswer).trim().length === 0) {
      setAnswerError('Please enter an answer');
      return;
    }
    setAnswerError('');
    setAnalyzing(true);

    try {
      let val = currentAnswer;
      if (session.question_type === 'numeric') val = Number(currentAnswer);

      const res = await answerTriageQuestionApi({
        session_id: session.session_id,
        feature: session.question_feature,
        value: val
      });

      setSession(res);
      setCurrentAnswer('');

      if (res.done) {
        handleSessionComplete(res);
      }
    } catch (err) {
      toast.error('Failed to submit answer');
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };



  /**
   * COMPLETION
   */
  const handleSessionComplete = (finalRes: QuestionResponse) => {
    const demo: DemoPatient = {
      id: Date.now(),
      age: formValues.age ?? 35,
      gender: (formValues.gender ?? 'M') as 'M' | 'F' | 'O',
      symptoms: formValues.symptoms ?? [],
      vitals: {
        bp: '120/80', // We might have this in session/known_answers but client types don't expose it easily yet
        hr: 72,
        temp: 98.6,
        respiratoryRate: 18
      },
      conditions: [], // gathered via questions
      aiOutput: {
        risk: (finalRes.current_risk_prediction?.toLowerCase() as RiskLevel) || 'medium',
        confidence: finalRes.confidence || 0.85,
        department: finalRes.current_department || 'General Medicine',
        factors: finalRes.top_features
          ? finalRes.top_features.map((f: any) => ({
            name: typeof f[0] === 'string' ? f[0] : 'Unknown',
            weight: typeof f[1] === 'number' ? f[1] : 0
          }))
          : [{ name: 'Assessment', weight: 1 }],
        reasoning: finalRes.nl_explanation || 'Assessment complete based on your answers.',
      }
    };
    setLastTriageResult(demo);
    navigate('/results');
  };

  /**
   * LEGACY FALLBACK (Network error or Demo mode)
   */
  const onAnalyzeLegacy = async () => {
    // ... (Keep existing fallback logic if needed, or redirect to results with dummy)
    const base = currentPatient ?? demoPatients[0];
    setLastTriageResult({
      ...base,
      id: Date.now(),
      age: formValues.age ?? defaultValues.age,
      gender: formValues.gender ?? defaultValues.gender,
      symptoms: formValues.symptoms ?? [],
      aiOutput: {
        risk: 'medium',
        confidence: 0.8,
        department: 'General Practice',
        factors: [],
        reasoning: 'Demo fallback: specific backend connectivity unavailable.'
      }
    });
    navigate('/results');
  };

  return (
    <div className="relative min-h-[60vh]">
      <div className="absolute inset-0 bg-medical-bg/50 dark:bg-gray-900/50 -z-10" />
      <div className="max-w-3xl mx-auto px-4 py-8">
        <div className="rounded-2xl bg-white/95 dark:bg-gray-800/95 backdrop-blur border border-gray-200/80 dark:border-gray-700/80 shadow-soft-lg p-6 sm:p-8 mb-8">

          {/* HEADER / PROGRESS */}
          {step === 1 && session && (
            <div className="mb-6">
              <div className="flex justify-between items-center text-sm text-gray-500 mb-2">
                <span>Question {session.questions_asked + 1}</span>
                <span>Confidence: {Math.round((session.confidence || 0) * 100)}%</span>
              </div>
              <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-medical-blue transition-all duration-500"
                  style={{ width: `${Math.min(100, (session.confidence || 0) * 100)}%` }}
                />
              </div>
            </div>
          )}

          <AnimatePresence mode="wait">
            {/* STEP 0: INITIAL ENTRY */}
            {step === 0 && (
              <motion.div
                key="step0"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div className="text-center mb-6">
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Patient Intake</h2>
                  <p className="text-gray-600 dark:text-gray-400">Please provide initial details or upload an EHR document.</p>
                </div>

                {/* Upload Section */}
                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">{t('triageUploadDocsMock')}</label>
                  <div className="rounded-xl border border-gray-200 dark:border-gray-600 bg-gray-50/80 dark:bg-gray-800/50 p-4 shadow-soft">
                    <div className="flex flex-wrap gap-2 mb-3">
                      <label className="cursor-pointer">
                        <input type="file" accept=".pdf,application/pdf" className="hidden" onChange={handleFileUpload} />
                        <span className="inline-flex px-4 py-2 rounded-xl bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600 focus-within:ring-2 focus-within:ring-medical-teal">
                          {t('triageUploadPDF')}
                        </span>
                      </label>
                    </div>
                    {ocrProcessing && <p className="text-sm text-medical-teal mb-2">{t('triageProcessingOcr')}</p>}
                    {uploadedFile && !ocrProcessing && (
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        <p className="font-medium text-gray-800 dark:text-gray-200">Loaded: {uploadedFile.name}</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Basic Fields */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">{t('triageAge')}</label>
                    <input
                      type="number"
                      min={0}
                      max={110}
                      {...register('age', { valueAsNumber: true })}
                      className={`w-full rounded-xl border border-gray-300 dark:border-gray-600 px-3 py-2 ${inputFocus}`}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">{t('triageGender')}</label>
                    <div className="flex gap-2">
                      {(['M', 'F', 'O'] as const).map((g) => (
                        <button
                          key={g}
                          type="button"
                          onClick={() => setValue('gender', g)}
                          className={`flex-1 px-4 py-2 rounded-xl border transition-all ${formValues.gender === g ? 'bg-medical-blue text-white border-medical-blue shadow-soft' : 'border-gray-300 dark:border-gray-600 hover:border-medical-teal/50'}`}
                        >
                          {g === 'M' ? t('triageMale') : g === 'F' ? t('triageFemale') : t('triageOther')}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">{t('triageSymptoms')}</label>
                  <div className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={symptomSearch}
                      onChange={(e) => setSymptomSearch(e.target.value)}
                      placeholder={t('triageSearchSymptoms')}
                      className={`flex-1 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 ${inputFocus}`}
                    />
                    <button
                      type="button"
                      onClick={startVoice}
                      className={`px-4 py-2 rounded-xl font-medium transition-all ${isListening ? 'bg-medical-red text-white animate-pulse' : 'bg-medical-teal text-white'}`}
                    >
                      🎤
                    </button>
                  </div>
                  {voiceTranscript && <p className="text-sm text-gray-500 mb-2">{t('triageTranscript')}: {voiceTranscript}</p>}
                  <div className="flex flex-wrap gap-2">
                    {symptomSuggestions.map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => toggleSymptom(s)}
                        className={`px-3 py-1 rounded-full text-sm ${formValues.symptoms?.includes(s) ? 'bg-medical-teal text-white' : 'bg-gray-200 dark:bg-gray-700'}`}
                      >
                        {t(getSymptomKey(s)) || s}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={startAssessment}
                  disabled={analyzing}
                  className="w-full py-4 mt-4 rounded-xl bg-medical-blue hover:bg-medical-blue/90 text-white font-semibold text-lg shadow-soft-lg disabled:opacity-70 flex items-center justify-center gap-2 transition-all"
                >
                  {analyzing ? <span className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent" /> : null}
                  Start Assessment
                </button>
              </motion.div>
            )}

            {/* STEP 1: ACTIVE QUESTIONING */}
            {step === 1 && session && (
              <motion.div
                key="questioning"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6 text-center"
              >
                <h3 className="text-xl font-semibold text-gray-800 dark:text-white mb-8">
                  {session.question_text}
                </h3>

                <div className="max-w-md mx-auto">
                  {session.question_type === 'numeric' && (
                    <input
                      type="number"
                      autoFocus
                      value={currentAnswer}
                      onChange={(e) => setCurrentAnswer(e.target.value)}
                      className={`w-full text-center text-3xl font-bold rounded-xl border-2 border-medical-blue/30 p-4 focus:ring-4 focus:ring-medical-blue/20 outline-none`}
                      placeholder="0"
                      onKeyDown={(e) => e.key === 'Enter' && submitAnswer()}
                    />
                  )}

                  {(session.question_type === 'text' || session.question_type === 'categorical') && (
                    <input
                      type="text"
                      autoFocus
                      value={currentAnswer}
                      onChange={(e) => setCurrentAnswer(e.target.value)}
                      className={`w-full text-lg rounded-xl border-gray-300 p-4 shadow-inner ${inputFocus}`}
                      placeholder="Type answer..."
                      onKeyDown={(e) => e.key === 'Enter' && submitAnswer()}
                    />
                  )}

                  {answerError && <p className="text-red-500 mt-2">{answerError}</p>}
                </div>

                <div className="flex gap-4 justify-center mt-8">
                  <button
                    onClick={submitAnswer}
                    disabled={analyzing}
                    className="px-8 py-3 bg-medical-blue text-white rounded-xl font-semibold shadow-soft hover:shadow-soft-lg transition-all transform active:scale-95"
                  >
                    {analyzing ? 'Submitting...' : 'Next'}
                  </button>
                  {/* Optional Skip
                        <button onClick={skipQuestion} className="px-4 py-3 text-gray-500 hover:text-gray-700">
                            Skip
                        </button>
                        */}
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </div>
      </div>
    </div >
  );
}

