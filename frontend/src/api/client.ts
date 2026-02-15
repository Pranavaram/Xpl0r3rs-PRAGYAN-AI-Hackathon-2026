import type { TriageFormData, DemoPatient, AIOutput, RiskLevel } from '../types';

/**
 * SOL backend (FastAPI) runs separately, e.g. at http://127.0.0.1:8000
 * Set VITE_API_URL to override (e.g. in production).
 */

/** Request body for SOL POST /triage/predict (PatientRequest) */
interface SolPatientRequest {
  PatientID?: string;
  Age: number;
  Gender: string;
  Chief_Complaint: string;
  Symptoms: string;
  Systolic_BP: number;
  Heart_Rate: number;
  Body_Temperature_C: number;
  Respiratory_Rate: number;
  SpO2: number;
  Onset_Duration_Value: number;
  Onset_Duration_Unit: string;
  Allergies?: string;
  Current_Medications?: string;
  Pre_Existing_Conditions?: string;
}

/** Response from SOL POST /triage/predict (TriageResponse) */
interface SolTriageResponse {
  risk_level?: string;
  risk_source?: string;
  risk_probabilities?: Record<string, number>;
  department?: string;
  department_source?: string;
  rule_reasons?: string[];
  top_features?: (string | number)[][];
  nl_explanation?: string;
}

export function fahrenheitToCelsius(f: number): number {
  return (f - 32) * (5 / 9);
}

export function parseSystolicBp(bp: string): number {
  const parts = bp.split('/').map((s) => parseInt(s.trim(), 10));
  return Number.isFinite(parts[0]) ? parts[0] : 120;
}

export function ensureNumber(value: unknown, fallback: number): number {
  const n = typeof value === 'number' && Number.isFinite(value) ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function mapFormToSolRequest(payload: TriageFormData): SolPatientRequest {
  const symptoms = Array.isArray(payload.symptoms) ? payload.symptoms : [];
  const chiefComplaint = symptoms[0] ?? 'general complaint';
  const vitals = payload.vitals ?? { bp: '120/80', hr: 72, temp: 98.6, respiratoryRate: 18 };
  const bpStr = typeof vitals.bp === 'string' ? vitals.bp : '120/80';
  const tempC = fahrenheitToCelsius(ensureNumber(vitals.temp, 98.6));
  const spo2 = ensureNumber(vitals.spo2, 98);
  const onsetVal = ensureNumber(payload.onsetDurationValue, 1);
  const onsetUnit = typeof payload.onsetDurationUnit === 'string' && payload.onsetDurationUnit
    ? payload.onsetDurationUnit
    : 'hours';

  return {
    Age: ensureNumber(payload.age, 35),
    Gender: payload.gender === 'F' || payload.gender === 'M' || payload.gender === 'O' ? payload.gender : 'M',
    Chief_Complaint: chiefComplaint,
    Symptoms: symptoms.join(', '),
    Systolic_BP: parseSystolicBp(bpStr),
    Heart_Rate: ensureNumber(vitals.hr, 72),
    Body_Temperature_C: tempC,
    Respiratory_Rate: ensureNumber(vitals.respiratoryRate, 18),
    SpO2: spo2,
    Onset_Duration_Value: onsetVal >= 0 ? onsetVal : 1,
    Onset_Duration_Unit: onsetUnit,
    Allergies: payload.allergies?.trim() || undefined,
    Current_Medications: payload.medications?.trim() || undefined,
    Pre_Existing_Conditions: (payload.conditions ?? []).length
      ? (payload.conditions ?? []).join(', ')
      : undefined,
  };
}

function normalizeRisk(riskLevel: string): RiskLevel {
  const r = (riskLevel || '').toLowerCase();
  if (r === 'high') return 'high';
  if (r === 'medium') return 'medium';
  return 'low';
}

function mapSolResponseToDemoPatient(
  sol: SolTriageResponse,
  payload: TriageFormData,
  id: number
): DemoPatient {
  const rawFeatures = sol.top_features ?? [];
  const factorsFromFeatures = rawFeatures.slice(0, 8).map((item) => {
    const name = Array.isArray(item) ? String(item[0] ?? '') : '';
    const value = Array.isArray(item) && item.length > 1 ? Number(item[1]) || 0 : 0;
    return { name, weight: Math.abs(value) };
  }).filter((f) => f.name);

  const ruleReasons = sol.rule_reasons ?? [];
  const fallbackFactors =
    factorsFromFeatures.length > 0
      ? factorsFromFeatures
      : [
        ...ruleReasons.slice(0, 3).map((r, i) => ({ name: r, weight: 0.35 - i * 0.08 })),
        ...(payload.symptoms ?? []).slice(0, 3).map((s, i) => ({ name: s, weight: 0.25 - i * 0.06 })),
      ].filter((f) => f.name && f.weight > 0);

  const total = fallbackFactors.reduce((s, f) => s + f.weight, 0) || 1;
  const normalizedFactors = fallbackFactors.map((f) => ({
    name: f.name,
    weight: Math.round((f.weight / total) * 100) / 100,
  }));

  const riskProbs = sol.risk_probabilities ?? {};
  const riskLevel = sol.risk_level ?? 'Medium';
  const confidence =
    typeof riskProbs[riskLevel] === 'number'
      ? riskProbs[riskLevel]
      : 0.85;

  const aiOutput: AIOutput = {
    risk: normalizeRisk(riskLevel),
    confidence,
    department: (sol.department || 'General Medicine').replace(/_/g, ' '),
    factors: normalizedFactors.length > 0 ? normalizedFactors : [{ name: 'Assessment', weight: 1 }],
    reasoning: sol.nl_explanation ?? '',
  };

  return {
    id,
    age: payload.age ?? 35,
    gender: (payload.gender ?? 'M') as 'M' | 'F' | 'O',
    symptoms: payload.symptoms ?? [],
    vitals: payload.vitals ?? { bp: '120/80', hr: 72, temp: 98.6, respiratoryRate: 18 },
    conditions: payload.conditions ?? [],
    allergies: payload.allergies,
    medications: payload.medications,
    aiOutput,
  };
}

/** Backend base URL (for error messages). */
export function getApiBaseUrl(): string {
  return import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';
}

/**
 * Call SOL backend POST /triage/predict and return a DemoPatient for the Results page.
 * Throws with a clear message if the backend is unreachable or returns an error.
 */
export async function triageApi(payload: TriageFormData): Promise<DemoPatient> {
  const base = getApiBaseUrl();
  const body = mapFormToSolRequest(payload);
  const url = `${base}/triage/predict`;

  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    const msg = e instanceof TypeError && e.message === 'Failed to fetch'
      ? `Cannot connect to backend at ${base}. Start the server (e.g. run your FastAPI app on port 8000) or set VITE_API_URL.`
      : e instanceof Error ? e.message : 'Network error';
    throw new Error(msg);
  }

  if (!res.ok) {
    const errText = await res.text();
    const short = errText.slice(0, 150);
    throw new Error(`Backend returned ${res.status}${short ? `: ${short}` : ''}. Using demo result.`);
  }

  const solResponse: SolTriageResponse = await res.json();
  return mapSolResponseToDemoPatient(solResponse, payload, Date.now());
}

export async function healthApi(): Promise<{ status: string }> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/health`);
  if (!res.ok) throw new Error('Backend unhealthy');
  return res.json();
}

/** Related diseases graph (from backend comorbidity rules). */
export interface RelatedDiseasesGraphNode {
  id: string;
  label: string;
  type: 'condition' | 'symptom';
}

export interface RelatedDiseasesGraphEdge {
  source: string;
  target: string;
  weight: number;
  explanation: string;
  preferred_department?: string;
}

export interface RelatedDiseasesGraphResponse {
  nodes: RelatedDiseasesGraphNode[];
  edges: RelatedDiseasesGraphEdge[];
}

/** Fetch related diseases graph (conditions ↔ symptoms) from backend. */
export async function relatedDiseasesGraphApi(): Promise<RelatedDiseasesGraphResponse> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/triage/related-diseases-graph`);
  if (!res.ok) throw new Error('Failed to load related diseases graph');
  return res.json();
}
/** Active Questioning Types */

export interface SessionStartRequest {
  Age: number;
  Gender: string;
  Chief_Complaint: string;
  Symptoms?: string;
  Systolic_BP?: number;
  Heart_Rate?: number;
  Body_Temperature_C?: number;
  Respiratory_Rate?: number;
  SpO2?: number;
  Onset_Duration_Value?: number;
  Onset_Duration_Unit?: string;
  Allergies?: string;
  Current_Medications?: string;
  Pre_Existing_Conditions?: string;
  max_questions?: number;
  confidence_threshold?: number;
}

export interface SessionAnswerRequest {
  session_id: string;
  feature: string;
  value: any;
}

export interface QuestionResponse {
  session_id: string;
  done: boolean;
  question_feature?: string;
  question_text?: string;
  question_type?: 'numeric' | 'binary' | 'categorical' | 'text';
  current_risk_prediction?: string;
  current_risk_probabilities?: Record<string, number>;
  current_department?: string;
  confidence?: number;
  questions_asked: number;
  rule_triggered: boolean;
  rule_reasons: string[];
  nl_explanation?: string;
  top_features?: (string | number)[][];
  error?: string;
}

/** Start a new active questioning session */
export async function startTriageSessionApi(payload: SessionStartRequest): Promise<QuestionResponse> {
  const base = getApiBaseUrl();
  const url = `${base}/triage/session/start`;

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Session start failed: ${res.status} ${errText}`);
  }

  return res.json();
}

/** Answer a question in the active session */
export async function answerTriageQuestionApi(payload: SessionAnswerRequest): Promise<QuestionResponse> {
  const base = getApiBaseUrl();
  const url = `${base}/triage/session/next`;

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Answer submission failed: ${res.status} ${errText}`);
  }

  return res.json();
}
