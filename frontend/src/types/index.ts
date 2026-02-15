export type RiskLevel = 'low' | 'medium' | 'high';
export type Gender = 'M' | 'F' | 'O';

export interface Vitals {
  bp: string;
  hr: number;
  temp: number;
  respiratoryRate: number;
  spo2?: number;
}

export interface AIFactor {
  name: string;
  weight: number;
}

export interface AIOutput {
  risk: RiskLevel;
  confidence: number;
  department: string;
  factors: AIFactor[];
  reasoning?: string;
}

export interface DemoPatient {
  id: number;
  age: number;
  gender: Gender;
  symptoms: string[];
  vitals: Vitals;
  conditions: string[];
  allergies?: string;
  medications?: string;
  aiOutput: AIOutput;
  createdAt?: string;
}

export interface TriageFormData {
  age: number;
  gender: Gender;
  symptoms: string[];
  vitals: Vitals;
  onsetDurationValue: number;
  onsetDurationUnit: string;
  conditions: string[];
  allergies: string;
  medications: string;
}
