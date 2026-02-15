import { SYMPTOM_OPTIONS } from '../data/dummyData';
import type { TriageFormData, Vitals } from '../types';

const CONDITION_IDS = [
  'hypertension', 'diabetes', 'asthma', 'COPD', 'heart failure', 'smoker', 'obese',
  'allergies', 'osteoporosis', 'kidney disease', 'liver disease', 'cancer history',
];

export interface ParsedEhr {
  age: number | null;
  gender: 'M' | 'F' | 'O' | null;
  symptoms: string[];
  vitals: Partial<Vitals>;
  conditions: string[];
  allergies: string;
  medications: string;
  rawSections: string[];
}

function extractNumber(text: string, pattern: RegExp): number | null {
  const m = text.match(pattern);
  if (!m) return null;
  const n = parseFloat(m[1].replace(/,/g, '.'));
  return Number.isFinite(n) ? n : null;
}

// Section headers that mark the start of a new EHR section (order matters for split).
const SECTION_HEADERS = [
  'age', 'dob', 'date of birth', 'gender', 'sex', 'blood pressure', 'bp ', 'vitals', 'heart rate', 'hr ', 'temp', 'temperature',
  'respiratory', 'rr ', 'symptoms', 'chief complaint', 'diagnosis', 'conditions', 'medical history', 'past medical',
  'onset', 'onset duration', 'duration of', 'generated time', 'date generated', 'time generated',
  'allergies', 'allergic to', 'medications', 'current medications', 'meds', 'rx ', 'prescription', 'notes', 'assessment',
  'plan', 'history of present', 'hpi', 'physical exam', 'lab', 'imaging',
];

/** Split text into sections by known headers. Returns Map of sectionKey -> content (only the content after that header until next header). */
function splitIntoSectionBlocks(text: string): Map<string, string> {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const blocks = new Map<string, string>();
  const headerRegex = new RegExp(
    '\\b(' + SECTION_HEADERS.map((h) => h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')\\s*:?\\s*',
    'gi'
  );
  let lastIndex = 0;
  let lastKey: string | null = null;
  let m: RegExpExecArray | null;
  const re = new RegExp(headerRegex.source, 'gi');
  while ((m = re.exec(normalized)) !== null) {
    const key = m[1].toLowerCase().trim();
    const contentStart = m.index + m[0].length;
    if (lastKey) {
      const content = normalized.slice(lastIndex, m.index).trim();
      if (content.length > 0 && content.length < 2000) blocks.set(lastKey, content);
    }
    lastKey = key;
    lastIndex = contentStart;
  }
  if (lastKey) {
    const content = normalized.slice(lastIndex).trim();
    if (content.length > 0 && content.length < 2000) blocks.set(lastKey, content);
  }
  return blocks;
}

/** Get content for a section by matching any of the given keys (e.g. allergies section). */
function getSectionContent(blocks: Map<string, string>, ...keys: string[]): string {
  const lowerKeys = keys.map((k) => k.toLowerCase());
  for (const [blockKey, content] of blocks) {
    const b = blockKey.toLowerCase();
    const match = lowerKeys.some((k) => b.includes(k) || k.includes(b));
    if (match) return content.trim().replace(/\s+/g, ' ').slice(0, 500);
  }
  return '';
}

/**
 * Parse raw EHR/PDF text into structured form data.
 * Splits by section headers first so each field gets only its own section content.
 */
export function parseEhrText(text: string): ParsedEhr {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const blocks = splitIntoSectionBlocks(normalized);

  const result: ParsedEhr = {
    age: null,
    gender: null,
    symptoms: [],
    vitals: {},
    conditions: [],
    allergies: '',
    medications: '',
    rawSections: [],
  };

  // Age: from full text or from "age" / "dob" section
  const ageBlock = getSectionContent(blocks, 'age', 'dob', 'date of birth') || normalized;
  let age = extractNumber(ageBlock, /\b(?:age|y\/o|y\.o\.?|years?\s+old)\s*:?\s*(\d{1,3})\b/i);
  if (age == null) age = extractNumber(normalized, /\b(\d{1,3})\s*(?:years?\s+old|y\/o|y\.o\.?)\b/i);
  if (age == null) {
    const dobMatch = normalized.match(/\b(?:dob|date\s+of\s+birth|birth\s+date)\s*:?\s*(\d{1,4})[-\/](\d{1,2})[-\/](\d{1,4})/i);
    if (dobMatch) {
      const year = parseInt(dobMatch[1].length === 4 ? dobMatch[1] : dobMatch[3], 10);
      const currentYear = new Date().getFullYear();
      age = currentYear - year;
      if (age < 0 || age > 120) age = null;
    }
  }
  if (age != null && age >= 0 && age <= 120) result.age = age;

  // Gender
  if (/\b(?:gender|sex)\s*:?\s*m(?:ale)?\b/i.test(normalized) || (/\bmale\b/i.test(normalized) && !/\bfemale\b/i.test(normalized)))
    result.gender = 'M';
  else if (/\b(?:gender|sex)\s*:?\s*f(?:emale)?\b/i.test(normalized) || /\bfemale\b/i.test(normalized))
    result.gender = 'F';

  // Vitals from full text (numbers are easy to find)
  const bpMatch = normalized.match(/\b(?:bp|blood\s*pressure)\s*:?\s*(\d{2,3})\s*[\/\-]\s*(\d{2,3})/i)
    || normalized.match(/\b(\d{2,3})\s*[\/\-]\s*(\d{2,3})\s*(?:mmhg|mmHg)?/);
  if (bpMatch) {
    const sys = parseInt(bpMatch[1], 10);
    const dia = parseInt(bpMatch[2], 10);
    if (sys >= 60 && sys <= 250 && dia >= 40 && dia <= 150) result.vitals.bp = `${sys}/${dia}`;
  }
  let hr = extractNumber(normalized, /\b(?:hr|heart\s*rate|pulse|bpm)\s*:?\s*(\d{2,3})\b/i);
  if (hr == null) hr = extractNumber(normalized, /\b(\d{2,3})\s*bpm\b/i);
  if (hr != null && hr >= 40 && hr <= 200) result.vitals.hr = hr;
  let temp = extractNumber(normalized, /\b(?:temp(?:erature)?|t)\s*:?\s*(\d{2,3}\.?\d*)\s*[fF°]?\b/i);
  if (temp == null) temp = extractNumber(normalized, /\b(\d{2,3}\.?\d*)\s*[fF°]\b/);
  if (temp == null) {
    const cTemp = extractNumber(normalized, /\b(?:temp(?:erature)?|t)\s*:?\s*(\d{2}\.?\d*)\s*[cC°]?\b/i);
    if (cTemp != null && cTemp >= 35 && cTemp <= 43) temp = (cTemp * 9) / 5 + 32;
  }
  if (temp != null && temp >= 95 && temp <= 108) result.vitals.temp = temp;
  let rr = extractNumber(normalized, /\b(?:rr|resp(?:iratory)?\s*rate?|breathing)\s*:?\s*(\d{1,2})\b/i);
  if (rr == null) rr = extractNumber(normalized, /\b(\d{1,2})\s*(?:\/min|breaths)/i);
  if (rr != null && rr >= 8 && rr <= 60) result.vitals.respiratoryRate = rr;

  // Symptoms: match known symptom phrases in full text
  for (const symptom of SYMPTOM_OPTIONS) {
    const regex = new RegExp('\\b' + symptom.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
    if (regex.test(normalized)) result.symptoms.push(symptom);
  }

  // Conditions
  for (const id of CONDITION_IDS) {
    const label = id.replace(/_/g, ' ');
    const re = new RegExp('\\b' + id.replace(/\s+/g, '\\s+') + '\\b|\\b' + label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
    if (re.test(normalized)) result.conditions.push(id);
  }

  // Allergies: only from a section that is clearly allergies (not the whole doc).
  // Strip any lines/phrases that are onset duration or generated time (not allergy info).
  function cleanAllergiesText(raw: string): string {
    const lines = raw.split(/\n/).filter((line) => {
      const t = line.trim().toLowerCase();
      if (!t) return false;
      if (/\bonset\b.*\bduration\b|\bduration\b.*\bonset\b/i.test(t)) return false;
      if (/\bgenerated\s+time\b|\bdate\s+generated\b|\btime\s+generated\b|\bgenerated\s+date\b/i.test(t)) return false;
      if (/^\d+\s*(?:hours?|days?|weeks?|months?|years?)\s*$/i.test(t)) return false;
      return true;
    });
    const joined = lines.join(' ').replace(/\s+/g, ' ').trim();
    return joined.replace(/\bonset\s*duration\s*:?\s*\d*\s*(?:hours?|days?|weeks?|months?|years?)\s*/gi, '').replace(/\bgenerated\s+time\s*:?\s*[^\s,;]+/gi, '').trim();
  }
  let allergiesText = getSectionContent(blocks, 'allergies', 'allergic to');
  if (!allergiesText) {
    const oneLineAllergy = normalized.match(/\b(?:allergies?|allergic\s+to)\s*:?\s*([^\n]+)/i);
    if (oneLineAllergy) allergiesText = oneLineAllergy[1].trim().replace(/\s+/g, ' ').slice(0, 300);
  }
  if (allergiesText) {
    allergiesText = cleanAllergiesText(allergiesText);
    if (!allergiesText || /\bnkda\b|n\/k\/d\/a|no\s+known\s+drug\s+allergies?\b/i.test(allergiesText))
      result.allergies = 'No known drug allergies';
    else
      result.allergies = allergiesText;
  } else if (/\bnkda\b|nka\b|no\s+known\s+allergies?\b/i.test(normalized))
    result.allergies = 'No known drug allergies';

  // Medications: only from medications/meds/rx section (or single line fallback)
  let medsText = getSectionContent(blocks, 'medications', 'current medications', 'meds', 'rx', 'prescription');
  if (!medsText) {
    const oneLineMeds = normalized.match(/\b(?:medications?|current\s+medications?|meds?|rx)\s*:?\s*([^\n]+)/i);
    if (oneLineMeds) medsText = oneLineMeds[1].trim().replace(/\s+/g, ' ').slice(0, 300);
  }
  if (medsText) result.medications = medsText;

  result.rawSections = normalized.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
  return result;
}

/**
 * Merge parsed EHR into form default values (only overwrite when parsed value exists).
 */
export function mergeParsedIntoForm(
  parsed: ParsedEhr,
  current: Partial<TriageFormData>
): Partial<TriageFormData> {
  const out: Partial<TriageFormData> = { ...current };
  if (parsed.age != null) out.age = parsed.age;
  if (parsed.gender != null) out.gender = parsed.gender;
  if (parsed.symptoms.length) out.symptoms = [...new Set([...(current.symptoms || []), ...parsed.symptoms])];
  if (Object.keys(parsed.vitals).length) {
    out.vitals = {
      bp: parsed.vitals.bp ?? current.vitals?.bp ?? '120/80',
      hr: parsed.vitals.hr ?? current.vitals?.hr ?? 72,
      temp: parsed.vitals.temp ?? current.vitals?.temp ?? 98.6,
      respiratoryRate: parsed.vitals.respiratoryRate ?? current.vitals?.respiratoryRate ?? 18,
      spo2: current.vitals?.spo2,
    };
  }
  if (parsed.conditions.length) out.conditions = [...new Set([...(current.conditions || []), ...parsed.conditions])];
  if (parsed.allergies) out.allergies = parsed.allergies;
  if (parsed.medications) out.medications = parsed.medications;
  return out;
}
