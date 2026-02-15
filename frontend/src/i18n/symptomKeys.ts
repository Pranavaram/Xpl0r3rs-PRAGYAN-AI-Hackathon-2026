/**
 * Converts a symptom string (e.g. "chest pain") to its translation key (e.g. "symptom_chest_pain").
 * Used so symptom labels can be translated while form/API still use English values.
 */
export function getSymptomKey(symptom: string): string {
  return 'symptom_' + symptom.trim().toLowerCase().replace(/\s+/g, '_').replace(/[<>/]/g, '');
}
