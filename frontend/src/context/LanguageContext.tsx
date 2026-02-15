import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import type { Lang } from '../i18n/translations';
import { t } from '../i18n/translations';

type LangCode = 'en' | 'ta' | 'hi';

const LanguageContext = createContext<{
  lang: LangCode;
  setLang: (l: LangCode) => void;
  t: (key: string) => string;
} | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<LangCode>('en');
  const setLang = useCallback((l: LangCode) => setLangState(l), []);
  const translate = useCallback((key: string) => t(lang as Lang, key), [lang]);
  return (
    <LanguageContext.Provider value={{ lang, setLang, t: translate }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error('useLanguage must be used within LanguageProvider');
  return ctx;
}
