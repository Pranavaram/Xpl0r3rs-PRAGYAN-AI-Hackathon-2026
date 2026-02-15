import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Bars3Icon, XMarkIcon, SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { HeartIcon as HeartSolid } from '@heroicons/react/24/solid';
import { useLanguage } from '../context/LanguageContext';

const ROUTES = [
  { path: '/', labelKey: 'navHome' },
  { path: '/triage', labelKey: 'navTriage' },
  { path: '/results', labelKey: 'navResults' },
  { path: '/dashboard', labelKey: 'navDashboard' },
  { path: '/about', labelKey: 'navAbout' },
];

export function Nav() {
  const [open, setOpen] = useState(false);
  const [dark, setDark] = useState(false);
  const location = useLocation();
  const { lang, setLang, t } = useLanguage();

  const toggleDark = () => {
    setDark((d) => !d);
    document.documentElement.classList.toggle('dark', !dark);
  };

  return (
    <nav
      className="sticky top-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur border-b border-gray-200 dark:border-gray-700 shadow-soft"
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-medical-teal rounded">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            >
              <HeartSolid className="h-8 w-8 text-medical-red" aria-hidden />
            </motion.div>
            <span className="font-bold text-lg text-medical-blue dark:text-medical-teal">
              Smart Triage
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-4">
            {ROUTES.map((r) => (
              <Link
                key={r.path}
                to={r.path}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-medical-teal ${
                  location.pathname === r.path
                    ? 'bg-medical-blue/10 text-medical-blue dark:text-medical-teal'
                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                {t(r.labelKey)}
              </Link>
            ))}
            <div className="flex items-center gap-2 pl-2 border-l border-gray-200 dark:border-gray-700">
              <button
                type="button"
                onClick={toggleDark}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-medical-teal"
                aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {dark ? <SunIcon className="h-5 w-5" /> : <MoonIcon className="h-5 w-5" />}
              </button>
              <select
                value={lang}
                onChange={(e) => setLang(e.target.value as 'en' | 'ta' | 'hi')}
                className="text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1.5 focus:ring-2 focus:ring-medical-teal focus:border-medical-teal min-w-[4rem]"
                aria-label="Language"
              >
                <option value="en">English</option>
                <option value="ta">தமிழ்</option>
                <option value="hi">हिंदी</option>
              </select>
            </div>
          </div>

          <button
            type="button"
            className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-label="Toggle menu"
          >
            {open ? <XMarkIcon className="h-6 w-6" /> : <Bars3Icon className="h-6 w-6" />}
          </button>
        </div>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-gray-200 dark:border-gray-700"
            >
              <div className="py-2 flex flex-col gap-1">
                {ROUTES.map((r) => (
                  <Link
                    key={r.path}
                    to={r.path}
                    onClick={() => setOpen(false)}
                    className={`px-4 py-2 rounded-lg ${
                      location.pathname === r.path ? 'bg-medical-blue/10' : ''
                    }`}
                  >
                    {t(r.labelKey)}
                  </Link>
                ))}
                <button type="button" onClick={toggleDark} className="px-4 py-2 text-left">
                  {dark ? 'Light mode' : 'Dark mode'}
                </button>
                <select
                  value={lang}
                  onChange={(e) => setLang(e.target.value as 'en' | 'ta' | 'hi')}
                  className="mx-4 my-2 px-2 py-1.5 rounded border"
                >
                  <option value="en">English</option>
                  <option value="ta">தமிழ்</option>
                  <option value="hi">हिंदी</option>
                </select>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  );
}
