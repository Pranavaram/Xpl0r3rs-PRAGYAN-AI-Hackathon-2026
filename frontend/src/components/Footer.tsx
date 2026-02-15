import { Link } from 'react-router-dom';
import { useLanguage } from '../context/LanguageContext';

export function Footer() {
  const { t } = useLanguage();

  return (
    <footer className="bg-medical-blue text-white mt-auto" role="contentinfo">
      <div className="max-w-7xl mx-auto px-6 py-14 sm:px-8 lg:px-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-16 lg:gap-20">
          {/* Brand */}
          <div className="space-y-3">
            <Link to="/" className="inline-block">
              <span className="font-bold text-2xl tracking-tight text-white">
                Smart Triage
              </span>
            </Link>
            <p className="text-sm text-blue-100/90 max-w-xs leading-relaxed">
              AI-powered precision for modern healthcare.
            </p>
          </div>

          {/* Quick links */}
          <div>
            <h3 className="font-bold text-sm uppercase tracking-wider text-white mb-5">
              {t('footerQuickLinks')}
            </h3>
            <ul className="space-y-3.5 text-sm">
              <li>
                <Link to="/" className="text-blue-100/90 hover:text-white transition-colors">
                  {t('navHome')}
                </Link>
              </li>
              <li>
                <Link to="/triage" className="text-blue-100/90 hover:text-white transition-colors">
                  {t('navTriage')}
                </Link>
              </li>
              <li>
                <Link to="/dashboard" className="text-blue-100/90 hover:text-white transition-colors">
                  {t('navDashboard')}
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-blue-100/90 hover:text-white transition-colors">
                  {t('navAbout')}
                </Link>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="font-bold text-sm uppercase tracking-wider text-white mb-5">
              {t('footerContact')}
            </h3>
            <ul className="space-y-3.5 text-sm text-blue-100/90">
              <li>
                <span className="font-medium text-white/90">{t('footerPhone')}:</span>{' '}
                <a href="tel:+1234567890" className="hover:text-white transition-colors">
                  +1 234 567 8900
                </a>
              </li>
              <li>
                <span className="font-medium text-white/90">{t('footerEmail')}:</span>{' '}
                <a href="mailto:hello@smarttriage.com" className="hover:text-white transition-colors break-all">
                  hello@smarttriage.com
                </a>
              </li>
              <li>
                <span className="font-medium text-white/90">{t('footerLocation')}:</span>{' '}
                <span>123 Healthcare Ave, City, ST 12345</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-white/10">
          <p className="text-center text-sm text-blue-100/80">
            © {new Date().getFullYear()} Smart Triage
          </p>
        </div>
      </div>
    </footer>
  );
}
