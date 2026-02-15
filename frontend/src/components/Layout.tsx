import { useEffect } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Nav } from './Nav';
import { Footer } from './Footer';
import { WaterWaveEffect } from './WaterWaveEffect';

export function Layout() {
  const navigate = useNavigate();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 't' || e.key === 'T') navigate('/dashboard');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [navigate]);

  // HD medical/hospital background — premium: partly visible like enterprise sites
  const bgImageUrl =
    'https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=2560&q=90';

  return (
    <div className="min-h-screen flex flex-col text-gray-900 dark:text-gray-100 relative">
      {/* Full-viewport HD medical background — visible through gradient overlay */}
      <div
        className="fixed inset-0 z-0 bg-cover bg-center bg-no-repeat min-h-screen"
        style={{
          backgroundImage: `url(${bgImageUrl})`,
          backgroundAttachment: 'fixed',
        }}
        aria-hidden
      />
      {/* Premium overlay: gradient so image shows through (top clearer, bottom subtle tint) */}
      <div
        className="fixed inset-0 z-0 min-h-screen pointer-events-none dark:hidden"
        style={{
          background: 'linear-gradient(180deg, rgba(255,255,255,0.68) 0%, rgba(248,250,252,0.78) 45%, rgba(241,245,249,0.85) 100%)',
        }}
        aria-hidden
      />
      {/* Dark mode: same idea, image partly visible */}
      <div
        className="fixed inset-0 z-0 min-h-screen pointer-events-none hidden dark:block"
        style={{
          background: 'linear-gradient(180deg, rgba(15,23,42,0.72) 0%, rgba(30,41,59,0.8) 50%, rgba(15,23,42,0.88) 100%)',
        }}
        aria-hidden
      />
      <WaterWaveEffect />
      <div className="relative z-10 flex flex-col min-h-screen flex-1">
        <Nav />
        <main className="flex-1" role="main">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className="w-full h-full"
          >
            <Outlet />
          </motion.div>
        </main>
        <Footer />
      </div>
    </div>
  );
}
