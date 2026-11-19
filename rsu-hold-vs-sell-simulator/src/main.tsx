import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';
import { SiteFooter, SiteHeader } from './site-chrome/SiteChrome';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SiteHeader />
    <App />
    <SiteFooter />
  </StrictMode>,
);
