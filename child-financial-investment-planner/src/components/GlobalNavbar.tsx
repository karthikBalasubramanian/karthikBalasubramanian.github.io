import React from 'react';
import { ArrowLeft, ExternalLink } from 'lucide-react';

interface GlobalNavbarProps {
  currentAppName: string;
}

export const GlobalNavbar: React.FC<GlobalNavbarProps> = ({ currentAppName }) => {
  return (
    <div className="bg-[#2f4050] text-white border-b border-slate-700/60 py-2.5 px-4 sm:px-8">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Back Link to Main Portfolio */}
        <a
          href="../"
          className="inline-flex items-center gap-2 text-xs font-semibold text-slate-200 hover:text-white bg-slate-800/80 hover:bg-slate-700/90 px-3 py-1.5 rounded-md transition-all shadow-xs"
        >
          <ArrowLeft className="w-3.5 h-3.5 text-[#1ab394]" />
          <span>Back to Karthik&apos;s Portfolio</span>
        </a>

        {/* Brand & App Title Badge */}
        <div className="flex items-center gap-2 sm:gap-3">
          <span className="text-xs font-bold tracking-wide text-slate-300 hidden sm:inline">
            Karthik Balasubramanian
          </span>
          <span className="text-slate-500 hidden sm:inline">|</span>
          <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold bg-[#1ab394]/20 text-[#1ab394] border border-[#1ab394]/30 px-2.5 py-0.5 rounded-full">
            {currentAppName}
          </span>
        </div>
      </div>
    </div>
  );
};
