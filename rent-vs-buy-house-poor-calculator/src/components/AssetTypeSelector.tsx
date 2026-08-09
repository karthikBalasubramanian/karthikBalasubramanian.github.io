import React from 'react';
import type { AssetCategory, UserHousingInputs } from '../types';
import { Home, Compass, Car, Anchor, Sliders } from 'lucide-react';

interface AssetTypeSelectorProps {
  inputs: UserHousingInputs;
  onChange: (updated: Partial<UserHousingInputs>) => void;
}

export const ASSET_TYPES: { id: AssetCategory; label: string; icon: any; description: string; badge: string }[] = [
  {
    id: 'primary_home',
    label: 'Primary Home',
    icon: Home,
    description: 'PITI, Property Taxes, HOA, & Maintenance Reserve',
    badge: 'Step 2 Core',
  },
  {
    id: 'second_home',
    label: 'Second / Vacation Home',
    icon: Compass,
    description: 'Dual Mortgages, Property Mgmt (10%), & Dual Taxes',
    badge: 'Vacation / Investment',
  },
  {
    id: 'luxury_car',
    label: 'Luxury Vehicle',
    icon: Car,
    description: 'Auto Loan/Lease, Premium Insurance, & Repairs',
    badge: 'Supercar / Exotic',
  },
  {
    id: 'yacht',
    label: 'Yacht / Boat',
    icon: Anchor,
    description: 'Marine Loan, Dockage/Marina, & 10%/yr Maintenance',
    badge: 'Marine Asset',
  },
  {
    id: 'custom',
    label: 'Custom Major Asset',
    icon: Sliders,
    description: 'Custom monthly expense & carrying cost allocation',
    badge: 'Flexible Outlay',
  },
];

export const AssetTypeSelector: React.FC<AssetTypeSelectorProps> = ({ inputs, onChange }) => {
  const currentAsset = inputs.assetType || 'primary_home';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl text-slate-100 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm sm:text-base font-extrabold text-white flex items-center gap-2">
            <span>Select Big Purchase Category to Stress-Test</span>
            <span className="text-[10px] font-mono uppercase bg-indigo-950 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded-full">
              Multi-Asset Engine
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Choose what major asset class you are evaluating against your monthly net paycheck.
          </p>
        </div>
      </div>

      {/* 5 Asset Category Selector Tabs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {ASSET_TYPES.map((type) => {
          const Icon = type.icon;
          const isSelected = currentAsset === type.id;

          return (
            <button
              key={type.id}
              onClick={() => onChange({ assetType: type.id })}
              className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between space-y-2.5 ${
                isSelected
                  ? 'bg-gradient-to-br from-indigo-950/90 via-slate-900 to-slate-900 border-indigo-500 ring-2 ring-indigo-500/30 shadow-lg shadow-indigo-950'
                  : 'bg-slate-950 border-slate-800 hover:bg-slate-800/80 hover:border-slate-700 text-slate-300'
              }`}
            >
              <div className="flex items-center justify-between w-full">
                <div
                  className={`p-2 rounded-lg ${
                    isSelected ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'bg-slate-900 text-slate-400'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                </div>
                <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded-full ${
                  isSelected ? 'bg-indigo-900 text-indigo-200' : 'bg-slate-900 text-slate-500'
                }`}>
                  {type.badge}
                </span>
              </div>

              <div>
                <span className={`text-xs font-bold block ${isSelected ? 'text-white font-extrabold' : 'text-slate-200'}`}>
                  {type.label}
                </span>
                <p className="text-[10px] text-slate-400 leading-snug line-clamp-2 mt-0.5">
                  {type.description}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
