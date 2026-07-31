import { StateTaxInfo } from '../types';

export const TAX_LIMITS_2026 = {
  TRADITIONAL_401K_MAX: 23500,
  TRADITIONAL_401K_CATCHUP: 7500, // Age 50+
  IRA_MAX: 7000,
  IRA_CATCHUP: 1000, // Age 50+
  HSA_SINGLE_MAX: 4300,
  HSA_FAMILY_MAX: 8550,
  HSA_CATCHUP: 1000, // Age 55+
  FSA_MAX: 3200,

  SOCIAL_SECURITY_RATE: 0.062,
  SOCIAL_SECURITY_WAGE_CAP: 176100,

  MEDICARE_RATE: 0.0145,
  ADDITIONAL_MEDICARE_RATE: 0.009,
  ADDITIONAL_MEDICARE_THRESHOLD_SINGLE: 200000,
  ADDITIONAL_MEDICARE_THRESHOLD_MARRIED: 250000,

  FEDERAL_STANDARD_DEDUCTION: {
    single: 15000,
    married: 30000,
    head_of_household: 22500,
  },

  FEDERAL_BRACKETS: {
    single: [
      { min: 0, max: 11925, rate: 0.10 },
      { min: 11925, max: 48475, rate: 0.12 },
      { min: 48475, max: 103350, rate: 0.22 },
      { min: 103350, max: 197300, rate: 0.24 },
      { min: 197300, max: 250525, rate: 0.32 },
      { min: 250525, max: 626350, rate: 0.35 },
      { min: 626350, max: Infinity, rate: 0.37 },
    ],
    married: [
      { min: 0, max: 23850, rate: 0.10 },
      { min: 23850, max: 96950, rate: 0.12 },
      { min: 96950, max: 206700, rate: 0.22 },
      { min: 206700, max: 394600, rate: 0.24 },
      { min: 394600, max: 501050, rate: 0.32 },
      { min: 501050, max: 751600, rate: 0.35 },
      { min: 751600, max: Infinity, rate: 0.37 },
    ],
    head_of_household: [
      { min: 0, max: 17000, rate: 0.10 },
      { min: 17000, max: 64850, rate: 0.12 },
      { min: 64850, max: 103350, rate: 0.22 },
      { min: 103350, max: 197300, rate: 0.24 },
      { min: 197300, max: 250500, rate: 0.32 },
      { min: 250500, max: 626350, rate: 0.35 },
      { min: 626350, max: Infinity, rate: 0.37 },
    ],
  },
};

export const US_STATES: Record<string, StateTaxInfo> = {
  CA: {
    code: 'CA',
    name: 'California',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.011, // 1.1% CASDI
    type: 'progressive',
    brackets: {
      single: [
        { min: 0, max: 10412, rate: 0.01 },
        { min: 10412, max: 24684, rate: 0.02 },
        { min: 24684, max: 38959, rate: 0.04 },
        { min: 38959, max: 54081, rate: 0.06 },
        { min: 54081, max: 68350, rate: 0.08 },
        { min: 68350, max: 349137, rate: 0.093 },
        { min: 349137, max: 418961, rate: 0.103 },
        { min: 418961, max: 698271, rate: 0.113 },
        { min: 698271, max: Infinity, rate: 0.123 },
      ],
      married: [
        { min: 0, max: 20824, rate: 0.01 },
        { min: 20824, max: 49368, rate: 0.02 },
        { min: 49368, max: 77918, rate: 0.04 },
        { min: 77918, max: 108162, rate: 0.06 },
        { min: 108162, max: 136700, rate: 0.08 },
        { min: 136700, max: 698274, rate: 0.093 },
        { min: 698274, max: 837922, rate: 0.103 },
        { min: 837922, max: 1396542, rate: 0.113 },
        { min: 1396542, max: Infinity, rate: 0.123 },
      ],
    },
    notes: 'Includes 1.1% California State Disability Insurance (CASDI).',
  },
  NY: {
    code: 'NY',
    name: 'New York',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.005, // ~0.5% NY SDI/PFL
    type: 'progressive',
    brackets: {
      single: [
        { min: 0, max: 8500, rate: 0.04 },
        { min: 8500, max: 11700, rate: 0.045 },
        { min: 11700, max: 13900, rate: 0.0525 },
        { min: 13900, max: 80650, rate: 0.055 },
        { min: 80650, max: 215400, rate: 0.06 },
        { min: 215400, max: 1077550, rate: 0.0685 },
        { min: 1077550, max: Infinity, rate: 0.0965 },
      ],
      married: [
        { min: 0, max: 17150, rate: 0.04 },
        { min: 17150, max: 23600, rate: 0.045 },
        { min: 23600, max: 27900, rate: 0.0525 },
        { min: 27900, max: 161550, rate: 0.055 },
        { min: 161550, max: 323200, rate: 0.06 },
        { min: 323200, max: 2155350, rate: 0.0685 },
        { min: 2155350, max: Infinity, rate: 0.0965 },
      ],
    },
    notes: 'Includes NY State Income Tax and NY Disability/PFL contributions.',
  },
  TX: {
    code: 'TX',
    name: 'Texas',
    hasStateTax: false,
    hasSDI: false,
    sdiRate: 0,
    type: 'none',
    notes: 'No state income tax!',
  },
  FL: {
    code: 'FL',
    name: 'Florida',
    hasStateTax: false,
    hasSDI: false,
    sdiRate: 0,
    type: 'none',
    notes: 'No state income tax!',
  },
  WA: {
    code: 'WA',
    name: 'Washington',
    hasStateTax: false,
    hasSDI: true,
    sdiRate: 0.0074, // Paid Family & Medical Leave
    type: 'none',
    notes: 'No income tax, but includes WA PFML (Paid Family & Medical Leave ~0.74%).',
  },
  MA: {
    code: 'MA',
    name: 'Massachusetts',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.00318, // PFML
    type: 'flat',
    flatRate: 0.05, // 5% flat
    notes: '5.0% flat income tax + MA PFML deduction.',
  },
  IL: {
    code: 'IL',
    name: 'Illinois',
    hasStateTax: true,
    hasSDI: false,
    sdiRate: 0,
    type: 'flat',
    flatRate: 0.0495, // 4.95% flat
    notes: '4.95% flat state income tax.',
  },
  PA: {
    code: 'PA',
    name: 'Pennsylvania',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.0007,
    type: 'flat',
    flatRate: 0.0307, // 3.07% flat
    notes: '3.07% flat state income tax.',
  },
  NJ: {
    code: 'NJ',
    name: 'New Jersey',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.0035, // FLI + TDI
    type: 'progressive',
    brackets: {
      single: [
        { min: 0, max: 20000, rate: 0.014 },
        { min: 20000, max: 35000, rate: 0.0175 },
        { min: 35000, max: 40000, rate: 0.035 },
        { min: 40000, max: 75000, rate: 0.05525 },
        { min: 75000, max: 500000, rate: 0.0637 },
        { min: 500000, max: Infinity, rate: 0.0897 },
      ],
      married: [
        { min: 0, max: 20000, rate: 0.014 },
        { min: 20000, max: 50000, rate: 0.0175 },
        { min: 50000, max: 70000, rate: 0.0245 },
        { min: 70000, max: 80000, rate: 0.035 },
        { min: 80000, max: 150000, rate: 0.05525 },
        { min: 150000, max: 500000, rate: 0.0637 },
        { min: 500000, max: Infinity, rate: 0.0897 },
      ],
    },
    notes: 'NJ State Tax + Family Leave / Disability Insurance.',
  },
  VA: {
    code: 'VA',
    name: 'Virginia',
    hasStateTax: true,
    hasSDI: false,
    sdiRate: 0,
    type: 'progressive',
    brackets: {
      single: [
        { min: 0, max: 3000, rate: 0.02 },
        { min: 3000, max: 5000, rate: 0.03 },
        { min: 5000, max: 17000, rate: 0.0475 },
        { min: 17000, max: Infinity, rate: 0.0575 },
      ],
      married: [
        { min: 0, max: 3000, rate: 0.02 },
        { min: 3000, max: 5000, rate: 0.03 },
        { min: 5000, max: 17000, rate: 0.0475 },
        { min: 17000, max: Infinity, rate: 0.0575 },
      ],
    },
  },
  GA: {
    code: 'GA',
    name: 'Georgia',
    hasStateTax: true,
    hasSDI: false,
    sdiRate: 0,
    type: 'flat',
    flatRate: 0.0539,
    notes: '5.39% flat income tax rate.',
  },
  NC: {
    code: 'NC',
    name: 'North Carolina',
    hasStateTax: true,
    hasSDI: false,
    sdiRate: 0,
    type: 'flat',
    flatRate: 0.045,
    notes: '4.5% flat state income tax rate.',
  },
  CO: {
    code: 'CO',
    name: 'Colorado',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.0045, // FAMLI paid leave
    type: 'flat',
    flatRate: 0.044,
    notes: '4.4% flat state income tax + CO FAMLI paid leave.',
  },
  NV: { code: 'NV', name: 'Nevada', hasStateTax: false, hasSDI: false, sdiRate: 0, type: 'none' },
  TN: { code: 'TN', name: 'Tennessee', hasStateTax: false, hasSDI: false, sdiRate: 0, type: 'none' },
  WY: { code: 'WY', name: 'Wyoming', hasStateTax: false, hasSDI: false, sdiRate: 0, type: 'none' },
  SD: { code: 'SD', name: 'South Dakota', hasStateTax: false, hasSDI: false, sdiRate: 0, type: 'none' },
  AK: { code: 'AK', name: 'Alaska', hasStateTax: false, hasSDI: false, sdiRate: 0, type: 'none' },

  // Generic fallback for any other state
  OTHER: {
    code: 'OTHER',
    name: 'Other / Average State (~4.5%)',
    hasStateTax: true,
    hasSDI: false,
    sdiRate: 0,
    type: 'flat',
    flatRate: 0.045,
    notes: 'Estimated 4.5% flat state income tax.',
  },
};
