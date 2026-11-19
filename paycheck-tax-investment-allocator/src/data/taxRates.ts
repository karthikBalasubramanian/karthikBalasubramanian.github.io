import { StateTaxInfo } from '../types';

export interface TaxLimits {
  TRADITIONAL_401K_MAX: number;
  TRADITIONAL_401K_CATCHUP: number;
  TRADITIONAL_401K_SUPER_CATCHUP?: number; // SECURE 2.0 ages 60-63
  COMPENSATION_LIMIT_401K: number;
  TOTAL_401K_ADDITION_415C: number;
  IRA_MAX: number;
  IRA_CATCHUP: number;
  HSA_SINGLE_MAX: number;
  HSA_FAMILY_MAX: number;
  HSA_CATCHUP: number;
  FSA_MAX: number;

  SOCIAL_SECURITY_RATE: number;
  SOCIAL_SECURITY_WAGE_CAP: number;

  MEDICARE_RATE: number;
  ADDITIONAL_MEDICARE_RATE: number;
  ADDITIONAL_MEDICARE_THRESHOLD_SINGLE: number;
  ADDITIONAL_MEDICARE_THRESHOLD_MARRIED: number;

  FEDERAL_STANDARD_DEDUCTION: {
    single: number;
    married: number;
    head_of_household: number;
  };

  FEDERAL_BRACKETS: {
    single: { min: number; max: number; rate: number }[];
    married: { min: number; max: number; rate: number }[];
    head_of_household: { min: number; max: number; rate: number }[];
  };
}

export const TAX_LIMITS_BY_YEAR: Record<number, TaxLimits> = {
  2024: {
    TRADITIONAL_401K_MAX: 23000,
    TRADITIONAL_401K_CATCHUP: 7500,
    COMPENSATION_LIMIT_401K: 345000,
    TOTAL_401K_ADDITION_415C: 69000,
    IRA_MAX: 7000,
    IRA_CATCHUP: 1000,
    HSA_SINGLE_MAX: 4150,
    HSA_FAMILY_MAX: 8300,
    HSA_CATCHUP: 1000,
    FSA_MAX: 3200,
    SOCIAL_SECURITY_RATE: 0.062,
    SOCIAL_SECURITY_WAGE_CAP: 168600,
    MEDICARE_RATE: 0.0145,
    ADDITIONAL_MEDICARE_RATE: 0.009,
    ADDITIONAL_MEDICARE_THRESHOLD_SINGLE: 200000,
    ADDITIONAL_MEDICARE_THRESHOLD_MARRIED: 250000,
    FEDERAL_STANDARD_DEDUCTION: {
      single: 14600,
      married: 29200,
      head_of_household: 21900,
    },
    FEDERAL_BRACKETS: {
      single: [
        { min: 0, max: 11600, rate: 0.10 },
        { min: 11600, max: 47150, rate: 0.12 },
        { min: 47150, max: 100525, rate: 0.22 },
        { min: 100525, max: 191950, rate: 0.24 },
        { min: 191950, max: 243725, rate: 0.32 },
        { min: 243725, max: 609350, rate: 0.35 },
        { min: 609350, max: Infinity, rate: 0.37 },
      ],
      married: [
        { min: 0, max: 23200, rate: 0.10 },
        { min: 23200, max: 94300, rate: 0.12 },
        { min: 94300, max: 201050, rate: 0.22 },
        { min: 201050, max: 383900, rate: 0.24 },
        { min: 383900, max: 487450, rate: 0.32 },
        { min: 487450, max: 731200, rate: 0.35 },
        { min: 731200, max: Infinity, rate: 0.37 },
      ],
      head_of_household: [
        { min: 0, max: 16550, rate: 0.10 },
        { min: 16550, max: 63100, rate: 0.12 },
        { min: 63100, max: 100500, rate: 0.22 },
        { min: 100500, max: 191950, rate: 0.24 },
        { min: 191950, max: 243700, rate: 0.32 },
        { min: 243700, max: 609350, rate: 0.35 },
        { min: 609350, max: Infinity, rate: 0.37 },
      ],
    },
  },

  2025: {
    TRADITIONAL_401K_MAX: 23500,
    TRADITIONAL_401K_CATCHUP: 7500,
    COMPENSATION_LIMIT_401K: 350000,
    TOTAL_401K_ADDITION_415C: 70000,
    IRA_MAX: 7000,
    IRA_CATCHUP: 1000,
    HSA_SINGLE_MAX: 4300,
    HSA_FAMILY_MAX: 8550,
    HSA_CATCHUP: 1000,
    FSA_MAX: 3300,
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
  },

  2026: {
    TRADITIONAL_401K_MAX: 24500,
    TRADITIONAL_401K_CATCHUP: 8000,
    TRADITIONAL_401K_SUPER_CATCHUP: 11250,
    COMPENSATION_LIMIT_401K: 360000,
    TOTAL_401K_ADDITION_415C: 72000,
    IRA_MAX: 7500,
    IRA_CATCHUP: 1100,
    HSA_SINGLE_MAX: 4400,
    HSA_FAMILY_MAX: 8750,
    HSA_CATCHUP: 1000,
    FSA_MAX: 3400,
    SOCIAL_SECURITY_RATE: 0.062,
    SOCIAL_SECURITY_WAGE_CAP: 184500,
    MEDICARE_RATE: 0.0145,
    ADDITIONAL_MEDICARE_RATE: 0.009,
    ADDITIONAL_MEDICARE_THRESHOLD_SINGLE: 200000,
    ADDITIONAL_MEDICARE_THRESHOLD_MARRIED: 250000,
    FEDERAL_STANDARD_DEDUCTION: {
      single: 16100,
      married: 32200,
      head_of_household: 24150,
    },
    FEDERAL_BRACKETS: {
      single: [
        { min: 0, max: 12400, rate: 0.10 },
        { min: 12400, max: 50400, rate: 0.12 },
        { min: 50400, max: 105700, rate: 0.22 },
        { min: 105700, max: 201775, rate: 0.24 },
        { min: 201775, max: 256225, rate: 0.32 },
        { min: 256225, max: 640600, rate: 0.35 },
        { min: 640600, max: Infinity, rate: 0.37 },
      ],
      married: [
        { min: 0, max: 24800, rate: 0.10 },
        { min: 24800, max: 100800, rate: 0.12 },
        { min: 100800, max: 211400, rate: 0.22 },
        { min: 211400, max: 403550, rate: 0.24 },
        { min: 403550, max: 512450, rate: 0.32 },
        { min: 512450, max: 768700, rate: 0.35 },
        { min: 768700, max: Infinity, rate: 0.37 },
      ],
      head_of_household: [
        { min: 0, max: 17700, rate: 0.10 },
        { min: 17700, max: 67450, rate: 0.12 },
        { min: 67450, max: 105700, rate: 0.22 },
        { min: 105700, max: 201750, rate: 0.24 },
        { min: 201750, max: 256200, rate: 0.32 },
        { min: 256200, max: 640600, rate: 0.35 },
        { min: 640600, max: Infinity, rate: 0.37 },
      ],
    },
  },

  // Official 2027 IRS inflation adjustments (Notice 2026-66, Rev. Proc. 2026-35, Rev. Proc. 2026-24, SSA 2027)
  2027: {
    TRADITIONAL_401K_MAX: 25500,
    TRADITIONAL_401K_CATCHUP: 8500,
    TRADITIONAL_401K_SUPER_CATCHUP: 11750,
    COMPENSATION_LIMIT_401K: 375000,
    TOTAL_401K_ADDITION_415C: 75000,
    IRA_MAX: 7500,
    IRA_CATCHUP: 1100,
    HSA_SINGLE_MAX: 4500,
    HSA_FAMILY_MAX: 9000,
    HSA_CATCHUP: 1000,
    FSA_MAX: 3500,
    SOCIAL_SECURITY_RATE: 0.062,
    SOCIAL_SECURITY_WAGE_CAP: 192300,
    MEDICARE_RATE: 0.0145,
    ADDITIONAL_MEDICARE_RATE: 0.009,
    ADDITIONAL_MEDICARE_THRESHOLD_SINGLE: 200000,
    ADDITIONAL_MEDICARE_THRESHOLD_MARRIED: 250000,
    FEDERAL_STANDARD_DEDUCTION: {
      single: 16600,
      married: 33200,
      head_of_household: 24925,
    },
    FEDERAL_BRACKETS: {
      single: [
        { min: 0, max: 12800, rate: 0.10 },
        { min: 12800, max: 52025, rate: 0.12 },
        { min: 52025, max: 109125, rate: 0.22 },
        { min: 109125, max: 208325, rate: 0.24 },
        { min: 208325, max: 264550, rate: 0.32 },
        { min: 264550, max: 661375, rate: 0.35 },
        { min: 661375, max: Infinity, rate: 0.37 },
      ],
      married: [
        { min: 0, max: 25600, rate: 0.10 },
        { min: 25600, max: 104050, rate: 0.12 },
        { min: 104050, max: 218250, rate: 0.22 },
        { min: 218250, max: 416650, rate: 0.24 },
        { min: 416650, max: 529100, rate: 0.32 },
        { min: 529100, max: 793650, rate: 0.35 },
        { min: 793650, max: Infinity, rate: 0.37 },
      ],
      head_of_household: [
        { min: 0, max: 18250, rate: 0.10 },
        { min: 18250, max: 69650, rate: 0.12 },
        { min: 69650, max: 109100, rate: 0.22 },
        { min: 109100, max: 208300, rate: 0.24 },
        { min: 208300, max: 264550, rate: 0.32 },
        { min: 264550, max: 661375, rate: 0.35 },
        { min: 661375, max: Infinity, rate: 0.37 },
      ],
    },
  },
};

/**
 * Returns grounded IRS Tax Limits for any given year.
 * If year > 2027, automatically projects future limits using IRS Chained CPI-U inflation steps.
 */
export function getTaxLimitsForYear(year: number = 2027): TaxLimits {
  if (year <= 2024) return TAX_LIMITS_BY_YEAR[2024];
  if (year === 2025) return TAX_LIMITS_BY_YEAR[2025];
  if (year === 2026) return TAX_LIMITS_BY_YEAR[2026];
  if (year === 2027) return TAX_LIMITS_BY_YEAR[2027];

  // For future years (2028+), project limits using IRS indexing rules
  const base2027 = TAX_LIMITS_BY_YEAR[2027];
  const yearsAhead = year - 2027;
  const inflationMult = Math.pow(1.025, yearsAhead); // ~2.5% inflation per year

  return {
    ...base2027,
    TRADITIONAL_401K_MAX: base2027.TRADITIONAL_401K_MAX + Math.floor(yearsAhead * 500),
    COMPENSATION_LIMIT_401K: base2027.COMPENSATION_LIMIT_401K + yearsAhead * 5000,
    TOTAL_401K_ADDITION_415C: base2027.TOTAL_401K_ADDITION_415C + yearsAhead * 1500,
    IRA_MAX: base2027.IRA_MAX + Math.floor(yearsAhead * 500),
    HSA_SINGLE_MAX: base2027.HSA_SINGLE_MAX + yearsAhead * 100,
    HSA_FAMILY_MAX: base2027.HSA_FAMILY_MAX + yearsAhead * 250,
    FSA_MAX: base2027.FSA_MAX + yearsAhead * 100,
    SOCIAL_SECURITY_WAGE_CAP: Math.round(base2027.SOCIAL_SECURITY_WAGE_CAP * inflationMult / 100) * 100,

    FEDERAL_STANDARD_DEDUCTION: {
      single: Math.round((base2027.FEDERAL_STANDARD_DEDUCTION.single * inflationMult) / 50) * 50,
      married: Math.round((base2027.FEDERAL_STANDARD_DEDUCTION.married * inflationMult) / 50) * 50,
      head_of_household: Math.round((base2027.FEDERAL_STANDARD_DEDUCTION.head_of_household * inflationMult) / 50) * 50,
    },

    FEDERAL_BRACKETS: {
      single: base2027.FEDERAL_BRACKETS.single.map((b) => ({
        ...b,
        min: Math.round((b.min * inflationMult) / 25) * 25,
        max: b.max === Infinity ? Infinity : Math.round((b.max * inflationMult) / 25) * 25,
      })),
      married: base2027.FEDERAL_BRACKETS.married.map((b) => ({
        ...b,
        min: Math.round((b.min * inflationMult) / 25) * 25,
        max: b.max === Infinity ? Infinity : Math.round((b.max * inflationMult) / 25) * 25,
      })),
      head_of_household: base2027.FEDERAL_BRACKETS.head_of_household.map((b) => ({
        ...b,
        min: Math.round((b.min * inflationMult) / 25) * 25,
        max: b.max === Infinity ? Infinity : Math.round((b.max * inflationMult) / 25) * 25,
      })),
    },
  };
}

// Backward compatibility default
export const TAX_LIMITS_2026 = getTaxLimitsForYear(2026);
export const TAX_LIMITS_2027 = getTaxLimitsForYear(2027);

export const US_STATES: Record<string, StateTaxInfo> = {
  CA: {
    code: 'CA',
    name: 'California',
    hasStateTax: true,
    hasSDI: true,
    sdiRate: 0.014, // 1.4% CASDI (2027 CA EDD rate, no wage cap)
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
    notes: 'Includes 1.4% California State Disability Insurance (2027 CASDI, no wage cap).',
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
