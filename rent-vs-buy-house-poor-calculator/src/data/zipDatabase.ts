import redfinRawData from './redfinZipData.json' with { type: 'json' };

export interface RedfinZipRecord {
  city?: string;
  state: string;
  ppsf: number;
  medianPrice: number;
  date: string;
}

const REDFIN_ZIP_DATA = redfinRawData as Record<string, RedfinZipRecord>;

export interface ZipRegionData {
  zip: string;
  city: string;
  state: string;
  county: string;
  propertyTaxRate: number; // percentage, e.g. 1.25 for 1.25%
  avgPricePerSqFt: number; // Redfin Median PPSF
  avgRent3Bed: number;     // Median 3-Bed Rent Estimate
  medianPrice?: number;    // Official Redfin Median Sale Price
  isRedfinData?: boolean;
  marketDate?: string;
}

// Default State Tax Rates fallback
export const STATE_DEFAULT_TAX_RATES: Record<string, { propertyTaxRate: number; name: string }> = {
  CA: { propertyTaxRate: 1.25, name: 'California' },
  TX: { propertyTaxRate: 2.15, name: 'Texas' },
  WA: { propertyTaxRate: 0.98, name: 'Washington' },
  NY: { propertyTaxRate: 1.70, name: 'New York' },
  FL: { propertyTaxRate: 1.05, name: 'Florida' },
  IL: { propertyTaxRate: 2.10, name: 'Illinois' },
  CO: { propertyTaxRate: 0.65, name: 'Colorado' },
  MA: { propertyTaxRate: 1.10, name: 'Massachusetts' },
  NJ: { propertyTaxRate: 2.45, name: 'New Jersey' },
  PA: { propertyTaxRate: 1.55, name: 'Pennsylvania' },
  NC: { propertyTaxRate: 0.85, name: 'North Carolina' },
  GA: { propertyTaxRate: 0.92, name: 'Georgia' },
  OTHER: { propertyTaxRate: 1.20, name: 'United States Average' },
};

export function lookupZipCode(zip: string, fallbackState: string = 'CA'): ZipRegionData {
  const cleanZip = zip.trim();

  // 1. Check official Redfin dataset first (24,500+ US ZIP codes)
  if (cleanZip && REDFIN_ZIP_DATA[cleanZip]) {
    const rf = REDFIN_ZIP_DATA[cleanZip];
    const stateCode = rf.state || fallbackState;
    const stateInfo = STATE_DEFAULT_TAX_RATES[stateCode] || STATE_DEFAULT_TAX_RATES['OTHER'];

    // Estimate 3-bed rent based on local PPSF & market scale
    const estimatedRent = Math.max(1800, Math.round(rf.ppsf * 4.5));

    return {
      zip: cleanZip,
      city: (rf.city && rf.city.length > 0) ? rf.city : `ZIP ${cleanZip}`,
      state: stateCode,
      county: `${stateInfo.name} Region`,
      propertyTaxRate: stateInfo.propertyTaxRate,
      avgPricePerSqFt: rf.ppsf,
      avgRent3Bed: estimatedRent,
      medianPrice: rf.medianPrice,
      isRedfinData: true,
      marketDate: rf.date,
    };
  }

  // 2. Fallback if ZIP is not in Redfin dataset
  const stateInfo = STATE_DEFAULT_TAX_RATES[fallbackState] || STATE_DEFAULT_TAX_RATES['OTHER'];
  return {
    zip: cleanZip || '90001',
    city: 'Target Location',
    state: fallbackState,
    county: `${stateInfo.name} Region`,
    propertyTaxRate: stateInfo.propertyTaxRate,
    avgPricePerSqFt: 450,
    avgRent3Bed: 3200,
    medianPrice: 850000,
    isRedfinData: false,
  };
}
