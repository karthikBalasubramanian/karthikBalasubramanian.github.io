export interface ZipRegionData {
  zip: string;
  city: string;
  state: string;
  county: string;
  propertyTaxRate: number; // percentage, e.g. 1.25 for 1.25%
  avgPricePerSqFt: number;
  avgRent3Bed: number;
}

export const ZIP_DATABASE: Record<string, ZipRegionData> = {
  // California
  '95113': { zip: '95113', city: 'San Jose', state: 'CA', county: 'Santa Clara', propertyTaxRate: 1.25, avgPricePerSqFt: 780, avgRent3Bed: 3800 },
  '94102': { zip: '94102', city: 'San Francisco', state: 'CA', county: 'San Francisco', propertyTaxRate: 1.18, avgPricePerSqFt: 950, avgRent3Bed: 4500 },
  '90210': { zip: '90210', city: 'Beverly Hills', state: 'CA', county: 'Los Angeles', propertyTaxRate: 1.22, avgPricePerSqFt: 1400, avgRent3Bed: 7500 },
  '92618': { zip: '92618', city: 'Irvine', state: 'CA', county: 'Orange', propertyTaxRate: 1.35, avgPricePerSqFt: 690, avgRent3Bed: 4100 },
  '95630': { zip: '95630', city: 'Folsom', state: 'CA', county: 'Sacramento', propertyTaxRate: 1.15, avgPricePerSqFt: 410, avgRent3Bed: 2900 },

  // Texas
  '75001': { zip: '75001', city: 'Addison', state: 'TX', county: 'Dallas', propertyTaxRate: 2.15, avgPricePerSqFt: 280, avgRent3Bed: 2700 },
  '78701': { zip: '78701', city: 'Austin', state: 'TX', county: 'Travis', propertyTaxRate: 2.05, avgPricePerSqFt: 550, avgRent3Bed: 3500 },
  '77002': { zip: '77002', city: 'Houston', state: 'TX', county: 'Harris', propertyTaxRate: 2.25, avgPricePerSqFt: 260, avgRent3Bed: 2600 },

  // Washington
  '98101': { zip: '98101', city: 'Seattle', state: 'WA', county: 'King', propertyTaxRate: 0.98, avgPricePerSqFt: 620, avgRent3Bed: 3600 },
  '98004': { zip: '98004', city: 'Bellevue', state: 'WA', county: 'King', propertyTaxRate: 0.95, avgPricePerSqFt: 850, avgRent3Bed: 4200 },

  // New York
  '10001': { zip: '10001', city: 'New York', state: 'NY', county: 'New York', propertyTaxRate: 1.72, avgPricePerSqFt: 1350, avgRent3Bed: 5800 },
  '11201': { zip: '11201', city: 'Brooklyn', state: 'NY', county: 'Kings', propertyTaxRate: 1.55, avgPricePerSqFt: 920, avgRent3Bed: 4600 },

  // Florida
  '33131': { zip: '33131', city: 'Miami', state: 'FL', county: 'Miami-Dade', propertyTaxRate: 1.10, avgPricePerSqFt: 680, avgRent3Bed: 4400 },
  '32801': { zip: '32801', city: 'Orlando', state: 'FL', county: 'Orange', propertyTaxRate: 0.98, avgPricePerSqFt: 310, avgRent3Bed: 2600 },

  // Illinois
  '60601': { zip: '60601', city: 'Chicago', state: 'IL', county: 'Cook', propertyTaxRate: 2.10, avgPricePerSqFt: 380, avgRent3Bed: 3200 },

  // Colorado
  '80202': { zip: '80202', city: 'Denver', state: 'CO', county: 'Denver', propertyTaxRate: 0.65, avgPricePerSqFt: 490, avgRent3Bed: 3300 },

  // Massachusetts
  '02108': { zip: '02108', city: 'Boston', state: 'MA', county: 'Suffolk', propertyTaxRate: 1.08, avgPricePerSqFt: 980, avgRent3Bed: 4800 },
};

// Default State Tax Rates fallback if ZIP is not in database
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
  if (ZIP_DATABASE[cleanZip]) {
    return ZIP_DATABASE[cleanZip];
  }

  const stateInfo = STATE_DEFAULT_TAX_RATES[fallbackState] || STATE_DEFAULT_TAX_RATES['OTHER'];
  return {
    zip: cleanZip || '90001',
    city: 'Target Location',
    state: fallbackState,
    county: `${stateInfo.name} Region`,
    propertyTaxRate: stateInfo.propertyTaxRate,
    avgPricePerSqFt: 450,
    avgRent3Bed: 3200,
  };
}
