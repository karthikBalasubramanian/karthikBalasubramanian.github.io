export interface StateRsuTaxInfo {
  code: string;
  name: string;
  stateRatePct: number; // Supplemental state tax withholding rate for RSUs
  totalWithholdingPct: number; // 22% Fed + 7.65% FICA + State Rate
  notes: string;
}

export const US_STATE_RSU_TAXES: Record<string, StateRsuTaxInfo> = {
  CA: {
    code: 'CA',
    name: 'California',
    stateRatePct: 10.23,
    totalWithholdingPct: 40,
    notes: '22% Federal + 7.65% FICA + 10.23% CA Supplemental Tax = ~40% Withholding',
  },
  NY: {
    code: 'NY',
    name: 'New York',
    stateRatePct: 9.65,
    totalWithholdingPct: 39,
    notes: '22% Federal + 7.65% FICA + 9.65% NY Supplemental Tax = ~39% Withholding',
  },
  WA: {
    code: 'WA',
    name: 'Washington',
    stateRatePct: 0.0,
    totalWithholdingPct: 30,
    notes: '22% Federal + 7.65% FICA + 0% State Income Tax = ~30% Withholding',
  },
  TX: {
    code: 'TX',
    name: 'Texas',
    stateRatePct: 0.0,
    totalWithholdingPct: 30,
    notes: '22% Federal + 7.65% FICA + 0% State Income Tax = ~30% Withholding',
  },
  FL: {
    code: 'FL',
    name: 'Florida',
    stateRatePct: 0.0,
    totalWithholdingPct: 30,
    notes: '22% Federal + 7.65% FICA + 0% State Income Tax = ~30% Withholding',
  },
  MA: {
    code: 'MA',
    name: 'Massachusetts',
    stateRatePct: 5.0,
    totalWithholdingPct: 35,
    notes: '22% Federal + 7.65% FICA + 5.0% MA Flat Rate = ~35% Withholding',
  },
  NJ: {
    code: 'NJ',
    name: 'New Jersey',
    stateRatePct: 6.63,
    totalWithholdingPct: 36,
    notes: '22% Federal + 7.65% FICA + 6.63% NJ Supplemental Rate = ~36% Withholding',
  },
  PA: {
    code: 'PA',
    name: 'Pennsylvania',
    stateRatePct: 3.07,
    totalWithholdingPct: 33,
    notes: '22% Federal + 7.65% FICA + 3.07% PA Flat Rate = ~33% Withholding',
  },
  IL: {
    code: 'IL',
    name: 'Illinois',
    stateRatePct: 4.95,
    totalWithholdingPct: 35,
    notes: '22% Federal + 7.65% FICA + 4.95% IL Flat Rate = ~35% Withholding',
  },
  CO: {
    code: 'CO',
    name: 'Colorado',
    stateRatePct: 4.4,
    totalWithholdingPct: 34,
    notes: '22% Federal + 7.65% FICA + 4.4% CO Flat Rate = ~34% Withholding',
  },
  GA: {
    code: 'GA',
    name: 'Georgia',
    stateRatePct: 5.49,
    totalWithholdingPct: 35,
    notes: '22% Federal + 7.65% FICA + 5.49% GA Rate = ~35% Withholding',
  },
  NC: {
    code: 'NC',
    name: 'North Carolina',
    stateRatePct: 4.5,
    totalWithholdingPct: 34,
    notes: '22% Federal + 7.65% FICA + 4.5% NC Flat Rate = ~34% Withholding',
  },
  VA: {
    code: 'VA',
    name: 'Virginia',
    stateRatePct: 5.75,
    totalWithholdingPct: 35,
    notes: '22% Federal + 7.65% FICA + 5.75% VA Top Rate = ~35% Withholding',
  },
};
