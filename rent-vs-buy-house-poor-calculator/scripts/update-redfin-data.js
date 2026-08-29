import https from 'https';
import zlib from 'zlib';
import readline from 'readline';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REDFIN_URL = 'https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz';
const OUTPUT_FILE = path.join(__dirname, '../src/data/redfinZipData.json');

console.log('🚀 Downloading and processing official Redfin Housing Market Tracker dataset...');
console.log(`Source URL: ${REDFIN_URL}`);

const zipMap = {};

https.get(REDFIN_URL, (res) => {
  if (res.statusCode !== 200) {
    console.error(`❌ Failed to fetch Redfin dataset. HTTP Status: ${res.statusCode}`);
    process.exit(1);
  }

  const gunzip = zlib.createGunzip();
  const rl = readline.createInterface({ input: res.pipe(gunzip) });

  let headerMap = {};
  let lineCount = 0;

  rl.on('line', (line) => {
    lineCount++;
    if (lineCount % 1000000 === 0) {
      console.log(`...processed ${(lineCount / 1000000).toFixed(1)}M lines, found ${Object.keys(zipMap).length} ZIP codes so far.`);
    }

    const parts = line.split('\t').map(p => p.replace(/^"|"$/g, '').trim());

    if (lineCount === 1) {
      parts.forEach((colName, index) => {
        headerMap[colName] = index;
      });
      return;
    }

    const propType = parts[headerMap['PROPERTY_TYPE']];
    const region = parts[headerMap['REGION']];
    const periodBegin = parts[headerMap['PERIOD_BEGIN']];

    if (!region || !region.startsWith('Zip Code: ')) return;
    if (propType !== 'All Residential' && propType !== 'Single Family Residential') return;

    const zipCode = region.replace('Zip Code: ', '').trim();
    if (!/^\d{5}$/.test(zipCode)) return;

    const rawPpsf = parseFloat(parts[headerMap['MEDIAN_PPSF']]);
    const rawPrice = parseFloat(parts[headerMap['MEDIAN_SALE_PRICE']]);

    if (isNaN(rawPpsf) || rawPpsf <= 0) return;

    const city = parts[headerMap['CITY']] || '';
    const state = parts[headerMap['STATE_CODE']] || '';

    // If ZIP isn't added yet OR this record has a more recent period_begin date
    if (!zipMap[zipCode] || periodBegin > zipMap[zipCode].date) {
      zipMap[zipCode] = {
        date: periodBegin,
        city,
        state,
        ppsf: Math.round(rawPpsf),
        medianPrice: isNaN(rawPrice) ? 0 : Math.round(rawPrice),
      };
    }
  });

  rl.on('close', () => {
    console.log(`✅ Processing finished! Extracted latest market snapshot for ${Object.keys(zipMap).length} ZIP codes.`);

    const finalData = {};
    for (const [zip, data] of Object.entries(zipMap)) {
      finalData[zip] = {
        city: data.city,
        state: data.state,
        ppsf: data.ppsf,
        medianPrice: data.medianPrice,
        date: data.date,
      };
    }

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(finalData), 'utf8');
    const stats = fs.statSync(OUTPUT_FILE);
    console.log(`🎉 Saved Redfin dataset to: ${OUTPUT_FILE}`);
    console.log(`📦 Output File Size: ${(stats.size / 1024).toFixed(1)} KB`);
  });
}).on('error', (err) => {
  console.error('❌ Error downloading Redfin data:', err.message);
  process.exit(1);
});
