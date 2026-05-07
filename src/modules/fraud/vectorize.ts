const DIMS = 14;

interface Normalization {
  max_amount: number;
  max_installments: number;
  amount_vs_avg_ratio: number;
  max_minutes: number;
  max_km: number;
  max_tx_count_24h: number;
  max_merchant_avg_amount: number;
}

const normFile = Bun.file("resources/normalization.json");
const mccFile = Bun.file("resources/mcc_risk.json");

if (!(await normFile.exists())) throw new Error(`normalization file not found`);
if (!(await mccFile.exists())) throw new Error(`mcc risk file not found`);

const norm: Normalization = await normFile.json();
const mccRisk: Record<string, number> = await mccFile.json();

const DEFAULT_MCC_RISK = 0.5;

// Reused across calls — safe because search is synchronous and JS is single-threaded.
const _vF32 = new Float32Array(DIMS);

function clamp(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function parseISOtoMs(iso: string): number {
  const year =
    (iso.charCodeAt(0) - 48) * 1000 +
    (iso.charCodeAt(1) - 48) * 100 +
    (iso.charCodeAt(2) - 48) * 10 +
    (iso.charCodeAt(3) - 48);
  const month = (iso.charCodeAt(5) - 48) * 10 + (iso.charCodeAt(6) - 48);
  const day = (iso.charCodeAt(8) - 48) * 10 + (iso.charCodeAt(9) - 48);
  const hour = (iso.charCodeAt(11) - 48) * 10 + (iso.charCodeAt(12) - 48);
  const min = (iso.charCodeAt(14) - 48) * 10 + (iso.charCodeAt(15) - 48);
  const sec = (iso.charCodeAt(17) - 48) * 10 + (iso.charCodeAt(18) - 48);
  const y = year - 1;
  const daysSinceEpoch =
    365 * y +
    Math.floor(y / 4) -
    Math.floor(y / 100) +
    Math.floor(y / 400) -
    719162 +
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][month - 1]! +
    (month > 2 && ((year % 4 === 0 && year % 100 !== 0) || year % 400 === 0)
      ? 1
      : 0) +
    day -
    1;
  return (daysSinceEpoch * 86400 + hour * 3600 + min * 60 + sec) * 1000;
}

function isoUTCHour(iso: string): number {
  return (iso.charCodeAt(11) - 48) * 10 + (iso.charCodeAt(12) - 48);
}

function utcDayOfWeek(ms: number): number {
  return ((Math.floor(ms / 86400000) % 7) + 4 + 7) % 7;
}

// Parses the raw JSON body and builds the Float32Array directly — no intermediate
// object allocation. Relies on the fixed field order of the challenge payload.
export function vectorizeFromText(text: string): Float32Array {
  let pos = 0;

  function seek(key: string): void {
    const k = '"' + key + '":';
    pos = text.indexOf(k, pos) + k.length;
    let c: number;
    while ((c = text.charCodeAt(pos)) === 32 || c === 9 || c === 10 || c === 13)
      pos++;
  }

  function num(): number {
    let i = pos;
    const neg = text.charCodeAt(i) === 45; // '-'
    if (neg) i++;
    let int = 0;
    let c: number;
    while ((c = text.charCodeAt(i) - 48) >= 0 && c <= 9) {
      int = int * 10 + c;
      i++;
    }
    if (text.charCodeAt(i) !== 46) {
      // no decimal point
      pos = i;
      return neg ? -int : int;
    }
    i++;
    let frac = 0,
      div = 1;
    while ((c = text.charCodeAt(i) - 48) >= 0 && c <= 9) {
      frac = frac * 10 + c;
      div *= 10;
      i++;
    }
    pos = i;
    return neg ? -(int + frac / div) : int + frac / div;
  }

  seek("id");
  pos++;
  while (text.charCodeAt(pos) !== 34) pos++;
  pos++;

  seek("amount");
  const txAmount = num();
  seek("installments");
  const installments = num();
  seek("requested_at");
  pos++;
  const reqAt = text.slice(pos, pos + 20);
  pos += 21;

  seek("avg_amount");
  const custAvgAmount = num();
  seek("tx_count_24h");
  const txCount24h = num();

  seek("known_merchants");
  const kmStart = pos;
  const kmEnd = text.indexOf("]", pos + 1);
  pos = kmEnd + 1;

  seek("id");
  pos++;
  const mercIdStart = pos;
  while (text.charCodeAt(pos) !== 34) pos++;
  const mercId = text.slice(mercIdStart, pos);
  pos++;

  const mercIdPat = '"' + mercId + '"';
  const mercIdAt = text.indexOf(mercIdPat, kmStart + 1);
  const unknownMerchant = mercIdAt === -1 || mercIdAt >= kmEnd ? 1 : 0;

  seek("mcc");
  pos++;
  const mccStart = pos;
  while (text.charCodeAt(pos) !== 34) pos++;
  const mcc = text.slice(mccStart, pos);
  pos++;

  seek("avg_amount");
  const mercAvgAmount = num();

  seek("is_online");
  const isOnline = text.charCodeAt(pos) === 116 ? 1 : 0;

  seek("card_present");
  const cardPresent = text.charCodeAt(pos) === 116 ? 1 : 0;

  seek("km_from_home");
  const kmFromHome = num();

  seek("last_transaction");
  const txMs = parseISOtoMs(reqAt);

  if (text.charCodeAt(pos) !== 110) {
    seek("timestamp");
    pos++; // opening "
    const lastTs = text.slice(pos, pos + 20);
    pos += 21;
    const minutes = (txMs - parseISOtoMs(lastTs)) / 60000;
    _vF32[5] = clamp(minutes / norm.max_minutes);
    seek("km_from_current");
    _vF32[6] = clamp(num() / norm.max_km);
  } else {
    _vF32[5] = -1;
    _vF32[6] = -1;
  }

  _vF32[0] = clamp(txAmount / norm.max_amount);
  _vF32[1] = clamp(installments / norm.max_installments);
  _vF32[2] = clamp(txAmount / custAvgAmount / norm.amount_vs_avg_ratio);
  _vF32[3] = isoUTCHour(reqAt) / 23;
  _vF32[4] = ((utcDayOfWeek(txMs) + 6) % 7) / 6;
  _vF32[7] = clamp(kmFromHome / norm.max_km);
  _vF32[8] = clamp(txCount24h / norm.max_tx_count_24h);
  _vF32[9] = isOnline;
  _vF32[10] = cardPresent;
  _vF32[11] = unknownMerchant;
  _vF32[12] = mccRisk[mcc] ?? DEFAULT_MCC_RISK;
  _vF32[13] = clamp(mercAvgAmount / norm.max_merchant_avg_amount);

  return _vF32;
}
