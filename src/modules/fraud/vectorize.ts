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

const norm: Normalization = {
  max_amount: 10000,
  max_installments: 12,
  amount_vs_avg_ratio: 10,
  max_minutes: 1440,
  max_km: 1000,
  max_tx_count_24h: 20,
  max_merchant_avg_amount: 10000,
};

const DEFAULT_MCC_RISK = 0.5;

// Integer keys match the zero-alloc MCC parsing in the hot path.
const mccRiskByInt = new Map<number, number>([
  [5411, 0.15],
  [5812, 0.30],
  [5912, 0.20],
  [5944, 0.45],
  [7801, 0.80],
  [7802, 0.75],
  [7995, 0.85],
  [4511, 0.35],
  [5311, 0.25],
  [5999, 0.50],
]);

// Pre-computed byte patterns for field seeking.
const B_ID = Buffer.from('"id":');
const B_AMOUNT = Buffer.from('"amount":');
const B_INSTALLMENTS = Buffer.from('"installments":');
const B_REQUESTED_AT = Buffer.from('"requested_at":');
const B_AVG_AMOUNT = Buffer.from('"avg_amount":');
const B_TX_COUNT_24H = Buffer.from('"tx_count_24h":');
const B_KNOWN_MERCHANTS = Buffer.from('"known_merchants":');
const B_MCC = Buffer.from('"mcc":');
const B_IS_ONLINE = Buffer.from('"is_online":');
const B_CARD_PRESENT = Buffer.from('"card_present":');
const B_KM_FROM_HOME = Buffer.from('"km_from_home":');
const B_LAST_TRANSACTION = Buffer.from('"last_transaction":');
const B_TIMESTAMP = Buffer.from('"timestamp":');
const B_KM_FROM_CURRENT = Buffer.from('"km_from_current":');

// Module-level state — safe because JS is single-threaded and all calls are synchronous.
// Avoids allocating new closure objects on every request.
let _buf: Buffer = Buffer.alloc(0);
let _pos = 0;

function seek(pattern: Buffer): void {
  _pos = _buf.indexOf(pattern, _pos) + pattern.length;
  while (_buf[_pos] === 32 || _buf[_pos] === 9 || _buf[_pos] === 10 || _buf[_pos] === 13) _pos++;
}

function num(): number {
  let i = _pos;
  const neg = _buf[i] === 45;
  if (neg) i++;
  let int = 0;
  let c: number;
  while ((c = _buf[i]! - 48) >= 0 && c <= 9) { int = int * 10 + c; i++; }
  if (_buf[i] !== 46) { _pos = i; return neg ? -int : int; }
  i++;
  let frac = 0, div = 1;
  while ((c = _buf[i]! - 48) >= 0 && c <= 9) { frac = frac * 10 + c; div *= 10; i++; }
  _pos = i;
  return neg ? -(int + frac / div) : int + frac / div;
}

// Reused across calls — safe because search is synchronous and JS is single-threaded.
const _vF32 = new Float32Array(DIMS);

const MONTHS_OFFSET = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

function clamp(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function parseISOtoMsFromBuf(buf: Buffer, p: number): number {
  const year =
    (buf[p]! - 48) * 1000 +
    (buf[p + 1]! - 48) * 100 +
    (buf[p + 2]! - 48) * 10 +
    (buf[p + 3]! - 48);
  const month = (buf[p + 5]! - 48) * 10 + (buf[p + 6]! - 48);
  const day = (buf[p + 8]! - 48) * 10 + (buf[p + 9]! - 48);
  const hour = (buf[p + 11]! - 48) * 10 + (buf[p + 12]! - 48);
  const min = (buf[p + 14]! - 48) * 10 + (buf[p + 15]! - 48);
  const sec = (buf[p + 17]! - 48) * 10 + (buf[p + 18]! - 48);
  const y = year - 1;
  const daysSinceEpoch =
    365 * y +
    Math.floor(y / 4) -
    Math.floor(y / 100) +
    Math.floor(y / 400) -
    719162 +
    MONTHS_OFFSET[month - 1]! +
    (month > 2 && ((year % 4 === 0 && year % 100 !== 0) || year % 400 === 0) ? 1 : 0) +
    day - 1;
  return (daysSinceEpoch * 86400 + hour * 3600 + min * 60 + sec) * 1000;
}

function utcDayOfWeek(ms: number): number {
  return ((Math.floor(ms / 86400000) % 7) + 4 + 7) % 7;
}

// Zero-alloc hot path: operates directly on the raw request buffer.
// No string copies, no closure allocations — uses module-level _buf/_pos state.
export function vectorizeFromBuffer(buf: Buffer, start: number, end: number): Float32Array {
  _buf = buf;
  _pos = start;

  seek(B_ID);
  _pos++; // skip opening "
  while (_buf[_pos] !== 34) _pos++;
  _pos++;

  seek(B_AMOUNT);
  const txAmount = num();
  seek(B_INSTALLMENTS);
  const installments = num();
  seek(B_REQUESTED_AT);
  _pos++; // skip opening "
  const reqAtPos = _pos;
  _pos += 21; // 20 chars + closing "

  seek(B_AVG_AMOUNT);
  const custAvgAmount = num();
  seek(B_TX_COUNT_24H);
  const txCount24h = num();

  seek(B_KNOWN_MERCHANTS);
  const kmStart = _pos;
  const kmEnd = buf.indexOf(93, _pos + 1); // 93 = ']'
  _pos = kmEnd + 1;

  seek(B_ID);
  _pos++; // skip opening "
  const mercIdStart = _pos;
  while (_buf[_pos] !== 34) _pos++;
  const mercIdEnd = _pos;
  _pos++;

  // Byte-level merchant search within known_merchants — no string allocation.
  let unknownMerchant = 1;
  const idLen = mercIdEnd - mercIdStart;
  outer: for (let i = kmStart + 1; i < kmEnd - idLen - 1; i++) {
    if (buf[i] === 34) {
      for (let j = 0; j < idLen; j++) {
        if (buf[i + 1 + j] !== buf[mercIdStart + j]) continue outer;
      }
      if (buf[i + 1 + idLen] === 34) { unknownMerchant = 0; break; }
    }
  }

  seek(B_MCC);
  _pos++; // skip opening "
  let mccInt = 0;
  while (_buf[_pos] !== 34) { mccInt = mccInt * 10 + (_buf[_pos]! - 48); _pos++; }
  _pos++; // skip closing "

  seek(B_AVG_AMOUNT);
  const mercAvgAmount = num();

  seek(B_IS_ONLINE);
  const isOnline = _buf[_pos] === 116 ? 1 : 0; // 't'

  seek(B_CARD_PRESENT);
  const cardPresent = _buf[_pos] === 116 ? 1 : 0;

  seek(B_KM_FROM_HOME);
  const kmFromHome = num();

  seek(B_LAST_TRANSACTION);
  const txMs = parseISOtoMsFromBuf(buf, reqAtPos);

  if (_buf[_pos] !== 110) { // 'n' for null
    seek(B_TIMESTAMP);
    _pos++; // skip opening "
    const lastTsPos = _pos;
    _pos += 21;
    const minutes = (txMs - parseISOtoMsFromBuf(buf, lastTsPos)) / 60000;
    _vF32[5] = clamp(minutes / norm.max_minutes);
    seek(B_KM_FROM_CURRENT);
    _vF32[6] = clamp(num() / norm.max_km);
  } else {
    _vF32[5] = -1;
    _vF32[6] = -1;
  }

  _vF32[0] = clamp(txAmount / norm.max_amount);
  _vF32[1] = clamp(installments / norm.max_installments);
  _vF32[2] = clamp(txAmount / custAvgAmount / norm.amount_vs_avg_ratio);
  _vF32[3] = (buf[reqAtPos + 11]! - 48) * 10 / 23 + (buf[reqAtPos + 12]! - 48) / 23;
  _vF32[4] = ((utcDayOfWeek(txMs) + 6) % 7) / 6;
  _vF32[7] = clamp(kmFromHome / norm.max_km);
  _vF32[8] = clamp(txCount24h / norm.max_tx_count_24h);
  _vF32[9] = isOnline;
  _vF32[10] = cardPresent;
  _vF32[11] = unknownMerchant;
  _vF32[12] = mccRiskByInt.get(mccInt) ?? DEFAULT_MCC_RISK;
  _vF32[13] = clamp(mercAvgAmount / norm.max_merchant_avg_amount);

  return _vF32;
}
