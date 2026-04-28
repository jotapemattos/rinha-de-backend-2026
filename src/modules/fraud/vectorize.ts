import type { FraudScoreRequest } from "./fraud.types.ts";
import { DIMS } from "./references.ts";

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

// Reused across calls — safe because knn() is synchronous and JS is single-threaded
const _v = new Float32Array(DIMS);

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
  // Days from epoch to year start (Gregorian, UTC)
  const y = year - 1;
  const daysSinceEpoch =
    365 * y +
    Math.floor(y / 4) -
    Math.floor(y / 100) +
    Math.floor(y / 400) -
    719162 + // days from Gregorian year 1 Jan 1 to Unix epoch (1970-01-01)
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

export function vectorize(req: FraudScoreRequest): Float32Array {
  const txMs = parseISOtoMs(req.transaction.requested_at);

  _v[0] = clamp(req.transaction.amount / norm.max_amount);
  _v[1] = clamp(req.transaction.installments / norm.max_installments);
  _v[2] = clamp(
    req.transaction.amount / req.customer.avg_amount / norm.amount_vs_avg_ratio,
  );
  _v[3] = isoUTCHour(req.transaction.requested_at) / 23;
  _v[4] = ((utcDayOfWeek(txMs) + 6) % 7) / 6;

  if (req.last_transaction === null) {
    _v[5] = -1;
    _v[6] = -1;
  } else {
    const minutes =
      (txMs - parseISOtoMs(req.last_transaction.timestamp)) / 60000;
    _v[5] = clamp(minutes / norm.max_minutes);
    _v[6] = clamp(req.last_transaction.km_from_current / norm.max_km);
  }

  _v[7] = clamp(req.terminal.km_from_home / norm.max_km);
  _v[8] = clamp(req.customer.tx_count_24h / norm.max_tx_count_24h);
  _v[9] = req.terminal.is_online ? 1 : 0;
  _v[10] = req.terminal.card_present ? 1 : 0;
  _v[11] = req.customer.known_merchants.includes(req.merchant.id) ? 0 : 1;
  _v[12] = mccRisk[req.merchant.mcc] ?? DEFAULT_MCC_RISK;
  _v[13] = clamp(req.merchant.avg_amount / norm.max_merchant_avg_amount);

  return _v;
}
