import { run, bench, group, do_not_optimize } from "mitata";
import { vectorizeFromBuffer } from "./src/modules/fraud/vectorize.ts";
import { ivfFlatScore, getPathStats, resetPathStats } from "./src/modules/fraud/ivf-flat.ts";
import { scoreTransactionFromBuffer, warmupBothPaths } from "./src/modules/fraud/service.ts";

// Three representative shapes:
// LEGIT   — small known-merchant grocery, card present, close to home
// FRAUD   — max-risk: gambling MCC, unknown merchant, far from home, online, burst tx count
// BORDER  — borderline: moderate amount, slightly elevated features, has last_transaction
const LEGIT = Buffer.from(
  '{"id":"bench-legit","transaction":{"amount":45,"installments":1,"requested_at":"2025-01-15T14:30:00Z"},"customer":{"avg_amount":50,"tx_count_24h":2,"known_merchants":["m-grocery-01"]},"merchant":{"id":"m-grocery-01","mcc":"5411","avg_amount":45},"terminal":{"km_from_home":1,"is_online":false,"card_present":true},"last_transaction":null}',
);
const FRAUD = Buffer.from(
  '{"id":"bench-fraud","transaction":{"amount":9800,"installments":1,"requested_at":"2025-03-20T03:15:00Z"},"customer":{"avg_amount":200,"tx_count_24h":19,"known_merchants":[]},"merchant":{"id":"casino-x","mcc":"7995","avg_amount":9000},"terminal":{"km_from_home":980,"is_online":true,"card_present":false},"last_transaction":null}',
);
const BORDER = Buffer.from(
  '{"id":"bench-border","transaction":{"amount":600,"installments":3,"requested_at":"2025-06-10T14:00:00Z"},"customer":{"avg_amount":300,"tx_count_24h":6,"known_merchants":["m-rest-02"]},"merchant":{"id":"m-rest-99","mcc":"5812","avg_amount":450},"terminal":{"km_from_home":55,"is_online":true,"card_present":false},"last_transaction":{"timestamp":"2025-06-10T13:45:00Z","km_from_current":30}}',
);

// Warmup — same budget as server.ts so JIT tiers are fully committed before timing
const _wBuf = Buffer.from(
  '{"id":"w","transaction":{"amount":100,"installments":1,"requested_at":"2025-01-15T14:30:00Z"},"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":[]},"merchant":{"id":"m1","mcc":"5411","avg_amount":100},"terminal":{"km_from_home":5,"is_online":false,"card_present":true},"last_transaction":null}',
);
for (let i = 0; i < 50_000; i++) warmupBothPaths(_wBuf, 0, _wBuf.length);
Bun.gc(true);

// --- path-stat snapshot before timing ---
resetPathStats();
for (let i = 0; i < 10_000; i++) {
  scoreTransactionFromBuffer(LEGIT, 0, LEGIT.length);
  scoreTransactionFromBuffer(FRAUD, 0, FRAUD.length);
  scoreTransactionFromBuffer(BORDER, 0, BORDER.length);
}
const stats = getPathStats();
const total = stats.fastPathCount + stats.fullPathCount;
console.log(
  `\nPath split over 30k calls (10k × 3 payloads):` +
  `\n  fast  ${stats.fastPathCount.toLocaleString()} / ${total.toLocaleString()} (${((stats.fastPathCount / total) * 100).toFixed(1)}%)` +
  `\n  full  ${stats.fullPathCount.toLocaleString()} / ${total.toLocaleString()} (${((stats.fullPathCount / total) * 100).toFixed(1)}%)\n`,
);
resetPathStats();

group("phase breakdown (legit payload)", () => {
  bench("vectorize", () => do_not_optimize(vectorizeFromBuffer(LEGIT, 0, LEGIT.length)));
  bench("ivfFlatScore", () => do_not_optimize(ivfFlatScore(vectorizeFromBuffer(LEGIT, 0, LEGIT.length))));
  bench("end-to-end", () => do_not_optimize(scoreTransactionFromBuffer(LEGIT, 0, LEGIT.length)));
});

group("fast vs full path", () => {
  bench("legit  (→ fast path expected)", () => do_not_optimize(scoreTransactionFromBuffer(LEGIT, 0, LEGIT.length)));
  bench("fraud  (→ fast path expected)", () => do_not_optimize(scoreTransactionFromBuffer(FRAUD, 0, FRAUD.length)));
  bench("border (→ full path expected)", () => do_not_optimize(scoreTransactionFromBuffer(BORDER, 0, BORDER.length)));
});

await run({ avg: true, min_max: true, percentiles: true });
