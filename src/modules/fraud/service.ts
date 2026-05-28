import { vectorizeFromBuffer } from "./vectorize.ts";
import { ivfFlatScore, ivfFlatWarmupBothPaths } from "./ivf-flat.ts";

export function scoreTransactionFromBuffer(buf: Buffer, start: number, end: number): number {
  return ivfFlatScore(vectorizeFromBuffer(buf, start, end));
}

export function warmupBothPaths(buf: Buffer, start: number, end: number): void {
  ivfFlatWarmupBothPaths(vectorizeFromBuffer(buf, start, end));
}
