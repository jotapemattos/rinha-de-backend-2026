import { vectorizeFromBuffer } from "./vectorize.ts";
import { ivfFlatScore } from "./ivf-flat.ts";

export function scoreTransactionFromBuffer(buf: Buffer, start: number, end: number): number {
  return ivfFlatScore(vectorizeFromBuffer(buf, start, end));
}
