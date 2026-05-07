import { vectorizeFromText } from "./vectorize.ts";
import { ivfFlatScore } from "./ivf-flat.ts";

export function scoreTransactionFromText(text: string): number {
  return ivfFlatScore(vectorizeFromText(text));
}
