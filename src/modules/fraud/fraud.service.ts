import type { FraudScoreRequest, FraudScoreResponse } from "./fraud.types.ts";
import { knn } from "./knn.ts";
import { vectorize } from "./vectorize.ts";

const THRESHOLD = 0.6;

export function scoreTransaction(
  payload: FraudScoreRequest,
): FraudScoreResponse {
  const fraudScore = knn(vectorize(payload));
  return { approved: fraudScore < THRESHOLD, fraud_score: fraudScore };
}
