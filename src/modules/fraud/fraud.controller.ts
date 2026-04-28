import * as FraudService from "./fraud.service.ts";
import type { FraudScoreRequest } from "./fraud.types.ts";

export async function postFraudScore(req: Request): Promise<Response> {
  const payload = (await req.json()) as FraudScoreRequest;
  const result = FraudService.scoreTransaction(payload);
  return new Response(
    `{"approved":${result.approved},"fraud_score":${result.fraud_score}}`,
    { headers: { "Content-Type": "application/json" } },
  );
}
