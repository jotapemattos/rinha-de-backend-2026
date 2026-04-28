import { postFraudScore } from "./modules/fraud/fraud.controller.ts";
import { getReady } from "./modules/health/health.controller.ts";

const port = Number(3000);

const server = Bun.serve({
  port,
  routes: {
    "/ready": { GET: getReady },
    "/fraud-score": { POST: postFraudScore },
  },
});

console.log(`Listening on ${server.url}`);
