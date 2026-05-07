FROM oven/bun:1-alpine AS builder

WORKDIR /app

COPY package.json ./
COPY scripts ./scripts
COPY resources/references.json.gz resources/

RUN bun scripts/preprocess.ts
RUN bun scripts/build-ivf-flat.ts

# ---

FROM oven/bun:1-alpine

WORKDIR /app

COPY package.json ./
COPY src ./src
COPY resources/mcc_risk.json resources/normalization.json resources/
COPY --from=builder /app/resources/ivf-flat.bin resources/

CMD ["bun", "run", "src/server.ts"]
