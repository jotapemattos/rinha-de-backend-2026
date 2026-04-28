FROM oven/bun:1-alpine

WORKDIR /app

COPY package.json ./
COPY src ./src
COPY resources ./resources

USER bun

CMD ["bun", "run", "src/server.ts"]
