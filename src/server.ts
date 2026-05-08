import * as FraudService from "./modules/fraud/service.ts";
import type { FraudScoreRequest } from "./modules/fraud/types.ts";

const N_BUCKETS = 256;
const THRESHOLD = 0.6;

const RESPONSES: Buffer[] = new Array(N_BUCKETS);
for (let b = 0; b < N_BUCKETS; b++) {
  const score = b / (N_BUCKETS - 1);
  const body = `{"approved":${score < THRESHOLD},"fraud_score":${score.toFixed(4)}}`;
  RESPONSES[b] = Buffer.from(
    `HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\nConnection: keep-alive\r\n\r\n${body}`,
  );
}

const READY_RESP = Buffer.from(
  "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
);

type SocketState = { pending: Buffer | null };

function findHeaderEnd(buf: Buffer, from: number): number {
  for (let i = from, limit = buf.length - 3; i < limit; i++) {
    if (
      buf[i] === 13 &&
      buf[i + 1] === 10 &&
      buf[i + 2] === 13 &&
      buf[i + 3] === 10
    )
      return i;
  }
  return -1;
}

function parseContentLength(buf: Buffer, start: number, hEnd: number): number {
  const limit = hEnd - 17;
  for (let i = start; i < limit; i++) {
    if (
      buf[i] === 0x0d &&
      buf[i + 1] === 0x0a &&
      (buf[i + 2]! | 0x20) === 0x63 && // c
      (buf[i + 3]! | 0x20) === 0x6f && // o
      (buf[i + 4]! | 0x20) === 0x6e && // n
      (buf[i + 5]! | 0x20) === 0x74 && // t
      (buf[i + 6]! | 0x20) === 0x65 && // e
      (buf[i + 7]! | 0x20) === 0x6e && // n
      (buf[i + 8]! | 0x20) === 0x74 && // t
      buf[i + 9] === 0x2d && // -
      (buf[i + 10]! | 0x20) === 0x6c && // l
      (buf[i + 11]! | 0x20) === 0x65 && // e
      (buf[i + 12]! | 0x20) === 0x6e && // n
      (buf[i + 13]! | 0x20) === 0x67 && // g
      (buf[i + 14]! | 0x20) === 0x74 && // t
      (buf[i + 15]! | 0x20) === 0x68 && // h
      buf[i + 16] === 0x3a // :
    ) {
      let j = i + 17;
      while (buf[j] === 0x20 || buf[j] === 0x09) j++; // skip whitespace
      let n = 0,
        c: number;
      while ((c = buf[j]! - 48) >= 0 && c <= 9) {
        n = n * 10 + c;
        j++;
      }
      return n;
    }
  }
  return -1;
}

const udsPath = process.env.UDS_PATH ?? "/tmp/rinha.sock";
process.umask(0);

Bun.listen<SocketState>({
  unix: udsPath,
  socket: {
    open(socket) {
      socket.data = { pending: null };
    },
    data(socket, chunk) {
      const state = socket.data;
      const buf: Buffer = state.pending
        ? Buffer.concat([state.pending, chunk])
        : (chunk as Buffer);
      state.pending = null;

      let offset = 0;
      while (offset < buf.length) {
        const hEnd = findHeaderEnd(buf, offset);
        if (hEnd === -1) {
          state.pending = Buffer.from(buf.subarray(offset));
          break;
        }

        const bodyStart = hEnd + 4;

        if (buf[offset] === 71 /* 'G' — GET /ready */) {
          socket.write(READY_RESP);
          offset = bodyStart;
          continue;
        }

        // POST /fraud-score
        const contentLength = parseContentLength(buf, offset, hEnd);
        if (contentLength < 0) {
          offset = bodyStart;
          continue;
        }

        if (buf.length < bodyStart + contentLength) {
          state.pending = Buffer.from(buf.subarray(offset));
          break;
        }

        const bodyText = buf.toString(
          "utf8",
          bodyStart,
          bodyStart + contentLength,
        );

        const prob = FraudService.scoreTransactionFromText(bodyText);
        let bucket = (prob * (N_BUCKETS - 1)) | 0;
        if (bucket < 0) bucket = 0;
        else if (bucket >= N_BUCKETS) bucket = N_BUCKETS - 1;
        socket.write(RESPONSES[bucket]!);

        offset = bodyStart + contentLength;
      }
    },
    close() {},
    error(_, err) {
      console.error(err);
    },
  },
});

console.log(`Listening on ${udsPath}`);

const _warmup: FraudScoreRequest = {
  id: "warmup",
  transaction: {
    amount: 100,
    installments: 1,
    requested_at: "2025-01-15T14:30:00Z",
  },
  customer: { avg_amount: 100, tx_count_24h: 1, known_merchants: [] },
  merchant: { id: "m1", mcc: "5411", avg_amount: 100 },
  terminal: { km_from_home: 5, is_online: false, card_present: true },
  last_transaction: null,
};
const _warmupText = JSON.stringify(_warmup);
for (let i = 0; i < 50; i++) FraudService.scoreTransactionFromText(_warmupText);
