export const DIMS = 14;

interface RawRef {
  vector: number[];
  label: "fraud" | "legit";
}

const REFERENCES_PATH =
  process.env.REFERENCES_PATH ?? "resources/references.json.gz";

const file = Bun.file(REFERENCES_PATH);
if (!(await file.exists())) {
  throw new Error(`references file not found at "${REFERENCES_PATH}"`);
}

const json = new TextDecoder().decode(Bun.gunzipSync(await file.bytes()));
const raw: RawRef[] = JSON.parse(json);

export const N = raw.length;
export const vectors = new Float32Array(N * DIMS);
export const labels = new Uint8Array(N);

for (let i = 0; i < N; i++) {
  const v = raw[i]!.vector;
  for (let d = 0; d < DIMS; d++) vectors[i * DIMS + d] = v[d]!;
  labels[i] = raw[i]!.label === "fraud" ? 1 : 0;
}
