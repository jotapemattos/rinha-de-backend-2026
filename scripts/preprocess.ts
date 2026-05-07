const DIMS = 14;
const INPUT = "resources/references.json.gz";
const OUTPUT = "resources/references.bin";

interface RawRef {
  vector: number[];
  label: "fraud" | "legit";
}

const file = Bun.file(INPUT);
if (!(await file.exists())) throw new Error(`Input not found: ${INPUT}`);

console.log("Decompressing...");
const json = new TextDecoder().decode(Bun.gunzipSync(await file.bytes()));

console.log("Parsing...");
const raw: RawRef[] = JSON.parse(json);
const N = raw.length;

console.log(`Building binary for ${N} references...`);
// Format: [u32 N][f32 vecs: N×DIMS][u8 labels: N]
const buf = new ArrayBuffer(4 + N * DIMS * 4 + N);
new DataView(buf).setUint32(0, N, true);
const vecs = new Float32Array(buf, 4, N * DIMS);
const labs = new Uint8Array(buf, 4 + N * DIMS * 4, N);

for (let i = 0; i < N; i++) {
  const ref = raw[i]!;
  const base = i * DIMS;
  for (let d = 0; d < DIMS; d++) vecs[base + d] = ref.vector[d]!;
  labs[i] = ref.label === "fraud" ? 1 : 0;
}

await Bun.write(OUTPUT, buf);
console.log(`Done: ${(buf.byteLength / 1024 / 1024).toFixed(1)} MB written to ${OUTPUT}`);
