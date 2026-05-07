const DIMS = 14;
const NLIST = 2000;
const SAMPLE_COARSE = 100_000;
const ITER_COARSE = 20;
const M = 7;
const K_PQ = 256;
const SUB = DIMS / M;
const DEFAULT_NPROBE = 32;
const KNN = 50;
const K_FINAL = 5;
const SAMPLE_PQ = 200_000;
const ITER_PQ = 25;
const COST = 3;

const INPUT = "resources/references.bin";
const OUTPUT = "resources/ivf-flat.bin";

console.time("load");
const rawBuf = await Bun.file(INPUT).arrayBuffer();
const N = new DataView(rawBuf).getUint32(0, true);
const vecsF32 = new Float32Array(rawBuf, 4, N * DIMS);
const labels = new Uint8Array(rawBuf, 4 + N * DIMS * 4, N);
console.timeEnd("load");
console.log(
  `N=${N.toLocaleString()}  fraud=${[...labels].filter(Boolean).length.toLocaleString()}`,
);

function kmeans(
  data: Float32Array,
  nPts: number,
  dims: number,
  K: number,
  maxIter: number,
): Float32Array {
  const centroids = new Float32Array(K * dims);

  let first = Math.floor(Math.random() * nPts);
  for (let d = 0; d < dims; d++) {
    centroids[d] = data[first * dims + d]!;
  }

  const minDist = new Float64Array(nPts);

  for (let i = 0; i < nPts; i++) {
    let d2 = 0;
    const base = i * dims;
    for (let d = 0; d < dims; d++) {
      const diff = data[base + d]! - centroids[d]!;
      d2 += diff * diff;
    }
    minDist[i] = d2;
  }

  for (let c = 1; c < K; c++) {
    let total = 0;
    for (let i = 0; i < nPts; i++) total += minDist[i]!;

    let r = Math.random() * total;
    let idx = 0;
    for (; idx < nPts; idx++) {
      r -= minDist[idx]!;
      if (r <= 0) break;
    }

    const cBase = c * dims;
    const iBase = idx * dims;
    for (let d = 0; d < dims; d++) {
      centroids[cBase + d] = data[iBase + d]!;
    }

    for (let i = 0; i < nPts; i++) {
      let d2 = 0;
      const base = i * dims;
      for (let d = 0; d < dims; d++) {
        const diff = data[base + d]! - centroids[cBase + d]!;
        d2 += diff * diff;
      }
      if (d2 < minDist[i]!) {
        minDist[i] = d2;
      }
    }
  }

  const assignments = new Int32Array(nPts).fill(-1);
  const sums = new Float64Array(K * dims);
  const counts = new Uint32Array(K);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = 0;
    for (let i = 0; i < nPts; i++) {
      let minD = Infinity;
      let minC = 0;
      const iBase = i * dims;
      for (let c = 0; c < K; c++) {
        let d2 = 0;
        const cBase = c * dims;
        for (let d = 0; d < dims; d++) {
          const diff = data[iBase + d]! - centroids[cBase + d]!;
          d2 += diff * diff;
        }
        if (d2 < minD) {
          minD = d2;
          minC = c;
        }
      }
      if (assignments[i] !== minC) {
        assignments[i] = minC;
        changed++;
      }
    }

    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < nPts; i++) {
      const c = assignments[i]!;
      const iBase = i * dims,
        cBase = c * dims;
      for (let d = 0; d < dims; d++) sums[cBase + d]! += data[iBase + d]!;
      counts[c]!++;
    }
    for (let c = 0; c < K; c++) {
      if (counts[c] === 0) continue;
      const inv = 1 / counts[c]!;
      const cBase = c * dims;
      for (let d = 0; d < dims; d++)
        centroids[cBase + d] = sums[cBase + d]! * inv;
    }

    process.stdout.write(
      `\r  iter ${iter + 1}/${maxIter}  changed=${changed.toLocaleString()}   `,
    );
    if (changed === 0) break;
  }
  process.stdout.write("\n");
  return centroids;
}

console.log(
  `\nTraining coarse quantizer (K=${NLIST}) on ${SAMPLE_COARSE.toLocaleString()} samples...`,
);
const t0 = Bun.nanoseconds();

const coarseSample = new Float32Array(SAMPLE_COARSE * DIMS);
const step = Math.floor(N / SAMPLE_COARSE);
for (let i = 0; i < SAMPLE_COARSE; i++) {
  const src = i * step;
  for (let d = 0; d < DIMS; d++)
    coarseSample[i * DIMS + d] = vecsF32[src * DIMS + d]!;
}

const coarseCentroids = kmeans(
  coarseSample,
  SAMPLE_COARSE,
  DIMS,
  NLIST,
  ITER_COARSE,
);
console.log(`  done in ${((Bun.nanoseconds() - t0) / 1e9).toFixed(1)}s`);

console.log(
  `\nAssigning ${N.toLocaleString()} vectors to ${NLIST} clusters...`,
);
const t1 = Bun.nanoseconds();

const assignments = new Uint32Array(N);
const clusterSizes = new Uint32Array(NLIST);
const tmpVec = new Float32Array(DIMS);

for (let i = 0; i < N; i++) {
  const iBase = i * DIMS;
  for (let d = 0; d < DIMS; d++) tmpVec[d] = vecsF32[iBase + d]!;

  let minD = Infinity,
    minC = 0;
  for (let c = 0; c < NLIST; c++) {
    let d2 = 0;
    const cBase = c * DIMS;
    for (let d = 0; d < DIMS; d++) {
      const diff = tmpVec[d]! - coarseCentroids[cBase + d]!;
      d2 += diff * diff;
    }
    if (d2 < minD) {
      minD = d2;
      minC = c;
    }
  }
  assignments[i] = minC;
  clusterSizes[minC]!++;

  if (i % 200_000 === 0)
    process.stdout.write(`\r  ${((i / N) * 100).toFixed(0)}%...`);
}
console.log(`\r  done in ${((Bun.nanoseconds() - t1) / 1e9).toFixed(1)}s`);

console.log(
  `\nTraining PQ (M=${M}, K=${K_PQ}) on ${SAMPLE_PQ.toLocaleString()} residuals per sub-space...`,
);
const t2 = Bun.nanoseconds();

const pqCodebooks = new Float32Array(M * K_PQ * SUB);
const subSample = new Float32Array(SAMPLE_PQ * SUB);
const pqStep = Math.floor(N / SAMPLE_PQ);

for (let m = 0; m < M; m++) {
  const mOff = m * SUB;
  for (let s = 0; s < SAMPLE_PQ; s++) {
    const i = s * pqStep;
    const c = assignments[i]!;
    for (let ss = 0; ss < SUB; ss++) {
      subSample[s * SUB + ss] =
        vecsF32[i * DIMS + mOff + ss]! - coarseCentroids[c * DIMS + mOff + ss]!;
    }
  }
  process.stdout.write(`  sub-quantizer ${m + 1}/${M}...\n`);
  const cb = kmeans(subSample, SAMPLE_PQ, SUB, K_PQ, ITER_PQ);
  pqCodebooks.set(cb, m * K_PQ * SUB);
}
console.log(`  done in ${((Bun.nanoseconds() - t2) / 1e9).toFixed(1)}s`);

console.log(`\nEncoding ${N.toLocaleString()} vectors...`);
const t3 = Bun.nanoseconds();

const pqCodes = new Uint8Array(N * M);
const subVec = new Float32Array(SUB);

for (let m = 0; m < M; m++) {
  const mOff = m * SUB;
  const cbBase = m * K_PQ * SUB;

  for (let i = 0; i < N; i++) {
    const c = assignments[i]!;
    for (let ss = 0; ss < SUB; ss++) {
      subVec[ss] =
        vecsF32[i * DIMS + mOff + ss]! - coarseCentroids[c * DIMS + mOff + ss]!;
    }

    let bestK = 0,
      bestD = Infinity;
    for (let k = 0; k < K_PQ; k++) {
      const kBase = cbBase + k * SUB;
      const d0 = subVec[0]! - pqCodebooks[kBase]!;
      const d1 = subVec[1]! - pqCodebooks[kBase + 1]!;
      const d2 = d0 * d0 + d1 * d1;
      if (d2 < bestD) {
        bestD = d2;
        bestK = k;
      }
    }
    pqCodes[i * M + m] = bestK;
  }
  process.stdout.write(`\r  encoded sub-quantizer ${m + 1}/${M}...   `);
}
console.log(`\n  done in ${((Bun.nanoseconds() - t3) / 1e9).toFixed(1)}s`);

console.log(`\nBuilding per-cluster index...`);

const clusterOffsets = new Uint32Array(NLIST);
for (let c = 1; c < NLIST; c++)
  clusterOffsets[c] = clusterOffsets[c - 1]! + clusterSizes[c - 1]!;

const writePos = clusterOffsets.slice();
const sortedCodes = new Uint8Array(N * M);
const sortedLabels = new Uint8Array(N);
const sortedVecs = new Int16Array(N * DIMS);

for (let i = 0; i < N; i++) {
  const c = assignments[i]!;
  const pos = writePos[c]!++;
  sortedLabels[pos] = labels[i]!;
  const srcBase = i * M,
    dstBase = pos * M;
  for (let m = 0; m < M; m++) sortedCodes[dstBase + m] = pqCodes[srcBase + m]!;
  const vSrc = i * DIMS,
    vDst = pos * DIMS;
  for (let d = 0; d < DIMS; d++) {
    const x = Math.max(-1, Math.min(1, vecsF32[vSrc + d]!));
    sortedVecs[vDst + d] = Math.round(x * 32767);
  }
}

console.log("\nEvaluating on training sample (10K)...");

const EVAL_N = 10_000;
const evalStep = Math.floor(N / EVAL_N);
const adcBuf = new Float32Array(M * K_PQ);
let tp = 0,
  tn = 0,
  fp = 0,
  fn = 0;

for (let ei = 0; ei < EVAL_N; ei++) {
  const i = ei * evalStep;
  for (let d = 0; d < DIMS; d++) tmpVec[d] = vecsF32[i * DIMS + d]!;

  const coarseDists = new Float32Array(NLIST);
  for (let c = 0; c < NLIST; c++) {
    let d2 = 0;
    const cBase = c * DIMS;
    for (let d = 0; d < DIMS; d++) {
      const diff = tmpVec[d]! - coarseCentroids[cBase + d]!;
      d2 += diff * diff;
    }
    coarseDists[c] = d2;
  }

  const probeIdxs = new Uint16Array(DEFAULT_NPROBE);
  for (let p = 0; p < DEFAULT_NPROBE; p++) probeIdxs[p] = p;
  let worstInProbe = 0;
  let worstDist = 0;
  for (let p = 0; p < DEFAULT_NPROBE; p++) {
    if (coarseDists[p]! > worstDist) {
      worstDist = coarseDists[p]!;
      worstInProbe = p;
    }
  }
  for (let c = DEFAULT_NPROBE; c < NLIST; c++) {
    const d = coarseDists[c]!;
    if (d >= worstDist) continue;
    probeIdxs[worstInProbe] = c;
    worstDist = 0;
    for (let p = 0; p < DEFAULT_NPROBE; p++) {
      const dd = coarseDists[probeIdxs[p]!]!;
      if (dd > worstDist) {
        worstDist = dd;
        worstInProbe = p;
      }
    }
  }

  let totalCands = 0;
  const candDist: number[] = [];
  const candVIdx: number[] = [];

  for (let p = 0; p < DEFAULT_NPROBE; p++) {
    const c = probeIdxs[p]!;
    const cBase = c * DIMS;

    for (let mm = 0; mm < M; mm++) {
      const mOff = mm * SUB;
      const r0 = tmpVec[mOff]! - coarseCentroids[cBase + mOff]!;
      const r1 = tmpVec[mOff + 1]! - coarseCentroids[cBase + mOff + 1]!;
      const tableOff = mm * K_PQ;
      const cbBase = mm * K_PQ * SUB;
      for (let k = 0; k < K_PQ; k++) {
        const kOff = cbBase + k * SUB;
        const d0 = r0 - pqCodebooks[kOff]!;
        const d1 = r1 - pqCodebooks[kOff + 1]!;
        adcBuf[tableOff + k] = d0 * d0 + d1 * d1;
      }
    }

    const cOff = clusterOffsets[c]!,
      cSz = clusterSizes[c]!;
    for (let j = 0; j < cSz; j++) {
      const vIdx = cOff + j;
      const codeBase = vIdx * M;
      let dist = adcBuf[sortedCodes[codeBase]!]!;
      dist += adcBuf[K_PQ + sortedCodes[codeBase + 1]!]!;
      dist += adcBuf[2 * K_PQ + sortedCodes[codeBase + 2]!]!;
      dist += adcBuf[3 * K_PQ + sortedCodes[codeBase + 3]!]!;
      dist += adcBuf[4 * K_PQ + sortedCodes[codeBase + 4]!]!;
      dist += adcBuf[5 * K_PQ + sortedCodes[codeBase + 5]!]!;
      dist += adcBuf[6 * K_PQ + sortedCodes[codeBase + 6]!]!;
      candDist.push(dist);
      candVIdx.push(vIdx);
      totalCands++;
    }
  }

  const k = Math.min(KNN, totalCands);
  const idxs = Array.from({ length: totalCands }, (_, x) => x);
  idxs.sort((a, b) => candDist[a]! - candDist[b]!);

  const exactDist = new Float32Array(k);
  for (let j = 0; j < k; j++) {
    const vIdx = candVIdx[idxs[j]!]!;
    const vBase = vIdx * DIMS;
    let d2 = 0;
    for (let d = 0; d < DIMS; d++) {
      const q = Math.round(Math.max(-1, Math.min(1, tmpVec[d]!)) * 32767);
      const diff = q - sortedVecs[vBase + d]!;
      d2 += diff * diff;
    }
    exactDist[j] = d2;
  }
  const kf = Math.min(K_FINAL, k);
  const topIdxs = Array.from({ length: k }, (_, x) => x);
  for (let ki = 0; ki < kf; ki++) {
    let minI = ki;
    for (let j = ki + 1; j < k; j++) {
      if (exactDist[topIdxs[j]!]! < exactDist[topIdxs[minI]!]!) minI = j;
    }
    if (minI !== ki) {
      const tmp = topIdxs[ki]!;
      topIdxs[ki] = topIdxs[minI]!;
      topIdxs[minI] = tmp;
    }
  }
  let fraudNeighbors = 0;
  for (let j = 0; j < kf; j++) {
    const vIdx = candVIdx[idxs[topIdxs[j]!]!]!;
    if (sortedLabels[vIdx]) fraudNeighbors++;
  }

  const score = fraudNeighbors / kf;
  const pred = score >= 0.6 ? 1 : 0;
  const truth = labels[i]!;
  if (pred && truth) tp++;
  else if (!pred && !truth) tn++;
  else if (pred && !truth) fp++;
  else fn++;
}

const acc = (tp + tn) / EVAL_N;
const prec = tp / (tp + fp) || 0;
const rec = tp / (tp + fn) || 0;
const f1 = (2 * prec * rec) / (prec + rec) || 0;
console.log(
  `  acc=${(acc * 100).toFixed(2)}%  fp=${fp}  fn=${fn}  E=${fp + COST * fn}`,
);
console.log(
  `  Precision=${prec.toFixed(4)}  Recall=${rec.toFixed(4)}  F1=${f1.toFixed(4)}`,
);

// Layout:
//   [u32 NLIST][u32 N][u32 NPROBE][u32 M][u32 K_PQ][u32 KNN][u32 DIMS]  (7 × 4 = 28 B)
//   [f32 coarseCentroids: NLIST × DIMS]
//   [f32 pqCodebooks: M × K_PQ × SUB]
//   [u32 clusterSizes: NLIST]
//   [u32 clusterOffsets: NLIST]
//   [u8  sortedCodes: N × M]
//   [u8  sortedLabels: N]
//   [u8  padding: N % 2]
//   [i16 sortedVecs: N × DIMS]

const hdrB = 7 * 4;
const ctrdB = NLIST * DIMS * 4;
const cbB = M * K_PQ * SUB * 4;
const szB = NLIST * 4;
const offB = NLIST * 4;
const codesB = N * M;
const labB = N;
const padB = N & 1;
const vecsB = N * DIMS * 2;
const total = hdrB + ctrdB + cbB + szB + offB + codesB + labB + padB + vecsB;

console.log(`\nOutput: ${(total / 1024 / 1024).toFixed(1)} MB`);

const out = new ArrayBuffer(total);
const dv2 = new DataView(out);
let byteOff = 0;

dv2.setUint32(byteOff, NLIST, true);
byteOff += 4;
dv2.setUint32(byteOff, N, true);
byteOff += 4;
dv2.setUint32(byteOff, DEFAULT_NPROBE, true);
byteOff += 4;
dv2.setUint32(byteOff, M, true);
byteOff += 4;
dv2.setUint32(byteOff, K_PQ, true);
byteOff += 4;
dv2.setUint32(byteOff, KNN, true);
byteOff += 4;
dv2.setUint32(byteOff, DIMS, true);
byteOff += 4;

new Float32Array(out, byteOff, NLIST * DIMS).set(coarseCentroids);
byteOff += ctrdB;
new Float32Array(out, byteOff, M * K_PQ * SUB).set(pqCodebooks);
byteOff += cbB;
new Uint32Array(out, byteOff, NLIST).set(clusterSizes);
byteOff += szB;
new Uint32Array(out, byteOff, NLIST).set(clusterOffsets);
byteOff += offB;
new Uint8Array(out, byteOff, N * M).set(sortedCodes);
byteOff += codesB;
new Uint8Array(out, byteOff, N).set(sortedLabels);
byteOff += labB + padB;
new Int16Array(out, byteOff, N * DIMS).set(sortedVecs);

await Bun.write(OUTPUT, out);
console.log(`Saved → ${OUTPUT}`);
console.log(
  `Total build time: ${((Bun.nanoseconds() - t0) / 1e9).toFixed(1)}s`,
);
