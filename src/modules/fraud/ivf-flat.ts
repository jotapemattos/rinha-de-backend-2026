const FAST_NPROBE = 5;
const FULL_NPROBE = 20;
const K_FINAL = 5;

const MODEL = "resources/ivf-flat.bin";
if (!(await Bun.file(MODEL).exists()))
  throw new Error(`IVF-Flat model not found — run: bun scripts/build-ivf-flat.ts`);

const headBuf = await Bun.file(MODEL).slice(0, 28).arrayBuffer();
const headDv = new DataView(headBuf);
const NLIST = headDv.getUint32(0, true);
const N = headDv.getUint32(4, true);
const M = headDv.getUint32(12, true);
const K = headDv.getUint32(16, true);
const DIMS = headDv.getUint32(24, true);
const SUB = DIMS / M;

const hdrBlockBytes =
  28 + NLIST * DIMS * 4 + M * K * SUB * 4 + NLIST * 4 + NLIST * 4;
const labelOffset = hdrBlockBytes + N * M;
const vecsOffset = labelOffset + N + (N & 1);

const hdrBuf = await Bun.file(MODEL).slice(0, hdrBlockBytes).arrayBuffer();
const lblBuf = await Bun.file(MODEL)
  .slice(labelOffset, labelOffset + N)
  .arrayBuffer();
const vecsBuf = await Bun.file(MODEL)
  .slice(vecsOffset, vecsOffset + N * DIMS * 2)
  .arrayBuffer();

let byteOff = 28;
const coarseCentroids = new Float32Array(hdrBuf, byteOff, NLIST * DIMS);
byteOff += NLIST * DIMS * 4;
byteOff += M * K * SUB * 4;
const clusterSizes = new Uint32Array(hdrBuf, byteOff, NLIST);
byteOff += NLIST * 4;
const clusterOffsets = new Uint32Array(hdrBuf, byteOff, NLIST);

const sortedLabels = new Uint8Array(lblBuf);
const sortedVecs = new Int16Array(vecsBuf);

// Pre-allocated buffers (single-threaded — safe to reuse per call).
const _q = new Float32Array(DIMS);
const _qS = new Int16Array(DIMS);
const _probIdx = new Uint16Array(FULL_NPROBE); // top centroid indices, sorted by dist
const _probDist = new Float32Array(FULL_NPROBE);

// Max-heap of size K_FINAL: root = worst (largest) distance in current top-5.
const _hDist = new Float32Array(K_FINAL);
const _hLabel = new Uint8Array(K_FINAL);
let _hSize = 0;

function heapPush(dist: number, label: number): void {
  if (_hSize < K_FINAL) {
    let i = _hSize++;
    _hDist[i] = dist;
    _hLabel[i] = label;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (_hDist[p]! >= _hDist[i]!) break;
      const td = _hDist[p]!;
      _hDist[p] = _hDist[i]!;
      _hDist[i] = td;
      const tl = _hLabel[p]!;
      _hLabel[p] = _hLabel[i]!;
      _hLabel[i] = tl;
      i = p;
    }
  } else if (dist < _hDist[0]!) {
    _hDist[0] = dist;
    _hLabel[0] = label;
    let i = 0;
    while (true) {
      const l = 2 * i + 1,
        r = 2 * i + 2;
      let lg = i;
      if (l < K_FINAL && _hDist[l]! > _hDist[lg]!) lg = l;
      if (r < K_FINAL && _hDist[r]! > _hDist[lg]!) lg = r;
      if (lg === i) break;
      const td = _hDist[i]!;
      _hDist[i] = _hDist[lg]!;
      _hDist[lg] = td;
      const tl = _hLabel[i]!;
      _hLabel[i] = _hLabel[lg]!;
      _hLabel[lg] = tl;
      i = lg;
    }
  }
}

// Keep the top-nprobe nearest IVF centroids sorted by distance.
// This intentionally supports both FAST_NPROBE=8 and FULL_NPROBE=24.
// The fast path only keeps top-8; top-24 is recomputed only for borderline cases.
function findTopCentroids(nprobe: number): void {
  const cc = coarseCentroids;
  const q0 = _q[0]!;
  const q1 = _q[1]!;
  const q2 = _q[2]!;
  const q3 = _q[3]!;
  const q4 = _q[4]!;
  const q5 = _q[5]!;
  const q6 = _q[6]!;
  const q7 = _q[7]!;
  const q8 = _q[8]!;
  const q9 = _q[9]!;
  const q10 = _q[10]!;
  const q11 = _q[11]!;
  const q12 = _q[12]!;
  const q13 = _q[13]!;

  for (let p = 0; p < nprobe; p++) {
    _probIdx[p] = p;
    const cBase = p * DIMS;
    const d0 = q0 - cc[cBase]!;
    const d1 = q1 - cc[cBase + 1]!;
    const d2 = q2 - cc[cBase + 2]!;
    const d3 = q3 - cc[cBase + 3]!;
    const d4 = q4 - cc[cBase + 4]!;
    const d5 = q5 - cc[cBase + 5]!;
    const d6 = q6 - cc[cBase + 6]!;
    const d7 = q7 - cc[cBase + 7]!;
    const d8 = q8 - cc[cBase + 8]!;
    const d9 = q9 - cc[cBase + 9]!;
    const d10 = q10 - cc[cBase + 10]!;
    const d11 = q11 - cc[cBase + 11]!;
    const d12 = q12 - cc[cBase + 12]!;
    const d13 = q13 - cc[cBase + 13]!;
    _probDist[p] =
      d0 * d0 +
      d1 * d1 +
      d2 * d2 +
      d3 * d3 +
      d4 * d4 +
      d5 * d5 +
      d6 * d6 +
      d7 * d7 +
      d8 * d8 +
      d9 * d9 +
      d10 * d10 +
      d11 * d11 +
      d12 * d12 +
      d13 * d13;
  }

  for (let i = 1; i < nprobe; i++) {
    const di = _probDist[i]!;
    const ii = _probIdx[i]!;
    let j = i;
    while (j > 0 && _probDist[j - 1]! > di) {
      _probDist[j] = _probDist[j - 1]!;
      _probIdx[j] = _probIdx[j - 1]!;
      j--;
    }
    _probDist[j] = di;
    _probIdx[j] = ii;
  }

  for (let c = nprobe; c < NLIST; c++) {
    const cBase = c * DIMS;
    const d0 = q0 - cc[cBase]!;
    const d1 = q1 - cc[cBase + 1]!;
    const d2 = q2 - cc[cBase + 2]!;
    const d3 = q3 - cc[cBase + 3]!;
    const d4 = q4 - cc[cBase + 4]!;
    const d5 = q5 - cc[cBase + 5]!;
    const d6 = q6 - cc[cBase + 6]!;
    const d7 = q7 - cc[cBase + 7]!;
    const d8 = q8 - cc[cBase + 8]!;
    const d9 = q9 - cc[cBase + 9]!;
    const d10 = q10 - cc[cBase + 10]!;
    const d11 = q11 - cc[cBase + 11]!;
    const d12 = q12 - cc[cBase + 12]!;
    const d13 = q13 - cc[cBase + 13]!;
    const dist =
      d0 * d0 +
      d1 * d1 +
      d2 * d2 +
      d3 * d3 +
      d4 * d4 +
      d5 * d5 +
      d6 * d6 +
      d7 * d7 +
      d8 * d8 +
      d9 * d9 +
      d10 * d10 +
      d11 * d11 +
      d12 * d12 +
      d13 * d13;

    if (dist >= _probDist[nprobe - 1]!) continue;

    let j = nprobe - 1;
    while (j > 0 && _probDist[j - 1]! > dist) {
      _probDist[j] = _probDist[j - 1]!;
      _probIdx[j] = _probIdx[j - 1]!;
      j--;
    }
    _probDist[j] = dist;
    _probIdx[j] = c;
  }
}

// Scan clusters [fromP, toP) using exact L2 with early termination.
// Reads/updates the global max-heap (_hDist, _hLabel, _hSize).
function scanClusters(fromP: number, toP: number): void {
  const sv = sortedVecs;
  const sl = sortedLabels;
  const probIdx = _probIdx;
  const offsets = clusterOffsets;
  const sizes = clusterSizes;

  const q0 = _qS[0]!;
  const q1 = _qS[1]!;
  const q2 = _qS[2]!;
  const q3 = _qS[3]!;
  const q4 = _qS[4]!;
  const q5 = _qS[5]!;
  const q6 = _qS[6]!;
  const q7 = _qS[7]!;
  const q8 = _qS[8]!;
  const q9 = _qS[9]!;
  const q10 = _qS[10]!;
  const q11 = _qS[11]!;
  const q12 = _qS[12]!;
  const q13 = _qS[13]!;

  for (let p = fromP; p < toP; p++) {
    const c = probIdx[p]!;
    const cOff = offsets[c]!;
    const cSz = sizes[c]!;
    const end = cOff + cSz;

    for (let row = cOff; row < end; row++) {
      const vBase = row * DIMS;

      const e0 = q0 - sv[vBase]!;
      const e1 = q1 - sv[vBase + 1]!;
      const e2 = q2 - sv[vBase + 2]!;
      const e3 = q3 - sv[vBase + 3]!;

      const partial4 = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;

      if (_hSize >= K_FINAL && partial4 >= _hDist[0]!) continue;

      const e4 = q4 - sv[vBase + 4]!;
      const e5 = q5 - sv[vBase + 5]!;
      const e6 = q6 - sv[vBase + 6]!;
      const e7 = q7 - sv[vBase + 7]!;

      const partial8 = partial4 + e4 * e4 + e5 * e5 + e6 * e6 + e7 * e7;

      if (_hSize >= K_FINAL && partial8 >= _hDist[0]!) continue;

      const e8 = q8 - sv[vBase + 8]!;
      const e9 = q9 - sv[vBase + 9]!;
      const e10 = q10 - sv[vBase + 10]!;
      const e11 = q11 - sv[vBase + 11]!;
      const e12 = q12 - sv[vBase + 12]!;
      const e13 = q13 - sv[vBase + 13]!;

      heapPush(
        partial8 +
          e8 * e8 +
          e9 * e9 +
          e10 * e10 +
          e11 * e11 +
          e12 * e12 +
          e13 * e13,
        sl[row]!,
      );
    }
  }
}

function fraudCountFromHeap(): number {
  let fraudCount = 0;
  for (let i = 0; i < _hSize; i++) fraudCount += _hLabel[i]!;
  return fraudCount;
}

function decideFromHeap(fraudCount: number): number {
  return fraudCount / K_FINAL;
}

export function ivfFlatScore(vec: Float32Array): number {
  for (let d = 0; d < DIMS; d++) {
    const x = Math.max(-1, Math.min(1, vec[d]!));
    _q[d] = x;
    _qS[d] = Math.round(x * 32767);
  }

  findTopCentroids(FULL_NPROBE);

  _hSize = 0;
  scanClusters(0, FAST_NPROBE);

  let fraudCount = fraudCountFromHeap();

  if (fraudCount !== 2 && fraudCount !== 3) {
    return decideFromHeap(fraudCount);
  }

  scanClusters(FAST_NPROBE, FULL_NPROBE);

  fraudCount = fraudCountFromHeap();

  return decideFromHeap(fraudCount);
}
