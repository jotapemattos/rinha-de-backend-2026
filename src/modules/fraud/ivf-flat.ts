const FAST_NPROBE = Number(process.env.FAST_NPROBE ?? "4");
const FULL_NPROBE = Number(process.env.FULL_NPROBE ?? "64");
const K_FINAL = 5;

const MODEL = "resources/ivf-flat.bin";
if (Bun.file(MODEL).size === 0)
  throw new Error(
    `IVF-Flat model not found — run: bun scripts/build-ivf-flat.ts`,
  );

// mmap the index so the typed-array views read directly from the kernel page
// cache instead of the JS heap. Drops per-process RSS by ~50MB and lets
// multiple containers share the same physical pages when the file is mounted
// from a common volume.
const _mmap = Bun.mmap(MODEL);
if (_mmap.byteOffset !== 0)
  throw new Error("Unexpected Bun.mmap byteOffset != 0");
const buffer = _mmap.buffer;

const headDv = new DataView(buffer, 0, 28);
const NLIST = headDv.getUint32(0, true);
const N = headDv.getUint32(4, true);
const M = headDv.getUint32(12, true);
const K = headDv.getUint32(16, true);
const DIMS = headDv.getUint32(24, true);
const SUB = DIMS / M;

// Layout: [hdr 28B][centroids][pqCodebooks][sizes][offsets][fraudCounts][bboxMin][bboxMax]
//         [codes N*M][labels N][padding][vecs N*DIMS*2]
const hdrBlockBytes =
  28 +
  NLIST * DIMS * 4 +
  M * K * SUB * 4 +
  NLIST * 4 +
  NLIST * 4 +
  NLIST * 4 +
  NLIST * DIMS * 2 * 2;
const labelOffset = hdrBlockBytes + N * M;
const vecsOffset = labelOffset + N + (N & 1);

let byteOff = 28;
const coarseCentroids = new Float32Array(buffer, byteOff, NLIST * DIMS);
byteOff += NLIST * DIMS * 4;
byteOff += M * K * SUB * 4; // skip residual PQ codebooks
const clusterSizes = new Uint32Array(buffer, byteOff, NLIST);
byteOff += NLIST * 4;
const clusterOffsets = new Uint32Array(buffer, byteOff, NLIST);
byteOff += NLIST * 4;
const fraudCounts = new Uint32Array(buffer, byteOff, NLIST);
byteOff += NLIST * 4;
const bboxMin = new Int16Array(buffer, byteOff, NLIST * DIMS);
byteOff += NLIST * DIMS * 2;
const bboxMax = new Int16Array(buffer, byteOff, NLIST * DIMS);

const sortedLabels = new Uint8Array(buffer, labelOffset, N);
const sortedVecs = new Int16Array(buffer, vecsOffset, N * DIMS);

// Prefault every page of the mmap'd index so the kernel page cache is warm
// before traffic hits. Without this, lazy faults on cold cluster pages add
// ~100µs each to the p99 tail under burst.
{
  const u8 = new Uint8Array(buffer);
  let acc = 0;
  for (let i = 0; i < u8.length; i += 4096) acc |= u8[i]!;
  (globalThis as { __ivfPrefault?: number }).__ivfPrefault = acc;
}

// Pre-allocated buffers (single-threaded — safe to reuse per call).
const _q = new Float32Array(DIMS);
const _qS = new Int16Array(DIMS);
const _probIdx = new Uint16Array(FULL_NPROBE);
const _probDist = new Float32Array(FULL_NPROBE);
// Cache all NLIST centroid L2 distances from the first (fast) pass so the
// slow path can skip the second full centroid scan entirely.
const _centDist = new Float32Array(NLIST);

// Result heap: top K_FINAL by exact L2. Root is the worst (largest) distance.
const _hDist = new Float32Array(K_FINAL);
const _hLabel = new Uint8Array(K_FINAL);
let _hSize = 0;
// Running fraud count in the heap — updated incrementally to avoid the
// K_FINAL-element loop for unanimity checks.
let _heapFraudCount = 0;

function heapPush(dist: number, label: number): void {
  if (_hSize < K_FINAL) {
    let i = _hSize++;
    _hDist[i] = dist;
    _hLabel[i] = label;
    _heapFraudCount += label;
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
    // Replace root (current worst): update running fraud count before overwrite.
    _heapFraudCount += label - _hLabel[0]!;
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

// Heap-based top-nprobe centroid selection: O(NLIST × log nprobe).
// _probDist/_probIdx are used as a max-heap during scanning, then sorted ascending.
// With k=FAST_NPROBE (small heap), the threshold tightens quickly → many far centroids
// are skipped by the heap root comparison, making the fast-pass cheaper than k=FULL_NPROBE.
// All distances are also cached in _centDist[c] for the slow-path to reuse.
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

  let heapSize = 0;

  for (let c = 0; c < NLIST; c++) {
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

    _centDist[c] = dist;

    if (heapSize < nprobe) {
      _probDist[heapSize] = dist;
      _probIdx[heapSize] = c;
      let i = heapSize++;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (_probDist[p]! >= _probDist[i]!) break;
        const td = _probDist[p]!;
        _probDist[p] = _probDist[i]!;
        _probDist[i] = td;
        const ti = _probIdx[p]!;
        _probIdx[p] = _probIdx[i]!;
        _probIdx[i] = ti;
        i = p;
      }
    } else if (dist < _probDist[0]!) {
      _probDist[0] = dist;
      _probIdx[0] = c;
      let i = 0;
      while (true) {
        const l = 2 * i + 1,
          r = 2 * i + 2;
        let lg = i;
        if (l < nprobe && _probDist[l]! > _probDist[lg]!) lg = l;
        if (r < nprobe && _probDist[r]! > _probDist[lg]!) lg = r;
        if (lg === i) break;
        const td = _probDist[i]!;
        _probDist[i] = _probDist[lg]!;
        _probDist[lg] = td;
        const ti = _probIdx[i]!;
        _probIdx[i] = _probIdx[lg]!;
        _probIdx[lg] = ti;
        i = lg;
      }
    }
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
}

// Fast slow-path centroid selection: reuses the distances cached by findTopCentroids,
// so no FP multiplications are needed — just reads from _centDist and heap comparisons.
// ~13× cheaper than a second full L2 centroid scan.
function findTopCentroidsFromCache(nprobe: number): void {
  let heapSize = 0;
  for (let c = 0; c < NLIST; c++) {
    const dist = _centDist[c]!;
    if (heapSize < nprobe) {
      _probDist[heapSize] = dist;
      _probIdx[heapSize] = c;
      let i = heapSize++;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (_probDist[p]! >= _probDist[i]!) break;
        const td = _probDist[p]!;
        _probDist[p] = _probDist[i]!;
        _probDist[i] = td;
        const ti = _probIdx[p]!;
        _probIdx[p] = _probIdx[i]!;
        _probIdx[i] = ti;
        i = p;
      }
    } else if (dist < _probDist[0]!) {
      _probDist[0] = dist;
      _probIdx[0] = c;
      let i = 0;
      while (true) {
        const l = 2 * i + 1,
          r = 2 * i + 2;
        let lg = i;
        if (l < nprobe && _probDist[l]! > _probDist[lg]!) lg = l;
        if (r < nprobe && _probDist[r]! > _probDist[lg]!) lg = r;
        if (lg === i) break;
        const td = _probDist[i]!;
        _probDist[i] = _probDist[lg]!;
        _probDist[lg] = td;
        const ti = _probIdx[i]!;
        _probIdx[i] = _probIdx[lg]!;
        _probIdx[lg] = ti;
        i = lg;
      }
    }
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
}

// Returns the bbox lower-bound distance from _qS to cluster c in i16² space.
function bboxLb(c: number, worst: number): number {
  const bOff = c * DIMS;
  let lb = 0;
  let e: number;
  e =
    _qS[0]! < bboxMin[bOff]!
      ? bboxMin[bOff]! - _qS[0]!
      : _qS[0]! > bboxMax[bOff]!
        ? _qS[0]! - bboxMax[bOff]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[1]! < bboxMin[bOff + 1]!
      ? bboxMin[bOff + 1]! - _qS[1]!
      : _qS[1]! > bboxMax[bOff + 1]!
        ? _qS[1]! - bboxMax[bOff + 1]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[2]! < bboxMin[bOff + 2]!
      ? bboxMin[bOff + 2]! - _qS[2]!
      : _qS[2]! > bboxMax[bOff + 2]!
        ? _qS[2]! - bboxMax[bOff + 2]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[3]! < bboxMin[bOff + 3]!
      ? bboxMin[bOff + 3]! - _qS[3]!
      : _qS[3]! > bboxMax[bOff + 3]!
        ? _qS[3]! - bboxMax[bOff + 3]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[4]! < bboxMin[bOff + 4]!
      ? bboxMin[bOff + 4]! - _qS[4]!
      : _qS[4]! > bboxMax[bOff + 4]!
        ? _qS[4]! - bboxMax[bOff + 4]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[5]! < bboxMin[bOff + 5]!
      ? bboxMin[bOff + 5]! - _qS[5]!
      : _qS[5]! > bboxMax[bOff + 5]!
        ? _qS[5]! - bboxMax[bOff + 5]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[6]! < bboxMin[bOff + 6]!
      ? bboxMin[bOff + 6]! - _qS[6]!
      : _qS[6]! > bboxMax[bOff + 6]!
        ? _qS[6]! - bboxMax[bOff + 6]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[7]! < bboxMin[bOff + 7]!
      ? bboxMin[bOff + 7]! - _qS[7]!
      : _qS[7]! > bboxMax[bOff + 7]!
        ? _qS[7]! - bboxMax[bOff + 7]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[8]! < bboxMin[bOff + 8]!
      ? bboxMin[bOff + 8]! - _qS[8]!
      : _qS[8]! > bboxMax[bOff + 8]!
        ? _qS[8]! - bboxMax[bOff + 8]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[9]! < bboxMin[bOff + 9]!
      ? bboxMin[bOff + 9]! - _qS[9]!
      : _qS[9]! > bboxMax[bOff + 9]!
        ? _qS[9]! - bboxMax[bOff + 9]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[10]! < bboxMin[bOff + 10]!
      ? bboxMin[bOff + 10]! - _qS[10]!
      : _qS[10]! > bboxMax[bOff + 10]!
        ? _qS[10]! - bboxMax[bOff + 10]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[11]! < bboxMin[bOff + 11]!
      ? bboxMin[bOff + 11]! - _qS[11]!
      : _qS[11]! > bboxMax[bOff + 11]!
        ? _qS[11]! - bboxMax[bOff + 11]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[12]! < bboxMin[bOff + 12]!
      ? bboxMin[bOff + 12]! - _qS[12]!
      : _qS[12]! > bboxMax[bOff + 12]!
        ? _qS[12]! - bboxMax[bOff + 12]!
        : 0;
  lb += e * e;
  if (lb >= worst) return lb;
  e =
    _qS[13]! < bboxMin[bOff + 13]!
      ? bboxMin[bOff + 13]! - _qS[13]!
      : _qS[13]! > bboxMax[bOff + 13]!
        ? _qS[13]! - bboxMax[bOff + 13]!
        : 0;
  lb += e * e;
  return lb;
}

// Scan clusters [fromP, toP) with exact Int16 L2, accumulating into the K_FINAL heap.
// The heap persists across calls so fast + full scans share the same top-K.
// Partial-sum early exit: bail after each block of 4 dims if already worse than heap top.
// Pure-label same-label skip: if heap is unanimous AND cluster has the same pure label,
// additional vectors from that cluster cannot change the result — skip the scan entirely.
function scanClustersExact(fromP: number, toP: number): void {
  const sv = sortedVecs;
  const sl = sortedLabels;
  const offsets = clusterOffsets;
  const sizes = clusterSizes;
  const fc = fraudCounts;
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
    const c = _probIdx[p]!;

    // Pure-label same-label skip: heap full + unanimous + cluster same pure label.
    // Safe because adding same-label vectors to an already-unanimous heap cannot
    // flip the decision, regardless of their distances.
    if (_hSize >= K_FINAL) {
      const hfc = _heapFraudCount;
      if (hfc === 0 || hfc === K_FINAL) {
        const cfc = fc[c]!;
        const csz = sizes[c]!;
        if ((hfc === K_FINAL && cfc === csz) || (hfc === 0 && cfc === 0)) continue;
      }
    }

    if (_hSize >= K_FINAL && bboxLb(c, _hDist[0]!) >= _hDist[0]!) continue;

    const cOff = offsets[c]!;
    const end = cOff + sizes[c]!;

    for (let row = cOff; row < end; row++) {
      const vBase = row * DIMS;
      const e0 = q0 - sv[vBase]!;
      const e1 = q1 - sv[vBase + 1]!;
      const e2 = q2 - sv[vBase + 2]!;
      const e3 = q3 - sv[vBase + 3]!;
      let dist = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
      if (_hSize >= K_FINAL && dist >= _hDist[0]!) continue;
      const e4 = q4 - sv[vBase + 4]!;
      const e5 = q5 - sv[vBase + 5]!;
      const e6 = q6 - sv[vBase + 6]!;
      const e7 = q7 - sv[vBase + 7]!;
      dist += e4 * e4 + e5 * e5 + e6 * e6 + e7 * e7;
      if (_hSize >= K_FINAL && dist >= _hDist[0]!) continue;
      const e8 = q8 - sv[vBase + 8]!;
      const e9 = q9 - sv[vBase + 9]!;
      const e10 = q10 - sv[vBase + 10]!;
      const e11 = q11 - sv[vBase + 11]!;
      dist += e8 * e8 + e9 * e9 + e10 * e10 + e11 * e11;
      if (_hSize >= K_FINAL && dist >= _hDist[0]!) continue;
      const e12 = q12 - sv[vBase + 12]!;
      const e13 = q13 - sv[vBase + 13]!;
      dist += e12 * e12 + e13 * e13;
      heapPush(dist, sl[row]!);
    }
  }
}

let fastPathCount = 0;
let fullPathCount = 0;
export function getPathStats() { return { fastPathCount, fullPathCount }; }
export function resetPathStats() { fastPathCount = 0; fullPathCount = 0; }

// Forces both scanClustersExact branches through the JIT regardless of vec content.
// Call this during server warmup so the full-path code is FTL-compiled before real traffic.
export function ivfFlatWarmupBothPaths(vec: Float32Array): void {
  for (let d = 0; d < DIMS; d++) {
    const x = Math.max(-1, Math.min(1, vec[d]!));
    _q[d] = x;
    _qS[d] = Math.round(x * 32767);
  }
  findTopCentroids(FAST_NPROBE);
  _hSize = 0; _heapFraudCount = 0;
  scanClustersExact(0, FAST_NPROBE);
  _hSize = 0; _heapFraudCount = 0;
  findTopCentroidsFromCache(FULL_NPROBE);
  scanClustersExact(FAST_NPROBE, FULL_NPROBE);
  _hSize = 0; _heapFraudCount = 0;
}

export function ivfFlatScore(vec: Float32Array): number {
  for (let d = 0; d < DIMS; d++) {
    const x = Math.max(-1, Math.min(1, vec[d]!));
    _q[d] = x;
    _qS[d] = Math.round(x * 32767);
  }

  // Fast pass: small heap (k=FAST_NPROBE) fills quickly → tight threshold → cheap centroid
  // scan. All distances cached in _centDist for the slow path to reuse without recomputing.
  findTopCentroids(FAST_NPROBE);
  _hSize = 0;
  _heapFraudCount = 0;
  scanClustersExact(0, FAST_NPROBE);

  if (_heapFraudCount === 0 || _heapFraudCount === K_FINAL) {
    fastPathCount++;
    return _heapFraudCount / K_FINAL;
  }

  // Slow pass: reuse cached distances to find top-FULL_NPROBE without a second L2 scan.
  findTopCentroidsFromCache(FULL_NPROBE);
  scanClustersExact(FAST_NPROBE, FULL_NPROBE);
  fullPathCount++;
  return _heapFraudCount / K_FINAL;
}
