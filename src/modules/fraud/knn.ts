import { vectors, labels, N, DIMS } from "./references.ts";

export const K = 5;

export function knn(query: Float32Array): number {
  const v = vectors;
  const topD = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const topL = [0, 0, 0, 0, 0];
  let worst = Infinity;
  let worstIdx = 0;

  for (let r = 0; r < N; r++) {
    const rOff = r * DIMS;
    let s = 0;
    for (let d = 0; d < DIMS; d++) {
      const x = query[d]! - v[rOff + d]!;
      s += x * x;
    }
    if (s < worst) {
      topD[worstIdx] = s;
      topL[worstIdx] = labels[r]!;
      let mx = topD[0]!,
        mi = 0;
      for (let i = 1; i < K; i++) {
        if (topD[i]! > mx) {
          mx = topD[i]!;
          mi = i;
        }
      }
      worst = mx;
      worstIdx = mi;
    }
  }

  return (topL[0]! + topL[1]! + topL[2]! + topL[3]! + topL[4]!) / K;
}
