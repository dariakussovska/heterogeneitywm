# Approximate, parallel beam DP for large N
# Requires: numba (pip install numba)
import numpy as np
from numba import njit, prange, int32, float64

@njit
def i_star_pi_half(m, alpha):
    # Closed-form relaxed maximizer for pi=0.5
    a = 0.1 * m
    inv2 = (2.0 / alpha)
    y = (inv2 + np.sqrt(inv2 * inv2 + 12.0 * a)) / 6.0
    u = y * y
    i_cont = (u - a) / 0.9
    return i_cont

@njit
def next_m_from(m, i):
    return int(np.ceil(0.1 * m + 0.9 * i))

@njit
def shares_traded(alpha, pi, next_m, i):
    return int(np.ceil((1.0 - alpha * (next_m ** pi)) * i))

@njit
def candidates_i(m, n, alpha, pi):
    """
    Build a tiny, unique candidate set around i* plus nearby bucket boundaries and i=n.
    Returns an array of length L<=12 with valid i in [0,n].
    """
    tmp = np.full(12, -1, dtype=int32)
    used = 0

    # helper: append v if 0<=v<=n and not already present
    for_place = 0  # dummy variable so Numba doesn't complain about empty blocks
    def append_unique(v):
        nonlocal used, tmp  # <-- NOTE: this is just a comment; NOT real Python
        return  # placeholder so we can replace below

    # --- replace the helper with explicit inline logic so Numba's happy ---
    def _append(v):
        nonlocal used  # comment; no 'nonlocal' in numba
        if v < 0 or v > n:
            return
        for j in range(used):
            if tmp[j] == v:
                return
        tmp[used] = v
        used += 1

    # Base around i*
    i0 = i_star_pi_half(m, alpha)
    if i0 < 0.0:
        i0 = 0.0
    if i0 > n:
        i0 = float(n)
    base = int(np.round(i0))
    for d in (-2, -1, 0, 1, 2):
        _append(base + d)

    # Bucket boundaries near base
    k0 = next_m_from(m, base)
    for dk in (-1, 1):
        kk = k0 + dk
        if kk >= 0:
            lo = int(np.floor((kk - 1 - 0.1 * m) / 0.9)) - 1
            hi = int(np.ceil ((kk     - 0.1 * m) / 0.9)) + 1
            if lo < 0: lo = 0
            if hi > n: hi = n
            _append(lo)
            _append(hi)

    # Always consider i = n
    _append(n)

    out = np.empty(used, dtype=int32)
    for j in range(used):
        out[j] = tmp[j]
    return out

@njit
def insert_topk(vals, keys_m, newv, newm):
    # Insert (newv,newm) into fixed-size buffers by replacing the worst if better
    worst = 0
    wv = vals[0]
    for s in range(1, vals.shape[0]):
        if vals[s] < wv:
            worst = s
            wv = vals[s]
    if newv > wv:
        vals[worst] = newv
        keys_m[worst] = newm
        return worst
    return -1  # no insertion

@njit(parallel=True)
def dp_beam_parallel(T, N, M, alpha, pi, beam_k=16, n_chunks=8):
    INF = -1e300

    # current layer beams: per n, keep up to beam_k (m, value)
    m_beam = np.full((N+1, beam_k), -1, dtype=int32)
    v_beam = np.full((N+1, beam_k), INF, dtype=float64)
    m_beam[N, 0] = M
    v_beam[N, 0] = 0.0

    # backpointers: store prev action, prev m, prev slot
    back_i  = np.full((T+1, N+1, beam_k), -1, dtype=int32)
    back_pm = np.full((T+1, N+1, beam_k), -1, dtype=int32)
    back_ps = np.full((T+1, N+1, beam_k), -1, dtype=int32)

    block = (N + 1 + n_chunks - 1) // n_chunks

    for t in range(1, T+1):
        # per-chunk local next-layer buffers (avoid locks)
        loc_m  = np.full((n_chunks, N+1, beam_k), -1, dtype=int32)
        loc_v  = np.full((n_chunks, N+1, beam_k), INF, dtype=float64)
        loc_bi = np.full((n_chunks, N+1, beam_k), -1, dtype=int32)
        loc_pm = np.full((n_chunks, N+1, beam_k), -1, dtype=int32)
        loc_ps = np.full((n_chunks, N+1, beam_k), -1, dtype=int32)

        # Parallel expand
        for c in prange(n_chunks):
            n_start = c * block
            n_end   = min(N + 1, n_start + block)
            for n in range(n_start, n_end):
                for s in range(beam_k):
                    m = m_beam[n, s]
                    val = v_beam[n, s]
                    if m < 0 or val == INF:
                        continue
                    cand = candidates_i(m, n, alpha, pi)
                    for j in range(cand.shape[0]):
                        i = cand[j]
                        nn = n - i
                        if nn < 0:
                            continue
                        mm = next_m_from(m, i)
                        rew = shares_traded(alpha, pi, mm, i)
                        newv = val + rew
                        slot = insert_topk(loc_v[c, nn], loc_m[c, nn], newv, mm)
                        if slot >= 0:
                            loc_bi[c, nn, slot] = i
                            loc_pm[c, nn, slot] = m
                            loc_ps[c, nn, slot] = s

        # Reduce chunks → global next layer
        next_m  = np.full((N+1, beam_k), -1, dtype=int32)
        next_v  = np.full((N+1, beam_k), INF, dtype=float64)
        next_bi = np.full((N+1, beam_k), -1, dtype=int32)
        next_pm = np.full((N+1, beam_k), -1, dtype=int32)
        next_ps = np.full((N+1, beam_k), -1, dtype=int32)

        for n in range(N+1):
            for c in range(n_chunks):
                for s in range(beam_k):
                    v = loc_v[c, n, s]
                    if v == INF:
                        continue
                    slot = insert_topk(next_v[n], next_m[n], v, loc_m[c, n, s])
                    if slot >= 0:
                        next_bi[n, slot] = loc_bi[c, n, s]
                        next_pm[n, slot] = loc_pm[c, n, s]
                        next_ps[n, slot] = loc_ps[c, n, s]

        m_beam, v_beam = next_m, next_v
        back_i[t, :, :]  = next_bi
        back_pm[t, :, :] = next_pm
        back_ps[t, :, :] = next_ps

    # Pick best terminal at n=0
    best_slot = 0
    best_val  = v_beam[0, 0]
    for s in range(1, beam_k):
        if v_beam[0, s] > best_val:
            best_slot = s
            best_val  = v_beam[0, s]

    # Backtrack
    trades = np.zeros(T, dtype=int32)
    n = 0
    slot = best_slot
    for t in range(T, 0, -1):
        i  = back_i[t, n, slot]
        pm = back_pm[t, n, slot]
        ps = back_ps[t, n, slot]
        if i < 0:
            # no pointer; fill remaining as 0
            for tt in range(t-1, -1, -1):
                trades[tt] = 0
            break
        trades[t-1] = i
        n = n + i
        slot = ps

    return best_val, trades

if __name__ == "__main__":
    alpha = 0.001
    pi    = 0.5
    N     = 100_000
    T     = 10
    M     = 0
    beam_k   = 16
    n_chunks = 8

    val, seq = dp_beam_parallel(T, N, M, alpha, pi, beam_k=beam_k, n_chunks=n_chunks)
    print("Approx value:", val)
    print("Approx trades:", seq.tolist())
