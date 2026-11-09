# Fast exact kernel: for each exported structure, enumerate all lift vectors (Gray code), all 2^m signs
# (WHT), and test G00 != 0 and the 2x2 minors G00*Gee = |G0e|^2 exactly in Z[sqrt2][i].
# Order of patterns: all-fault (Galois partner of 0) first, then weight n-1, then the rest.
# Usage: julia -t 16 fast_enum.jl structs.txt n d m
# Optional: SHARD=k NSHARDS=K runs only the k-th of K contiguous pieces of the lift range of every structure
# (k = 0..K-1), so one structure can be spread over many jobs. Lift vectors are UInt128 (a*n may exceed 64).
using Base.Threads
const SHARD = parse(Int, get(ENV, "SHARD", "0")); const NSHARDS = parse(Int, get(ENV, "NSHARDS", "1"))
const fname = ARGS[1]; const n = parse(Int, ARGS[2]); const d = parse(Int, ARGS[3]); const m = parse(Int, ARGS[4])
const MASK = (1 << n) - 1; const C = (n + 1) ÷ 2; const NE = 1 << m
pc(x) = count_ones(x)
const GALOIS = parse(Int, get(ENV, "GALOIS", "1"))
const Eall = [e for e in 1:MASK if pc(e) < d || (GALOIS == 1 && pc(e) > n - d)]
const E = sort(Eall, by = e -> (min(pc(e), n - pc(e)), -pc(e)))

# accumulate value of element (p,x,z) for pair (e,f) into 4-vector slots
@inline function addval!(acc, k, p, x, z, e, f)
    kk = z ⊻ e ⊻ f
    (kk & (~x & MASK)) != 0 && return
    q = mod(p + 2 * pc(e & x) - pc(kk), 4)
    w = pc(x); mag = 1 << (C - (w + 1) ÷ 2)
    slot = (iseven(q) ? 0 : 2) + (w & 1)
    sg = (q == 0 || q == 1) ? mag : -mag
    @inbounds acc[k, slot + 1] += sg
end
function wht!(A)  # in place along dim 1, length NE
    h = 1
    while h < NE
        for i in 0:2h:NE-1, j in i:i+h-1, s in 1:4
            @inbounds a = A[j+1, s]; @inbounds b = A[j+h+1, s]
            @inbounds A[j+1, s] = a + b; @inbounds A[j+h+1, s] = a - b
        end
        h *= 2
    end
end
function gram!(A, P, X, Z, e, f)
    fill!(A, 0)
    for k in 1:NE; addval!(A, k, P[k], X[k], Z[k], e, f) end
    wht!(A)
end
# Z[sqrt2] products
@inline rmul(a0, a1, b0, b1) = (a0 * b0 + 2 * a1 * b1, a0 * b1 + a1 * b0)

function process(a, AX, BZ, basis)
    L = length(basis)
    nsurv = Atomic{Int}(0)
    slo = SHARD * (1 << L) ÷ NSHARDS; shi = (SHARD + 1) * (1 << L) ÷ NSHARDS - 1
    nlifts = shi - slo + 1
    nlifts <= 0 && return 0, 0
    nchunks = max(1, min(nlifts, 4 * nthreads()))
    chunk = nlifts ÷ nchunks
    @threads for t in 0:nchunks-1
        P = zeros(Int, NE); X = zeros(Int, NE); Z = zeros(Int, NE)
        G00 = zeros(Int, NE, 4); G0e = zeros(Int, NE, 4); Gee = zeros(Int, NE, 4)
        alive = falses(NE)
        GQ = zeros(Int, m); GX = zeros(Int, m); GZ = zeros(Int, m)
        lo = slo + t * chunk; hi = (t == nchunks - 1) ? shi : lo + chunk - 1
        for msk in lo:hi
            cv = UInt128(0)
            for bi in 1:L; (msk >> (bi - 1)) & 1 == 1 && (cv ⊻= basis[bi]) end
            # generators
            P[1] = 0; X[1] = 0; Z[1] = 0; cnt = 1
            for i in 0:a-1
                ci = Int((cv >> (i * n)) & MASK)
                GQ[i+1] = pc(AX[i+1] & ci) % 4; GX[i+1] = AX[i+1]; GZ[i+1] = ci
            end
            for (j, b) in enumerate(BZ); GQ[a+j] = 0; GX[a+j] = 0; GZ[a+j] = b end
            for g in 1:a+length(BZ)
                q = GQ[g]; x2 = GX[g]; z2 = GZ[g]
                for k in 1:cnt
                    P[cnt + k] = mod(P[k] + q + 2 * pc(Z[k] & x2), 4); X[cnt + k] = X[k] ⊻ x2; Z[cnt + k] = Z[k] ⊻ z2
                end
                cnt *= 2
            end
            gram!(G00, P, X, Z, 0, 0)
            anyalive = false
            for s in 1:NE
                alive[s] = G00[s, 1] != 0 || G00[s, 2] != 0
                anyalive |= alive[s]
            end
            for e in E
                anyalive || break
                gram!(G0e, P, X, Z, 0, e); gram!(Gee, P, X, Z, e, e)
                anyalive = false
                for s in 1:NE
                    alive[s] || continue
                    l0, l1 = rmul(G00[s, 1], G00[s, 2], Gee[s, 1], Gee[s, 2])
                    r0a, r1a = rmul(G0e[s, 1], G0e[s, 2], G0e[s, 1], G0e[s, 2])
                    r0b, r1b = rmul(G0e[s, 3], G0e[s, 4], G0e[s, 3], G0e[s, 4])
                    alive[s] = (l0 == r0a + r0b) && (l1 == r1a + r1b)
                    anyalive |= alive[s]
                end
            end
            if anyalive
                atomic_add!(nsurv, count(alive))
                println("SURVIVOR a=$a AX=$AX BZ=$BZ c=$cv signs=$(findall(alive) .- 1)")
            end
        end
    end
    nsurv[], nlifts
end

function main()
    tot = 0; totsurv = 0; t0 = time()
    for line in eachline(fname)
        f = split(line, "|")
        a = parse(Int, strip(f[1]))
        AX = [parse(Int, s) for s in split(strip(f[2]))]
        BZ = [parse(Int, s) for s in split(strip(f[3]))]
        basis = [parse(UInt128, s) for s in split(strip(f[4]))]
        s, nl = process(a, AX, BZ, basis)
        tot += nl * NE; totsurv += s
        println("structure AX=$AX L=$(length(basis)) survivors=$s  elapsed=$(round(time() - t0, digits=1))s"); flush(stdout)
    end
    println("RESULT file=$fname" * (NSHARDS > 1 ? " shard $SHARD of $NSHARDS" : "") * " signed candidates $tot survivors $totsurv")
end
main()
