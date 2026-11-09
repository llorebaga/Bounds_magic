# Independent re-implementation (Julia) of the exhaustive [[n,3]] search, written separately from
# enum_codes.py. Differences: no W-gauge (all commuting lifts enumerated), S_n orbit representatives
# chosen as the lexicographically minimal RREF over ALL permutations, Gram entries computed from the
# single-qubit T-basis matrix elements with exact integer arithmetic in Z[zeta8] (4 coordinates),
# collinearity tested by exact 2x2 minors G00*Gee - G0e*conj(G0e) = 0 in Z[zeta8].
# Usage: julia -t 16 independent_check.jl n d
using Base.Threads
# local replacements for Combinatorics.jl (no package installation)
function combinations(v, k)
    v = collect(v); out = Vector{Vector{eltype(v)}}()
    k == 0 && return [eltype(v)[]]
    function rec(start, cur)
        if length(cur) == k; push!(out, copy(cur)); return end
        for i in start:length(v); push!(cur, v[i]); rec(i + 1, cur); pop!(cur) end
    end
    rec(1, eltype(v)[]); out
end
function permutations(v)
    v = collect(v); out = Vector{Vector{eltype(v)}}()
    function rec(i)
        if i > length(v); push!(out, copy(v)); return end
        for j in i:length(v); v[i], v[j] = v[j], v[i]; rec(i + 1); v[i], v[j] = v[j], v[i] end
    end
    rec(1); out
end

const n = parse(Int, ARGS[1]); const d = parse(Int, ARGS[2]); const m = n - 3
const MASK = (1 << n) - 1
pc(x) = count_ones(x)

# ---- Z[zeta8] arithmetic: (a0,a1,a2,a3) = a0 + a1 z + a2 z^2 + a3 z^3, z^4 = -1
const Zq = NTuple{4,Int}
zadd(a::Zq, b::Zq) = (a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4])
zsub(a::Zq, b::Zq) = (a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4])
function zmul(a::Zq, b::Zq)
    c = zeros(Int, 4)
    for i in 0:3, j in 0:3
        k = i + j
        if k < 4; c[k+1] += a[i+1]*b[j+1] else c[k-3] -= a[i+1]*b[j+1] end
    end
    (c[1], c[2], c[3], c[4])
end
zconj(a::Zq) = (a[1], -a[4], -a[3], -a[2])
const ZERO = (0,0,0,0)
zpow(k) = (k = mod(k, 8); k < 4 ? ntuple(i -> i == k+1 ? 1 : 0, 4) : ntuple(i -> i == k-3 ? -1 : 0, 4))

# single-qubit <T|Z^e P Z^f|T> * sqrt2, P in {I,X,Y,Z} as (x,z) bits with Hermitian phase i^{x z}
# |T> = (|0> + z|1>)/sqrt2. Compute <T|Z^e X^x Z^zz Z^f|T>*2 exactly then track sqrt2 powers globally:
# entries of 2*<..> lie in Z[zeta8].
function qubit_elem(e, x, zz, f)
    # vector components of Z^f|T>*sqrt2: (1, (-1)^f z)
    v = (zpow(0), f == 1 ? zpow(5) : zpow(1))       # -z = z^5
    # apply Z^zz
    v = (v[1], zz == 1 ? zmul(zpow(4), v[2]) : v[2])
    # apply X^x
    v = x == 1 ? (v[2], v[1]) : v
    # Hermitian phase i^{x*zz} = z^{2 x zz}
    ph = zpow(2 * x * zz)
    # bra <T|Z^e * sqrt2 = (1, conj((-1)^e z)) = (1, (-1)^e z^7)
    b2 = e == 1 ? zpow(3) : zpow(7)
    zmul(ph, zadd(v[1], zmul(b2, v[2])))
end
const QE = [qubit_elem(e, x, z, f) for e in 0:1, x in 0:1, z in 0:1, f in 0:1]

# full <phi_e|P|phi_f> * 2^n (P = i^{|x&z|} X^x Z^z, Hermitian), as product over qubits; sign s
function elem(e, f, x, z)
    acc = zpow(0)
    for j in 0:n-1
        q = QE[((e>>j)&1)+1, ((x>>j)&1)+1, ((z>>j)&1)+1, ((f>>j)&1)+1]
        q == ZERO && return ZERO
        acc = zmul(acc, q)
    end
    acc
end

# ---- F2 helpers
function rref(rows::Vector{Int})
    basis = Int[]
    for r0 in rows
        r = r0
        for b in basis
            if (r >> (63 - leading_zeros(b))) & 1 == 1; r ⊻= b end
        end
        if r != 0
            hb = 63 - leading_zeros(r)
            basis = [((b >> hb) & 1 == 1) ? b ⊻ r : b for b in basis]
            push!(basis, r)
        end
    end
    sort(basis, rev=true)
end
permbits(v, p) = (r = 0; for j in 0:n-1; if (v>>j)&1 == 1; r |= 1 << p[j+1] end end; r)
function allsubspaces(k)  # all k-dim subspaces of F2^n (rref keys) by brute force over rref patterns
    out = Set{Vector{Int}}()
    for pivs in combinations(0:n-1, k)
        pv = sort(collect(pivs), rev=true)
        frees = [[b for b in 0:p-1 if !(b in pv)] for p in pv]
        tot = sum(length.(frees); init=0)
        for msk in 0:(1<<tot)-1
            rows = Int[]; t = 0
            for (p, fr) in zip(pv, frees)
                r = 1 << p
                for b in fr; if (msk >> t) & 1 == 1; r |= 1 << b end; t += 1 end
                push!(rows, r)
            end
            push!(out, sort(rows, rev=true))
        end
    end
    collect(out)
end
const PERMS = [collect(p) .- 1 for p in permutations(1:n)]
canon(S) = minimum([rref([permbits(r, p) for r in S]) for p in PERMS])
inspan(v, rr) = (for r in rr; if (v >> (63 - leading_zeros(r))) & 1 == 1; v ⊻= r end end; v == 0)
function nullspace_f2(cons::Vector{Int}, nv)
    rr = Int[]
    for c0 in cons
        c = c0
        for r in rr; if (c >> (63 - leading_zeros(r))) & 1 == 1; c ⊻= r end end
        if c != 0
            hb = 63 - leading_zeros(c)
            rr = [((r >> hb) & 1 == 1) ? r ⊻ c : r for r in rr]; push!(rr, c)
        end
    end
    piv = Dict(63 - leading_zeros(r) => r for r in rr)
    basis = Int[]
    for f in 0:nv-1
        haskey(piv, f) && continue
        v = 1 << f
        for (p, r) in piv; if isodd(pc(r & v)); v ⊻= 1 << p end end
        push!(basis, v)
    end
    basis
end
perp(AX) = [v for v in 1:MASK if all(iseven(pc(r & v)) for r in AX)]

const H = [e for e in 0:MASK if pc(e) < d || pc(e) > n - d]
const E = [e for e in H if e != 0]

function survives(gens::Vector{Tuple{Int,Int}}, nsig)
    # group elements as (phase power of i, x, z) for x-bits
    els = [(0, 0, 0)]
    for (gx, gz) in gens
        q = pc(gx & gz) % 4
        els = vcat(els, [((p + q + 2*pc(z & gx)) % 4, x ⊻ gx, z ⊻ gz) for (p, x, z) in els])
    end
    # Gram entries for all signs: G^s_{ef} = sum_k chi_s(k) i^{p_k} <phi_e|P_k|phi_f>
    function gram(e, f)
        vals = [zmul(zpow(2*p), elem(e, f, x, z)) for (p, x, z) in els]
        [reduce(zadd, (isodd(pc(s & k-1)) ? zmul(zpow(4), vals[k]) : vals[k] for k in 1:length(vals))) for s in 0:nsig-1]
    end
    G00 = gram(0, 0)
    alive = [g != ZERO for g in G00]
    for e in E
        any(alive) || return false
        G0e = gram(0, e); Gee = gram(e, e)
        for s in 1:nsig
            alive[s] || continue
            alive[s] = zsub(zmul(G00[s], Gee[s]), zmul(G0e[s], zconj(G0e[s]))) == ZERO
        end
    end
    any(alive)
end

function main()
    total = Atomic{Int}(0); surv = Atomic{Int}(0)
    for a in 0:min(m, n)
        reps = unique(canon(S) for S in allsubspaces(a))
        for AX in reps
            P = perp(AX); Pb = rref(P)
            # all (m-a)-dim subspaces of span(P): enumerate rref subsets via subsets of the perp basis span
            BZs = Set{Vector{Int}}()
            spanP = P
            for combo in combinations(spanP, m - a)
                rr = rref(collect(combo)); length(rr) == m - a && push!(BZs, rr)
            end
            m - a == 0 && push!(BZs, Int[])
            for BZ in BZs
                bad = any(e -> all(iseven(pc(r & e)) for r in AX) && !inspan(e, BZ), E)
                bad && continue
                # lifts c_i reduced mod BZ, pairwise commuting (no gauge)
                cons = Int[]
                for r in BZ, i in 0:a-1; push!(cons, 1 << (i*n + 63 - leading_zeros(r))) end
                for i in 0:a-1, j in i+1:a-1
                    msk = 0
                    for t in 0:n-1
                        (AX[i+1] >> t) & 1 == 1 && (msk ⊻= 1 << (j*n + t))
                        (AX[j+1] >> t) & 1 == 1 && (msk ⊻= 1 << (i*n + t))
                    end
                    push!(cons, msk)
                end
                basis = nullspace_f2(cons, a*n)
                L = length(basis)
                @threads for mask in 0:(1<<L)-1
                    cv::Int = 0
                    for bi in 1:L; (mask >> (bi-1)) & 1 == 1 && (cv ⊻= basis[bi]) end
                    gens = Tuple{Int,Int}[(AX[i+1], (cv >> (i*n)) & MASK) for i in 0:a-1]
                    for b in BZ; push!(gens, (0, b)) end
                    atomic_add!(total, 1 << m)
                    survives(gens, 1 << m) && atomic_add!(surv, 1)
                end
            end
        end
        println("a=$a done: signed candidates $(total[]), surviving groups $(surv[])"); flush(stdout)
    end
    println("RESULT n=$n d=$d: signed candidates $(total[]), surviving groups $(surv[])")
end
main()
