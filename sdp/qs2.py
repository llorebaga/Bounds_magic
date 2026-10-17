"""Exact arithmetic in Q(sqrt2) and an exact phase-1 simplex (Bland's rule) for feasibility LPs."""
from fractions import Fraction
import sympy as sp


class QS2:
    __slots__ = ("a", "b")          # a + b*sqrt2

    def __init__(self, a=0, b=0):
        self.a = Fraction(a)
        self.b = Fraction(b)

    @staticmethod
    def from_sympy(x):
        x = sp.expand(sp.radsimp(sp.sympify(x)))
        b = x.coeff(sp.sqrt(2))
        a = sp.expand(x - b * sp.sqrt(2))
        assert a.is_rational and sp.Rational(b).is_rational, x
        return QS2(Fraction(int(sp.numer(a)), int(sp.denom(a))), Fraction(int(sp.numer(b)), int(sp.denom(b))))

    def __add__(self, o): return QS2(self.a + o.a, self.b + o.b)
    def __sub__(self, o): return QS2(self.a - o.a, self.b - o.b)
    def __neg__(self): return QS2(-self.a, -self.b)
    def __mul__(self, o): return QS2(self.a * o.a + 2 * self.b * o.b, self.a * o.b + self.b * o.a)

    def __truediv__(self, o):
        den = o.a * o.a - 2 * o.b * o.b
        return QS2((self.a * o.a - 2 * self.b * o.b) / den, (self.b * o.a - self.a * o.b) / den)

    def sign(self):
        a, b = self.a, self.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if (a > 0) == (b > 0):
            return 1 if a > 0 else -1
        # opposite signs: compare a^2 with 2 b^2
        if a * a > 2 * b * b:
            return 1 if a > 0 else -1
        return 1 if b > 0 else -1

    def is_zero(self): return self.a == 0 and self.b == 0
    def __float__(self): return float(self.a) + float(self.b) * 2 ** 0.5
    def __repr__(self): return f"({self.a}+{self.b}*sqrt2)"


ZERO, ONE = QS2(0), QS2(1)


def simplex_feasible(A, rhs):
    """Find x >= 0 with A x = rhs (exact). A: list of rows (lists of QS2). Returns x or None.
    Phase 1 with artificials, Bland's rule (terminates)."""
    m, n = len(A), len(A[0])
    A = [row[:] for row in A]
    rhs = rhs[:]
    for i in range(m):
        if rhs[i].sign() < 0:
            A[i] = [-v for v in A[i]]
            rhs[i] = -rhs[i]
    # tableau with artificials n..n+m-1
    T = [A[i] + [ONE if j == i else ZERO for j in range(m)] + [rhs[i]] for i in range(m)]
    basis = [n + i for i in range(m)]
    N = n + m
    # objective: minimize sum of artificials -> reduced costs
    def reduced_costs():
        cost = [ZERO] * N
        for j in range(n, N):
            cost[j] = ONE
        rc = cost[:]
        for i, bi in enumerate(basis):
            cb = cost[bi]
            if cb.is_zero():
                continue
            for j in range(N):
                rc[j] = rc[j] - cb * T[i][j]
        return rc
    while True:
        rc = reduced_costs()
        enter = next((j for j in range(N) if rc[j].sign() < 0), None)
        if enter is None:
            break
        ratios = [(T[i][-1] / T[i][enter], basis[i], i) for i in range(m) if T[i][enter].sign() > 0]
        if not ratios:
            return None   # unbounded phase 1 cannot happen
        best = min(ratios, key=lambda t: (float(t[0]), t[1]))
        # exact tie-break: smallest ratio exactly, then smallest basis index (Bland)
        cand = [t for t in ratios if (t[0] - best[0]).sign() == 0]
        for t in ratios:
            if (t[0] - best[0]).sign() < 0:
                best = t
        cand = [t for t in ratios if (t[0] - best[0]).sign() == 0]
        _, _, r = min(cand, key=lambda t: t[1])
        piv = T[r][enter]
        T[r] = [v / piv for v in T[r]]
        for i in range(m):
            if i != r and not T[i][enter].is_zero():
                f = T[i][enter]
                T[i] = [vi - f * vr for vi, vr in zip(T[i], T[r])]
        basis[r] = enter
    obj = ZERO
    for i, bi in enumerate(basis):
        if bi >= n:
            obj = obj + T[i][-1]
    if obj.sign() != 0:
        return None
    x = [ZERO] * n
    for i, bi in enumerate(basis):
        if bi < n:
            x[bi] = T[i][-1]
    return x
