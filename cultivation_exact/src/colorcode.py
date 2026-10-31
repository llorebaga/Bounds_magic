"""Triangular 6.6.6 colour code faces (same geometry as Gidney et al.'s make_color_code, reimplemented)."""


def color_code_faces(base_width):
    def inside(q):
        if q.imag < 0:
            return False
        if q.real * 3 < q.imag * 2 - 1:
            return False
        if q.real * 3 > -3 + base_width * 6 - q.imag * 2:
            return False
        return True
    offs = [2, -1, 1j, -1j, 1 + 1j, 1 - 1j]
    faces = []
    for x in range(0, base_width * 2):
        for y in range(0, base_width * 2):
            if x % 2 != 1 or y % 2 != x // 2 % 2:
                continue
            m = x + 1j * y
            f = [m + d for d in offs if inside(m + d)]
            if len(f) in (4, 6):
                faces.append(f)
    qubits = sorted({q for f in faces for q in f}, key=lambda z: (z.real, z.imag))
    idx = {q: k for k, q in enumerate(qubits)}
    return [sorted(idx[q] for q in f) for f in faces], len(qubits)
