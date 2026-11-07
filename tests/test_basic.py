import numpy as np
import pydispersion
import qcelemental as qcel


def test_disp_ATM_CHG_trimer_nambe():
    # data from psi4/tests/fisapt-siao1/input.dat
    data = {
        "symbols": np.array([6, 6, 1, 1, 1, 8, 1, 1, 6, 6, 1, 1, 1, 8, 1, 1, 6, 1, 1]),
        "frags": [8, 16],
        "geometry": np.array(
            [
                [2.51268, -0.79503, -0.22006],
                [1.23732, 0.03963, -0.27676],
                [2.46159, -1.62117, -0.94759],
                [2.64341, -1.21642, 0.78902],
                [3.39794, -0.18468, -0.46590],
                [1.26614, 1.11169, 0.70005],
                [2.10603, 1.58188, 0.59592],
                [1.13110, 0.48209, -1.28412],
                [-1.26007, 0.07291, 0.27398],
                [-2.53390, -0.75742, 0.20501],
                [-2.48461, -1.59766, 0.91610],
                [-2.65872, -1.16154, -0.81233],
                [-3.41092, -0.13922, 0.44665],
                [-1.38660, 1.11180, -0.71748],
                [-1.17281, 0.53753, 1.27129],
                [-0.70002, 1.76332, -0.50799],
                [-0.01090, -0.78649, 0.02607],
                [0.17071, -1.41225, 0.91863],
                [-0.19077, -1.46135, -0.82966],
            ]
        ),
    }
    geom = np.hstack((data["symbols"].reshape(-1, 1), data["geometry"]))
    qcel_trimer = qcel.models.Molecule.from_data(geom, frags=data["frags"])
    # print(qcel_trimer)
    C6s, C8s, C6s_ATM = pydispersion.qcel_mol_acquire_c6s(qcel_trimer)
    # Same parameters as Yi in
    # https://pubs.aip.org/aip/jcp/article/158/9/094110/2881313/Assessment-of-three-body-dispersion-models-against
    nambe_ATM = pydispersion.compute.qcel_nambe_ATM(
        qcel_trimer,
        paramaters={
            "a1": 0.44959224,
            "a2": 3.35743605,
            "s9": 1.0,
        },
    )
    # print(f"nambe_ATM: {nambe_ATM}")
    return


def test_disp_ATM_TT_trimer_nambe():
    # data from psi4/tests/fisapt-siao1/input.dat
    data = {
        "symbols": np.array([6, 6, 1, 1, 1, 8, 1, 1, 6, 6, 1, 1, 1, 8, 1, 1, 6, 1, 1]),
        "frags": [8, 16],
        "geometry": np.array(
            [
                [2.51268, -0.79503, -0.22006],
                [1.23732, 0.03963, -0.27676],
                [2.46159, -1.62117, -0.94759],
                [2.64341, -1.21642, 0.78902],
                [3.39794, -0.18468, -0.46590],
                [1.26614, 1.11169, 0.70005],
                [2.10603, 1.58188, 0.59592],
                [1.13110, 0.48209, -1.28412],
                [-1.26007, 0.07291, 0.27398],
                [-2.53390, -0.75742, 0.20501],
                [-2.48461, -1.59766, 0.91610],
                [-2.65872, -1.16154, -0.81233],
                [-3.41092, -0.13922, 0.44665],
                [-1.38660, 1.11180, -0.71748],
                [-1.17281, 0.53753, 1.27129],
                [-0.70002, 1.76332, -0.50799],
                [-0.01090, -0.78649, 0.02607],
                [0.17071, -1.41225, 0.91863],
                [-0.19077, -1.46135, -0.82966],
            ]
        ),
    }
    geom = np.hstack((data["symbols"].reshape(-1, 1), data["geometry"]))
    qcel_trimer = qcel.models.Molecule.from_data(geom, frags=data["frags"])
    # print(qcel_trimer)
    C6s, C8s, C6s_ATM = pydispersion.qcel_mol_acquire_c6s(qcel_trimer)
    # Same parameters as Yi in
    # https://pubs.aip.org/aip/jcp/article/158/9/094110/2881313/Assessment-of-three-body-dispersion-models-against
    nambe_ATM = pydispersion.compute.qcel_nambe_ATM(
        qcel_trimer, damping_function="TT", b1=-0.31, b2=3.43, s9=1.0
    )
    nambe_ATM_legacy = pydispersion.compute.qcel_nambe_ATM(
        qcel_trimer, paramaters={"b1": -0.31, "b2": 3.43, "s9": 1.0}
    )
    assert np.isclose(nambe_ATM, nambe_ATM_legacy, rtol=1e-12, atol=0)
    # print(f"nambe_ATM: {nambe_ATM}")
    return

def _atm_tt_reference(Z, X, c6, triples, b1, b2, s9, vdw):
    """Xie, Glick, Sherrill, JCP 158, 094110 (2023), Eqs. 1, 3, 5-7."""
    import math

    def f6(beta, r):
        x = beta * r
        return 1 - math.exp(-x) * sum(x**k / math.factorial(k) for k in range(7))

    e = 0.0
    for i, j, k in triples:
        rij, rik, rjk = (np.linalg.norm(X[a] - X[b]) for a, b in ((i, j), (i, k), (j, k)))
        ca = np.dot(X[j] - X[i], X[k] - X[i]) / (rij * rik)
        cb = np.dot(X[i] - X[j], X[k] - X[j]) / (rij * rjk)
        cc = np.dot(X[i] - X[k], X[j] - X[k]) / (rik * rjk)
        c9 = s9 * math.sqrt(abs(c6[i, j] * c6[i, k] * c6[j, k]))
        beta = lambda a, b: b1 * (vdw[Z[a]] + vdw[Z[b]]) + b2
        e += (
            f6(beta(i, j), rij) * f6(beta(i, k), rik) * f6(beta(j, k), rjk)
            * c9 * (1 + 3 * ca * cb * cc) / (rij * rik * rjk) ** 3
        )
    return e


def test_disp_ATM_TT_matches_reference():
    # water trimer (bohr)
    Z = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=np.int32)
    X = np.array(
        [
            [0.000, 0.000, 0.000], [1.431, 1.108, 0.000], [-1.431, 1.108, 0.000],
            [5.291, 0.567, 0.378], [4.700, -0.928, 0.378], [3.584, 1.880, 0.378],
            [2.457, 4.913, -0.756], [1.700, 3.300, -0.756], [4.200, 4.500, -0.756],
        ]
    )
    rng = np.random.default_rng(0)
    c6 = rng.uniform(2.0, 15.0, (9, 9))
    c6 = (c6 + c6.T) / 2
    b1, b2, s9 = -0.31, 3.43, 1.0
    vdw = pydispersion.vdw_radii_table()
    frags = [range(0, 3), range(3, 6), range(6, 9)]
    full = [(i, j, k) for i in range(9) for j in range(i) for k in range(j)]
    cross = [(i, j, k) for i in frags[0] for j in frags[1] for k in frags[2]]

    e = pydispersion.calculate_atm_energy(Z, X, c6, damping_function="TT")
    assert np.isclose(e, _atm_tt_reference(Z, X, c6, full, b1, b2, s9, vdw), rtol=1e-12, atol=0)

    import dispersion

    pA, pB, pC = [np.array(f, dtype=np.int32) for f in frags]
    e = dispersion.disp.disp_ATM_TT_trimer_nambe(Z, X, c6, pA, pB, pC, np.array([b1, b2, s9]))
    assert np.isclose(e, _atm_tt_reference(Z, X, c6, cross, b1, b2, s9, vdw), rtol=1e-12, atol=0)

    # per-element override, e.g. the O radius of 3.07 bohr quoted by Xie et al.
    vdw_o = pydispersion.vdw_radii_table({8: 3.07})
    assert vdw_o[8] == 3.07 and vdw[8] == 3.09
    e = pydispersion.calculate_atm_energy(Z, X, c6, damping_function="TT", vdw_radii={8: 3.07})
    assert np.isclose(e, _atm_tt_reference(Z, X, c6, full, b1, b2, s9, vdw_o), rtol=1e-12, atol=0)


def test_qcel_2B():
    # water trimer (Angstrom input)
    qcel_trimer = qcel.models.Molecule.from_data(
        """
        O  0.000  0.000  0.000
        H  0.757  0.587  0.000
        H -0.757  0.587  0.000
        --
        O  2.800  0.300  0.200
        H  3.300 -0.400  0.600
        H  1.900  0.200  0.500
        --
        O  1.300  2.600 -0.400
        H  0.900  1.800 -0.100
        H  2.200  2.700  0.000
        """
    )
    import dispersion

    C6s, _, _ = pydispersion.qcel_mol_acquire_c6s(qcel_trimer)
    pos = np.array(qcel_trimer.atomic_numbers, dtype=np.int32)
    carts = np.array(qcel_trimer.geometry)
    params = np.array([1.0, 1.61679827, 0.44959224, 3.35743605])

    total = pydispersion.qcel_2B(qcel_trimer, intermolecular=False)
    assert np.isclose(total, dispersion.disp.disp_2B(pos, carts, C6s, params), rtol=1e-12, atol=0)

    # intermolecular part = total minus intra-fragment pairs (same C6s)
    intra = 0.0
    for f in range(3):
        ix = np.arange(3 * f, 3 * f + 3)
        intra += dispersion.disp.disp_2B(
            pos[ix], np.ascontiguousarray(carts[ix]), np.ascontiguousarray(C6s[np.ix_(ix, ix)]), params
        )
    inter = pydispersion.qcel_2B(qcel_trimer)
    assert inter < 0
    assert np.isclose(inter, total - intra, rtol=1e-12, atol=0)

    # legacy dict path ignores the unused 's9'
    legacy = pydispersion.qcel_2B(
        qcel_trimer,
        paramaters={"s6": 1.0, "s8": 1.61679827, "a1": 0.44959224, "a2": 3.35743605, "s9": 1.0},
    )
    assert np.isclose(legacy, inter, rtol=1e-12, atol=0)


if __name__ == "__main__":
    test_disp_ATM_CHG_trimer_nambe()
    test_disp_ATM_TT_trimer_nambe()
    test_disp_ATM_TT_matches_reference()
    test_qcel_2B()
