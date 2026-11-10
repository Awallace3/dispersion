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


def test_disp_2B_TT_C8_matches_D4_undamped():
    """With beta large (f_n -> 1), the TT kernels reduce to undamped -D4 sums:
    C8 = 3 C6 Q_A Q_B, Q = sqrt(0.5 sqrt(Z) <r4>/<r2>), both terms attractive."""
    import dispersion

    d = dispersion.disp
    Z = np.array([8, 1, 1, 7, 1, 1, 1], dtype=np.int32)
    X = np.array(
        [
            [0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0],
            [5.6, 0.3, 0.4], [6.3, -1.4, 0.9], [6.4, 1.6, 1.3], [6.1, 0.8, -1.4],
        ]
    )
    rng = np.random.default_rng(1)
    c6 = rng.uniform(2.0, 20.0, (7, 7))
    c6 = (c6 + c6.T) / 2
    A, B = np.arange(3, dtype=np.int32), np.arange(3, 7, dtype=np.int32)
    undamped = np.array([0.0, 50.0])  # b1, b2 -> beta = 50 bohr^-1
    for s6, s8 in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        e_tt = d.disp_2B_TT_inter(Z, X, c6, A, B, np.array([s6, s8, *undamped]))
        e_ref = d.disp_2B_inter_NO_DAMPING(Z, X, c6, A, B, np.array([s6, s8]))
        assert np.isclose(e_tt, e_ref, rtol=1e-10, atol=0), (s6, s8, e_tt, e_ref)
        assert e_tt < 0
        # disp_2B_TT counts each pair twice (j < i, added for i, j and j, i)
        e_tt = d.disp_2B_TT(Z, X, c6, np.array([s6, s8, *undamped]))
        e_ref = d.disp_2B_NO_DAMPING(Z, X, c6, np.array([s6, s8]))
        assert np.isclose(e_tt, 2 * e_ref, rtol=1e-10, atol=0), (s6, s8, e_tt, e_ref)
        assert e_tt < 0


def _xdm_switch_system():
    Z = np.array([8, 1, 1, 7, 1, 1, 1], dtype=np.int32)
    X = np.array(
        [
            [0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0],
            [5.6, 0.3, 0.4], [6.3, -1.4, 0.9], [6.4, 1.6, 1.3], [6.1, 0.8, -1.4],
        ]
    )
    rng = np.random.default_rng(2)
    sym = lambda m: (m + m.T) / 2
    c6 = sym(rng.uniform(2.0, 20.0, (7, 7)))
    c8 = sym(rng.uniform(50.0, 400.0, (7, 7)))
    c10 = sym(rng.uniform(1e3, 1e4, (7, 7)))
    rc = sym(rng.uniform(4.0, 7.0, (7, 7)))
    return Z, X, c6, c8, c10, rc, np.arange(3, dtype=np.int32), np.arange(3, 7, dtype=np.int32)


def _xdm_switch_numpy(form, X, c6, c8, c10, rc, A, B, w, s, c):
    """Reference: damping_survey/grid_survey.py centre "C10/C8", constant width."""
    from scipy.special import erf

    ii, jj = (x.ravel() for x in np.meshgrid(A, B, indexing="ij"))
    R = np.linalg.norm(X[ii] - X[jj], axis=1)
    R0 = s * np.sqrt(c10[ii, jj] / c8[ii, jj])
    f = {0: lambda: R**w / (R**w + R0**w),
         1: lambda: 0.5 * (1 + erf(w / R0 * (R - R0) / np.sqrt(2))),
         2: lambda: 1 / (1 + np.exp(-w / R0 * (R - R0)))}[form]()
    e = 0.0
    for n, C in ((6, c6), (8, c8), (10, c10)):
        core = R**n / (R**n + (c * rc[ii, jj]) ** n) if c else 1.0
        e += np.sum(f * core * C[ii, jj] / R**n)
    return -e


def test_disp_2B_XDM_switch_matches_numpy():
    """Rational (XDM-op), erf and Fermi switches against an independent numpy sum."""
    import dispersion

    d = dispersion.disp
    Z, X, c6, c8, c10, rc, A, B = _xdm_switch_system()
    cases = [(0, 4.95, 0.959, 0.4), (0, 5.0, 0.96, 0.0), (1, 3.54, 0.973, 0.0),
             (1, 3.51, 0.973, 0.4), (2, 6.10, 0.972, 0.4)]
    for form, w, s, c in cases:
        e = d.disp_2B_XDM_switch_inter(Z, X, c6, c8, c10, rc, A, B, form, np.array([w, s, c]))
        ref = _xdm_switch_numpy(form, X, c6, c8, c10, rc, A, B, w, s, c)
        assert np.isclose(e, ref, rtol=1e-12, atol=0), (form, e, ref)
        assert e < 0
    # two params means no core
    e2 = d.disp_2B_XDM_switch_inter(Z, X, c6, c8, c10, rc, A, B, 0, np.array([5.0, 0.96]))
    e3 = d.disp_2B_XDM_switch_inter(Z, X, c6, c8, c10, rc, A, B, 0, np.array([5.0, 0.96, 0.0]))
    assert e2 == e3


def test_disp_2B_XDM_switch_pairs_batched():
    """Batched pair sum equals per-system kernels; form -1 reproduces BJ inter."""
    import dispersion

    d = dispersion.disp
    Z, X, c6, c8, c10, rc, A, B = _xdm_switch_system()
    ii, jj = (x.ravel() for x in np.meshgrid(A, B, indexing="ij"))
    r = np.linalg.norm(X[ii] - X[jj], axis=1)
    # the same dimer twice, second copy weighted by -0.5
    cat = lambda v: np.concatenate([v, v])
    pairs = [cat(r), cat(c6[ii, jj]), cat(c8[ii, jj]), cat(c10[ii, jj]), cat(rc[ii, jj])]
    sys = np.repeat(np.array([0, 1], dtype=np.int32), r.size)
    w = np.concatenate([np.ones(r.size), np.full(r.size, -0.5)])
    for form, params in ((0, [4.95, 0.959, 0.4]), (1, [3.54, 0.973]), (2, [6.1, 0.972, 0.4])):
        params = np.array(params)
        one = d.disp_2B_XDM_switch_inter(Z, X, c6, c8, c10, rc, A, B, form, params)
        got = d.disp_2B_XDM_switch_pairs(form, *pairs, sys, 2, w, params)
        assert np.allclose(got, [one, -0.5 * one], rtol=1e-12, atol=0), (form, got, one)
        got = d.disp_2B_XDM_switch_pairs(form, *pairs, sys, 2, np.array([]), params)
        assert np.allclose(got, [one, one], rtol=1e-12, atol=0)
    bj = np.array([0.9, 0.3])
    e_bj = d.disp_2B_XDM_inter(Z, X, c6, c8, c10, rc, A, B, bj)
    got = d.disp_2B_XDM_switch_pairs(-1, *pairs, sys, 2, np.array([]), bj)
    assert np.allclose(got, [e_bj, e_bj], rtol=1e-12, atol=0)
    assert np.isclose(d.xdm_switch_pair(-1, r[0], c6[0, 3], c8[0, 3], c10[0, 3], rc[0, 3], 0.9, 0.3),
                      d.disp_2B_XDM_switch_pairs(-1, r[:1], *(np.array([m[0, 3]]) for m in (c6, c8, c10, rc)),
                                                 np.zeros(1, dtype=np.int32), 1, np.array([]), bj)[0])


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


def test_qcel_mol_acquire_c6s_units(tmp_path, monkeypatch):
    """qcel geometries are bohr; dftd4 reads Angstrom. The qcel route must give
    the same C6s as calling acquire_c6s with the Angstrom coordinates directly
    (9c7b197 divided by bohr2angstroms, so dftd4 saw a 3.57x larger molecule)."""
    monkeypatch.chdir(tmp_path)  # dftd4 writes its json/xyz files to cwd
    xyz_ang = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.757, 0.587, 0.000],
            [-0.757, 0.587, 0.000],
            [2.800, 0.300, 0.200],
            [3.300, -0.400, 0.600],
            [1.900, 0.200, 0.500],
        ]
    )
    Z = np.array([8, 1, 1, 8, 1, 1])
    mol = qcel.models.Molecule.from_data(
        "\n".join(f"{qcel.periodictable.to_E(z)} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(Z, xyz_ang))
    )
    assert np.allclose(mol.geometry * qcel.constants.bohr2angstroms, xyz_ang, atol=1e-6)

    C6s, C8s, C6s_ATM = pydispersion.qcel_mol_acquire_c6s(mol)
    ref_C6s, ref_C8s, _, _, ref_C6s_ATM = pydispersion.acquire_c6s(Z, xyz_ang, 0)
    assert np.allclose(C6s, ref_C6s, rtol=1e-6, atol=0)
    assert np.allclose(C8s, ref_C8s, rtol=1e-6, atol=0)
    assert np.allclose(C6s_ATM, ref_C6s_ATM, rtol=1e-6, atol=0)

    # sensitivity: the old conversion gives clearly different C6s
    bad_C6s, _, _, _, _ = pydispersion.acquire_c6s(Z, xyz_ang / qcel.constants.bohr2angstroms**2, 0)
    assert not np.allclose(bad_C6s, ref_C6s, rtol=1e-3)
