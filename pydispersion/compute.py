"""
Analysis functions for dispersion calculations.

These functions help analyze dispersion energies, compare different methods,
and calculate statistics.
"""

import numpy as np
import subprocess
import json
import os
import qcelemental as qcel


def _ext():
    """Return the compiled `dispersion` extension module."""
    import dispersion

    return dispersion


def has_cpp_extensions():
    """
    Check if C++ extensions are available.

    Returns
    -------
    bool
        True if the C++ extensions are available, False otherwise
    """
    try:
        from . import dispersion

        return True
    except ImportError:
        return False


def calculate_dispersion_energy(
    positions,
    cartesians,
    c6s,
    method="S",
    damping_function="BJ",
    params=None,
    monAs=[],
    monBs=[],
):
    """
    Calculate dispersion energy using the specified damping_function.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of atomic numbers
    cartesians : numpy.ndarray
        Array of atomic coordinates
    c6s : numpy.ndarray
        C6 coefficient matrix
    method : str
        Method to use: 'S' for supermolecular. 'I' for intermolecular.
    damping_function : str
        damping_function to use: 'BJ' or 'TT'
    params : numpy.ndarray, optional
        Parameters for dispersion calculation. If None, default parameters are used.
    monAs : list, required for "I"
        Indices of monomer A atoms (used for intermolecular calculations)
    monBs : list, optional for "I"
        Indices of monomer B atoms (used for intermolecular calculations)

    Returns
    -------
    float
        Dispersion energy in ha
    """
    if not has_cpp_extensions():
        raise ImportError(
            "C++ extensions are not available. Cannot calculate dispersion energy."
        )

    from . import dispersion

    if params is None:
        if method == "S":
            if damping_function.upper() == "BJ":
                # SAPT0/aDZ parameters fit for CCSD(T)/CBS IE
                params = np.array([1.0, 0.829861, 0.706055, 1.123903], dtype=np.float64)
            elif damping_function.upper() == "TT":
                # Common TT damping parameters
                params = np.array([1.0, 1.0, -0.33, 4.39], dtype=np.float64)
            else:
                raise ValueError(f"Unknown damping_function: {damping_function}. Use 'BJ' or 'TT'.")
        elif method == "I":
            if damping_function.upper() == "BJ":
                # SAPT(PBE0)/aTZ parameters fit for CCSD(T)/CBS IE
                params = np.array([1.0, 0.89529649, -0.82043591, 0.03264695], dtype=np.float64)
            else:
                raise ValueError(
                    "Only 'BJ' damping is supported for intermolecular calculations."
                )

    if method == "S":
        if damping_function.upper() == "BJ":
            energy = dispersion.disp.disp_2B(positions, cartesians, c6s, params)
        elif damping_function.upper() == "TT":
            energy = dispersion.disp.disp_2B_TT(positions, cartesians, c6s, params)
        else:
            raise ValueError(f"Unknown damping_function: {damping_function}. Use 'BJ' or 'TT'.")
    elif method == "I":
        if len(monAs) == 0 or len(monBs) == 0:
            raise ValueError(
                "monAs and monBs must be provided for intermolecular calculations."
                "These must be atom indices for monomer A and B respectively in the dimer geometry."
            )
        if damping_function.upper() == "BJ":
            monAs = np.array(monAs, dtype=np.int32)
            monBs = np.array(monBs, dtype=np.int32)
            energy = dispersion.disp.disp_2B_BJ_inter(
                positions, cartesians, c6s, monAs, monBs, params
            )
        else:
            raise ValueError(
                "Only 'BJ' damping is supported for intermolecular calculations."
            )

    return energy


def write_xyz_from_np(atom_numbers, carts, outfile="dat.xyz") -> None:
    """
    write_xyz_from_np
    """
    with open(outfile, "w") as f:
        f.write(str(len(carts)) + "\n\n")
        for n, i in enumerate(carts):
            el = str(int(atom_numbers[n]))
            v = "    ".join(["%.16f" % k for k in i])
            line = "%s    %s\n" % (el, v)
            f.write(line)
    return


def qcel_mol_acquire_c6s(
    qcel_mol,
    paramaters={
        "s6": 1.0,
        "s8": 1.61679827,
        "a1": 0.44959224,
        "a2": 3.35743605,
        "s9": 1.0,
    },
    dftd4_bin="dftd4",
):
    """
    If using C6s for downstream tasks does NOT require changing parameters.
    """
    C6s, C8s, _, _, C6s_ATM = acquire_c6s(
        qcel_mol.atomic_numbers,
        qcel_mol.geometry * qcel.constants.bohr2angstroms,  # bohr -> Angstrom for the xyz file
        int(qcel_mol.molecular_charge),
        paramaters=paramaters,
        dftd4_bin=dftd4_bin,
    )
    return C6s, C8s, C6s_ATM


def acquire_c6s(
    atom_numbers: np.array,
    carts: np.array,  # angstroms
    charge: int,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "dftd4",
    paramaters={
        "s6": 1.0,
        "s8": 1.61679827,
        "a1": 0.44959224,
        "a2": 3.35743605,
        "s9": 1.0,
    },
):
    """
    Ensure that dftd4 binary is from compiling git@github.com:Awallace3/dftd4
        - this is used to generate more decimal places on values for c6, c8,
          and pairDisp2
    """

    write_xyz_from_np(
        atom_numbers,
        carts,
        outfile=input_xyz,
    )
    args = [
        dftd4_bin,
        input_xyz,
        "--property",
        "--param",
        str(paramaters["s6"]),
        str(paramaters["s8"]),
        str(paramaters["a1"]),
        str(paramaters["a2"]),
        "--mbdscale",
        f"{paramaters['s9']}",
        "-c",
        str(charge),
        "--pair-resolved",
    ]
    v = subprocess.call(
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    assert v == 0
    output_json = "C_n.json"
    with open(output_json) as f:
        cs = json.load(f)
    C6s = np.array(cs["c6"], dtype=np.float64)
    C8s = np.array(cs["c8"], dtype=np.float64)
    output_json = "pairs.json"
    with open(output_json) as f:
        pairs = json.load(f)
        pairs = np.array(pairs["pairs2"])
    with open(".EDISP", "r") as f:
        e = float(f.read())
    with open("C_n_ATM.json") as f:
        cs = json.load(f)
    C6s_ATM = np.array(cs["c6_ATM"], dtype=np.float64)

    # cleanup
    os.remove(input_xyz)
    os.remove("C_n.json")
    os.remove("pairs.json")
    os.remove(".EDISP")
    os.remove("C_n_ATM.json")
    return C6s, C8s, pairs, e, C6s_ATM


# D4 rational (BJ) two-body parameters for HF (dftd4 assets/parameters.toml).
D4_2B_BJ_DEFAULTS = {"s6": 1.0, "s8": 1.61679827, "a1": 0.44959224, "a2": 3.35743605}


def _fragment_indices(qcel_mol):
    """Atom indices of each fragment, in geometry order."""
    sizes = [len(f) for f in qcel_mol.fragments]
    offsets = np.cumsum([0] + sizes)
    return [
        np.arange(offsets[i], offsets[i + 1], dtype=np.int32)
        for i in range(len(sizes))
    ]


def qcel_2B(
    qcel_mol,
    paramaters=None,
    dftd4_bin="dftd4",
    s6=D4_2B_BJ_DEFAULTS["s6"],
    s8=D4_2B_BJ_DEFAULTS["s8"],
    a1=D4_2B_BJ_DEFAULTS["a1"],
    a2=D4_2B_BJ_DEFAULTS["a2"],
    intermolecular=None,
):
    """
    D4 two-body dispersion energy (hartree) of a qcel molecule with
    Becke-Johnson (rational) damping.

    C6 coefficients are computed once for the whole molecule (EEQ charges for
    the total molecular charge), so with ``intermolecular=False`` the result
    equals dftd4's two-body energy for the same parameters.

    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
    paramaters : dict, optional
        Legacy dict with any of 's6', 's8', 'a1', 'a2'; values override the
        keywords. Other keys (e.g. 's9') are ignored.
    s6, s8, a1, a2 : float
        BJ parameters; defaults are D4 HF.
    intermolecular : bool, optional
        True sums only atom pairs on different fragments (all fragment pairs);
        False sums all atom pairs. Defaults to True when the molecule has more
        than one fragment.
    """
    disp = _ext().disp
    values = {"s6": s6, "s8": s8, "a1": a1, "a2": a2}
    if paramaters:
        values.update({k: v for k, v in paramaters.items() if k in values})
    params = np.array(
        [values["s6"], values["s8"], values["a1"], values["a2"]], dtype=np.float64
    )

    c6s, _, _ = qcel_mol_acquire_c6s(qcel_mol, dftd4_bin=dftd4_bin)
    c6s = np.array(c6s, dtype=np.float64)
    pos = np.array(qcel_mol.atomic_numbers, dtype=np.int32)
    carts = np.array(qcel_mol.geometry, dtype=np.float64)
    frags = _fragment_indices(qcel_mol)
    if intermolecular is None:
        intermolecular = len(frags) > 1
    if not intermolecular:
        return disp.disp_2B(pos, carts, c6s, params)
    if len(frags) < 2:
        raise ValueError("intermolecular=True requires at least two fragments.")
    return sum(
        disp.disp_2B_BJ_inter(pos, carts, c6s, frags[i], frags[j], params)
        for i in range(len(frags))
        for j in range(i + 1, len(frags))
    )


# D4 rational-damping parameters for HF (dftd4 assets/parameters.toml), used
# with Chai-Head-Gordon (CHG) zero damping of the ATM term.
ATM_CHG_DEFAULTS = {"a1": 0.44959224, "a2": 3.35743605, "s9": 1.0}
# Tang-Toennies (TT) damping of the ATM term, beta_ij = b1 * (r_i + r_j) + b2,
# from Xie, Glick, Sherrill, J. Chem. Phys. 158, 094110 (2023), Eq. 7.
ATM_TT_DEFAULTS = {"b1": -0.31, "b2": 3.43, "s9": 1.0}


def vdw_radii_table(overrides=None):
    """
    Per-element vdW radii (bohr) used by TT damping, indexed by atomic number.

    The default table is libmbd's Tkatchenko-Scheffler free-atom R_vdw for
    Z = 1-102 with H, C, N, O replaced by 2.63, 3.34, 3.18, 3.09 bohr.

    Parameters
    ----------
    overrides : dict, optional
        {atomic_number: radius_in_bohr} entries replacing table values, e.g.
        {8: 3.07} for the oxygen radius quoted by Xie et al. (2023).

    Returns
    -------
    numpy.ndarray
    """
    table = np.array(_ext().disp.default_vdw_radii(), dtype=np.float64)
    for z, r in (overrides or {}).items():
        z = int(z)
        if z >= len(table):
            table = np.concatenate([table, np.zeros(z + 1 - len(table))])
        table[z] = float(r)
    return table


def _resolve_vdw_radii(vdw_radii):
    """None -> [] (C++ default table); dict -> overrides; array -> full table."""
    if vdw_radii is None:
        return []
    if isinstance(vdw_radii, dict):
        return vdw_radii_table(vdw_radii).tolist()
    return np.asarray(vdw_radii, dtype=np.float64).tolist()


def calculate_atm_energy(
    positions,
    cartesians,
    c6s_atm,
    damping_function="TT",
    b1=ATM_TT_DEFAULTS["b1"],
    b2=ATM_TT_DEFAULTS["b2"],
    a1=ATM_CHG_DEFAULTS["a1"],
    a2=ATM_CHG_DEFAULTS["a2"],
    s9=1.0,
    vdw_radii=None,
):
    """
    Supermolecular ATM three-body dispersion energy (hartree).

    Parameters
    ----------
    positions : numpy.ndarray
        Atomic numbers.
    cartesians : numpy.ndarray
        Coordinates in bohr.
    c6s_atm : numpy.ndarray
        Charge-neutral D4 C6 matrix (third return of qcel_mol_acquire_c6s).
    damping_function : str
        'TT' (product of pairwise Tang-Toennies f6) or 'CHG' (D4 zero damping).
    b1, b2 : float
        TT parameters, beta_ij = b1 * (r_i + r_j) + b2. Used for 'TT'.
    a1, a2 : float
        Rational-damping radii parameters. Used for 'CHG'.
    s9 : float
        Scale of the C9 coefficients.
    vdw_radii : dict or array-like, optional
        TT only. dict {Z: radius} overrides the default table; an array is a
        full table indexed by atomic number. All radii in bohr.
    """
    disp = _ext().disp
    pos = np.asarray(positions, dtype=np.int32)
    carts = np.asarray(cartesians, dtype=np.float64)
    c6s_atm = np.asarray(c6s_atm, dtype=np.float64)
    damping = damping_function.upper()
    if damping == "TT":
        params = np.array([0.0, 0.0, b1, b2, s9], dtype=np.float64)
        return disp.disp_ATM_TT(pos, carts, c6s_atm, params, _resolve_vdw_radii(vdw_radii))
    if damping == "CHG":
        if vdw_radii is not None:
            raise ValueError("vdw_radii only applies to TT damping.")
        params = np.array([0.0, 0.0, a1, a2, s9], dtype=np.float64)
        return disp.disp_ATM_CHG(pos, carts, c6s_atm, params)
    raise ValueError(f"Unknown damping_function: {damping_function}. Use 'TT' or 'CHG'.")


def qcel_nambe_ATM(
    qcel_mol,
    paramaters=None,
    dftd4_bin="dftd4",
    damping_function=None,
    b1=ATM_TT_DEFAULTS["b1"],
    b2=ATM_TT_DEFAULTS["b2"],
    a1=ATM_CHG_DEFAULTS["a1"],
    a2=ATM_CHG_DEFAULTS["a2"],
    s9=1.0,
    vdw_radii=None,
):
    """
    Nonadditive three-body ATM dispersion (hartree) of a qcel trimer: the sum
    over triples with one atom on each fragment, using C6s of the full trimer.

    damping_function : 'CHG' (default) or 'TT'.
    b1, b2 : TT parameters, beta_ij = b1 * (r_i + r_j) + b2 (Xie et al. 2023).
    a1, a2 : CHG rational-damping parameters (D4 HF).
    s9 : C9 scale.
    vdw_radii : TT only; dict {Z: radius_bohr} overrides or a full table.
    paramaters : legacy dict. Keys 'a1'/'a2' select CHG and 'b1'/'b2' select TT
        when damping_function is not given; its values override the keywords.
    """
    disp = _ext().disp
    values = {"a1": a1, "a2": a2, "b1": b1, "b2": b2, "s9": s9}
    if paramaters:
        if damping_function is None:
            if "b1" in paramaters:
                damping_function = "TT"
            elif "a1" in paramaters:
                damping_function = "CHG"
            else:
                raise ValueError("Unknown parameters")
        values.update(paramaters)
    damping = (damping_function or "CHG").upper()

    _, _, c6s_atm = qcel_mol_acquire_c6s(qcel_mol, dftd4_bin=dftd4_bin)
    c6s_atm = np.array(c6s_atm, dtype=np.float64)
    frags = _fragment_indices(qcel_mol)
    if len(frags) != 3:
        raise ValueError(f"qcel_nambe_ATM needs 3 fragments, got {len(frags)}.")
    pA, pB, pC = frags
    pos = np.array(qcel_mol.atomic_numbers, dtype=np.int32)
    carts = np.array(qcel_mol.geometry, dtype=np.float64)
    if damping == "TT":
        params = np.array([values["b1"], values["b2"], values["s9"]], dtype=np.float64)
        return disp.disp_ATM_TT_trimer_nambe(
            pos, carts, c6s_atm, pA, pB, pC, params, _resolve_vdw_radii(vdw_radii)
        )
    if damping == "CHG":
        if vdw_radii is not None:
            raise ValueError("vdw_radii only applies to TT damping.")
        params = np.array([values["a1"], values["a2"], values["s9"]], dtype=np.float64)
        return disp.disp_ATM_CHG_trimer_nambe(pos, carts, c6s_atm, pA, pB, pC, params)
    raise ValueError(f"Unknown damping_function: {damping_function}. Use 'TT' or 'CHG'.")
