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
        qcel_mol.geometry / qcel.constants.bohr2angstroms,
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


def qcel_2B(
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
    if 'a1' and 'a2' then mCHG damping, elif 'b1' and 'b2' TT damping
    """
    import dispersion

    c6s, c8s, _ = qcel_mol_acquire_c6s(
        qcel_mol,
        dftd4_bin=dftd4_bin,
    )
    pA = np.array(
        [i for i in range(len(qcel_mol.get_fragment(0).atomic_numbers))], dtype=np.int32
    )
    pB = np.array(
        [
            i
            for i in range(
                len(pA), len(pA) + len(qcel_mol.get_fragment(1).atomic_numbers)
            )
        ],
        dtype=np.int32,
    )
    pC = np.array(
        [
            i
            for i in range(
                len(pA) + len(pB),
                len(pA) + len(pB) + len(qcel_mol.get_fragment(2).atomic_numbers),
            )
        ],
        dtype=np.int32,
    )
    params = np.zeros(3, dtype=np.float64)
    if "a1" in paramaters:
        params[0] = paramaters["a1"]
        params[1] = paramaters["a2"]
        params[2] = paramaters["s9"]
        disp_func = dispersion.disp.disp_ATM_CHG_trimer_nambe
    elif "b1" in paramaters:
        params[0] = paramaters["b1"]
        params[1] = paramaters["b2"]
        params[2] = paramaters["s9"]
        disp_func = dispersion.disp.disp_ATM_TT_trimer_nambe
    else:
        raise ValueError("Unknown parameters")
    nambe_ATM = disp_func(
        np.array(qcel_mol.atomic_numbers, dtype=np.int32),
        np.array(qcel_mol.geometry, dtype=np.float64),
        c6s_atm,
        pA,
        pB,
        pC,
        params,
    )
    return nambe_ATM


def qcel_nambe_ATM(
    qcel_mol,
    paramaters={
        "a1": 0.44959224,
        "a2": 3.35743605,
        "s9": 1.0,
    },
    dftd4_bin="dftd4",
):
    """
    if 'a1' and 'a2' then mCHG damping, elif 'b1' and 'b2' TT damping
    """
    import dispersion

    c6s, c8s, c6s_atm = qcel_mol_acquire_c6s(
        qcel_mol,
        dftd4_bin=dftd4_bin,
    )
    c6s_atm = np.array(c6s_atm, dtype=np.float64)
    pA = np.array(
        [i for i in range(len(qcel_mol.get_fragment(0).atomic_numbers))], dtype=np.int32
    )
    pB = np.array(
        [
            i
            for i in range(
                len(pA), len(pA) + len(qcel_mol.get_fragment(1).atomic_numbers)
            )
        ],
        dtype=np.int32,
    )
    pC = np.array(
        [
            i
            for i in range(
                len(pA) + len(pB),
                len(pA) + len(pB) + len(qcel_mol.get_fragment(2).atomic_numbers),
            )
        ],
        dtype=np.int32,
    )
    params = np.zeros(3, dtype=np.float64)
    if "a1" in paramaters:
        params[0] = paramaters["a1"]
        params[1] = paramaters["a2"]
        params[2] = paramaters["s9"]
        disp_func = dispersion.disp.disp_ATM_CHG_trimer_nambe
    elif "b1" in paramaters:
        params[0] = paramaters["b1"]
        params[1] = paramaters["b2"]
        params[2] = paramaters["s9"]
        disp_func = dispersion.disp.disp_ATM_TT_trimer_nambe
    else:
        raise ValueError("Unknown parameters")
    nambe_ATM = disp_func(
        np.array(qcel_mol.atomic_numbers, dtype=np.int32),
        np.array(qcel_mol.geometry, dtype=np.float64),
        c6s_atm,
        pA,
        pB,
        pC,
        params,
    )
    return nambe_ATM
