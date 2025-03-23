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
    print(qcel_trimer)
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
    print(f"nambe_ATM: {nambe_ATM}")
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
    print(qcel_trimer)
    C6s, C8s, C6s_ATM = pydispersion.qcel_mol_acquire_c6s(qcel_trimer)
    # Same parameters as Yi in
    # https://pubs.aip.org/aip/jcp/article/158/9/094110/2881313/Assessment-of-three-body-dispersion-models-against
    nambe_ATM = pydispersion.compute.qcel_nambe_ATM(
        qcel_trimer,
        paramaters={
            "b1": -0.31,
            "b2": 3.43,
            "s9": 1.0,
        },
    )
    print(f"nambe_ATM: {nambe_ATM}")
    return

if __name__ == "__main__":
    test_disp_ATM_CHG_trimer_nambe()
    test_disp_ATM_TT_trimer_nambe()
