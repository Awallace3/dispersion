#include "disp.hpp"
#include "r4r2.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
namespace disp {

double np_array_sum_test(std::vector<double> &v) {
  double sum = 0;
  for (auto &x : v) {
    sum += x;
  }
  return sum;
};

void np_array_multiply_test(std::vector<std::vector<double>> &v, double &a) {
  for (uint64_t i = 0; i < v.size(); i++) {
    for (uint64_t j = 0; j < v[i].size(); j++) {
      v[i][j] *= a;
    }
  }
};

pybind11::array_t<double> add_arrays(pybind11::array_t<double> input1,
                                     pybind11::array_t<double> input2) {
  auto buf1 = input1.request(), buf2 = input2.request();
  if (buf1.ndim != 1 || buf2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");
  if (buf1.shape[0] != buf2.shape[0])
    throw std::runtime_error("Input shapes must match");
  auto result = pybind11::array(pybind11::buffer_info(
      nullptr, sizeof(double), pybind11::format_descriptor<double>::value,
      buf1.ndim, {buf1.shape[0]}, {sizeof(double)}));
  auto buf3 = result.request();
  double *ptr1 = (double *)buf1.ptr, *ptr2 = (double *)buf2.ptr,
         *ptr3 = (double *)buf3.ptr;
  for (int64_t idx = 0; idx < buf1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];
  return result;
};

void add_arrays_eigen(py::EigenDRef<MatrixXd> v1, py::EigenDRef<MatrixXd> v2) {
  v1 += v2;
};

double disp_2B(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
               py::EigenDRef<MatrixXd> C6s, Ref<VectorXd> params) {
  int lattice_points = 1;
  double energy = 0;
  int n = pos.size();
  double Q_A, Q_B, rrij, r0ij, dis;
  int el1, el2, i, j, k;
  double s6, s8, a1, a2, de, edisp, t6, t8;
  s6 = params[0];
  s8 = params[1];
  a1 = params[2];
  a2 = params[3];
#pragma omp parallel for shared(C6s, carts, params, pos) private(              \
        i, j, k, el1, el2, Q_A, Q_B, rrij, r0ij, dis, t6, t8, de)              \
    reduction(+ : energy)
  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * r4r2::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * r4r2::r4r2_ls[el2 - 1], 0.5);
      for (k = 0; k < lattice_points; k++) {
        rrij = 3 * Q_A * Q_B;
        r0ij = a1 * pow(rrij, 0.5) + a2;
        dis = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
              (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
              (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));

        t6 = 1 / (pow(dis, 3) + pow(r0ij, 6));
        t8 = 1 / (pow(dis, 4) + pow(r0ij, 8));

        edisp = s6 * t6 + s8 * rrij * t8;

        de = -C6s(i, j) * edisp * 0.5;
        energy += de;
      }
    }
  };
  return energy *= 2;
};

double disp_2B_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                     py::EigenDRef<MatrixXd> C6s, Ref<VectorXi> pA,
                     py::EigenDRef<MatrixXd> cA, py::EigenDRef<MatrixXd> C6s_A,
                     Ref<VectorXi> pB, py::EigenDRef<MatrixXd> cB,
                     py::EigenDRef<MatrixXd> C6s_B, Ref<VectorXd> params) {
  double d, a, b;
  d = disp_2B(pos, carts, C6s, params);
  a = disp_2B(pA, cA, C6s_A, params);
  b = disp_2B(pB, cB, C6s_B, params);
  return d - a - b;
};

/*
for i in range(M_tot):
    el1 = int(pos[i])
    Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5
    for j in range(i + 1):
        el2 = int(pos[j])
        Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5
        c6ij = C6s_ATM[i, j]
        r0ij = a1 * np.sqrt(3 * Q_A * Q_B) + a2
        ri, rj = carts[i, :], carts[j, :]
        r2ij = np.subtract(ri, rj)
        r2ij = np.sum(np.multiply(r2ij, r2ij))
        if np.all(r2ij < 1e-8):
            continue
        for k in range(j + 1):
            el3 = int(pos[k])
            Q_C = (0.5 * el3**0.5 * r4r2_ls[el3 - 1]) ** 0.5
            c6ik = C6s_ATM[i, k]
            c6jk = C6s_ATM[j, k]
            c9 = -s9 * np.sqrt(np.abs(c6ij * c6ik * c6jk))
            r0ik = a1 * np.sqrt(3 * Q_C * Q_A) + a2
            r0jk = a1 * np.sqrt(3 * Q_C * Q_B) + a2
            r0 = r0ij * r0ik * r0jk
            triple = triple_scale(i, j, k)
            for ktr in range(lattice_points):
                rk = carts[k, :]
                r2ik = np.subtract(ri, rk)
                r2ik = np.sum(np.multiply(r2ik, r2ik))
                if np.all(r2ik < 1e-8):
                    continue
                r2jk = np.subtract(rj, rk)
                r2jk = np.sum(np.multiply(r2jk, r2jk))
                if np.all(r2jk < 1e-8):
                    continue
                r2 = r2ij * r2ik * r2jk
                r1 = np.sqrt(r2)
                r3 = r2 * r1
                r5 = r3 * r2

                fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** (alp / 3.0))

                ang = (
                    0.375
                    * (r2ij + r2jk - r2ik)
                    * (r2ij - r2jk + r2ik)
                    * (-r2ij + r2jk + r2ik)
                    / r5
                    + 1.0 / r3
                )

                rr = ang * fdmp

                dE = rr * c9 * triple / 6
                e_ATM -= dE
                energies[j, i] -= dE
                energies[k, i] -= dE
                energies[i, j] -= dE
                energies[k, j] -= dE
                energies[i, k] -= dE
                energies[j, k] -= dE
energy = np.sum(energies)
if not ATM_only:
    energy += e_two_body_disp
 */

double triple_scale(int i, int j, int k) {
  if (i == j) {
    if (i == k) {
      return 1.0 / 6.0;
    } else {
      return 0.5;
    };
  } else {
    if (i != k && j != k) {
      return 1.0;
    } else {
      return 0.5;
    };
  };
};

double disp_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                    py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params) {
  /* int lattice_points = 1; */
  double energy = 0;
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, fdmp, ang; // private
  int el1, el2, el3, i, j, k;        // private
  double a1, a2, s9, alph = 16.0;    // public

  /* s6 = params[0]; */
  /* s8 = params[1]; */
  a1 = params[2];
  a2 = params[3];
  s9 = params[4];
#pragma omp parallel for shared(                                               \
        C6s_ATM, carts, params, pos, a1, a2, s9,                               \
            alph) private(el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik,   \
                              r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,        \
                              dis_ik, triple, c9, fdmp, ang)                   \
    reduction(+ : energy)
  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * r4r2::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * r4r2::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * r4r2::r4r2_ls[el3 - 1], 0.5);
        c9 = -s9 * pow(abs(C6s_ATM(i, j) * C6s_ATM(i, k) * C6s_ATM(j, k)), 0.5);
        r0ik = a1 * pow(3 * Q_A * Q_C, 0.5) + a2;
        r0jk = a1 * pow(3 * Q_B * Q_C, 0.5) + a2;
        r0 = r0ij * r0ik * r0jk;
        triple = triple_scale(i, j, k);
        dis_ik = (carts(i, 0) - carts(k, 0)) * (carts(i, 0) - carts(k, 0)) +
                 (carts(i, 1) - carts(k, 1)) * (carts(i, 1) - carts(k, 1)) +
                 (carts(i, 2) - carts(k, 2)) * (carts(i, 2) - carts(k, 2));
        dis_jk = (carts(j, 0) - carts(k, 0)) * (carts(j, 0) - carts(k, 0)) +
                 (carts(j, 1) - carts(k, 1)) * (carts(j, 1) - carts(k, 1)) +
                 (carts(j, 2) - carts(k, 2)) * (carts(j, 2) - carts(k, 2));
        r2 = dis_ij * dis_ik * dis_jk;
        if (r2 < 1e-8) {
          std::cout << "r2 too small" << std::endl;
          continue;
        };
        r1 = pow(r2, 0.5);
        r3 = r2 * r1;
        r5 = r3 * r2;
        fdmp = 1.0 / (1.0 + 6.0 * pow(r0 / r1, alph / 3.0));
        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);
        energy -= 6 * (ang * fdmp * c9 * triple / 6.0);
      };
    };
  };
  return energy;
};

double disp_ATM_CHG_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                          py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                          py::EigenDRef<MatrixXd> cA,
                          py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                          py::EigenDRef<MatrixXd> cB,
                          py::EigenDRef<MatrixXd> C6s_ATM_B,
                          Ref<VectorXd> params) {
  double d, a, b;
  d = disp_ATM_CHG(pos, carts, C6s_ATM, params);
  a = disp_ATM_CHG(pA, cA, C6s_ATM_A, params);
  b = disp_ATM_CHG(pB, cB, C6s_ATM_B, params);
  return d - a - b;
};

double disp_2B_BJ_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                          py::EigenDRef<MatrixXd> C6s,
                          py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                          py::EigenDRef<MatrixXd> cA,
                          py::EigenDRef<MatrixXd> C6s_A,
                          py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                          py::EigenDRef<MatrixXd> cB,
                          py::EigenDRef<MatrixXd> C6s_B,
                          py::EigenDRef<MatrixXd> C6s_ATM_B,
                          Ref<VectorXd> params_2B, Ref<VectorXd> params_ATM) {
  double energy = 0;
  energy +=
      disp_2B_dimer(pos, carts, C6s, pA, cA, C6s_A, pB, cB, C6s_B, params_2B);

  if (params_ATM.size() >= 4 && params_ATM[4] != 0.0) {
    // checking to see if ATM is disabled
    energy += disp_ATM_CHG_dimer(pos, carts, C6s_ATM, pA, cA, C6s_ATM_A, pB, cB,
                                 C6s_ATM_B, params_ATM);
  }
  return energy;
};
} // namespace disp
