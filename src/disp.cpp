#include "disp.hpp"
#include "constants.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

const double hartree_to_kcalmol = 627.5094737775374;

namespace py = pybind11;
namespace disp {

double cube(double x) { return x * x * x; };
double sqrt(double x) { return pow(x, 0.5); };
double square(double x) { return pow(x, 2.0); };
double neg_exp(double x){ return exp(-x);};
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
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
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

void vals_for_SR(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params,
                 py::EigenDRef<MatrixXd> vals) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;   // private
  int el1, el2, el3, i, j, k, c; // private
  double a1, a2, s9;             // public
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params[2];
  a2 = params[3];
  s9 = params[4];
#pragma omp parallel for shared(                                               \
        C6s_ATM, carts, params, pos, a1, a2,                                   \
            s9) private(el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik,     \
                            r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk, dis_ik,  \
                            triple, c9, ang, c)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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
        // TODO: modify fdmp

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k);
        vals(c, 0) = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        vals(c, 1) = r0;
        vals(c, 2) = r1;
        vals(c, 3) = r2;
        vals(c, 4) = pow(r2, 2) - r2;
      };
    };
  };
};

double disp_SR_1(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang; // private
  int el1, el2, el3, i, j, k;  // private
  double a1, a2, s9;           // public

  //
  double energy, x1, x2, x3, x4, x5, x6;
  energy = 0;
  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];

#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2,   \
            r3, r5, dis_ij, dis_jk, dis_ik, triple, c9, ang, x1, x2, x3, x4,   \
            x5, x6) reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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
        // TODO: modify fdmp

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        /* c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k); */
        /* std::cout << i << " " << j << " " << k << " " << c << std::endl; */

        x1 = 6 * (ang * c9 * triple / 6.0);
        x2 = r0;
        x3 = r1;
        x4 = r2;
        x5 = pow(r2, 2) - r2;
        energy +=
            (x1 / exp(((-0.9566564821449741 + x2) * 1.5069192979905495) / x3));
      };
    };
  };
  /* printf("ATM energy: %f\n", energy); */
  return energy;
};

double disp_SR_2(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;    // private
  int el1, el2, el3, i, j, k;     // private
  double a1, a2, s9, alph = 16.0; // public

  //
  double energy, x1, x2, x3, x4, x5, x6;
  energy = 0;
  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];

#pragma omp parallel for shared(                                               \
        C6s_ATM, carts, pos, a1, a2, s9,                                       \
            alph) private(el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik,   \
                              r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,        \
                              dis_ik, triple, c9, ang, x1, x2, x3, x4, x5, x6) \
    reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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
        // TODO: modify fdmp

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        /* c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k); */
        /* std::cout << i << " " << j << " " << k << " " << c << std::endl; */

        x1 = 6 * (ang * c9 * triple / 6.0);
        x2 = r0;
        x3 = r1;
        x4 = r2;
        x5 = alph;
        energy += ((-1.7084029927853925 * x1 / x2) / (x3 + x5));
      };
    };
  };
  /* printf("ATM energy: %f\n", energy); */
  return energy;
};

double disp_SR_3(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang; // private
  int el1, el2, el3, i, j, k;  // private
  double a1, a2, s9;           // public

  //
  double energy, x1, x2, x3, x4, x5, x6;
  energy = 0;
  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];

#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2,   \
            r3, r5, dis_ij, dis_jk, dis_ik, triple, c9, ang, x1, x2, x3, x4,   \
            x5, x6) reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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
        // TODO: modify fdmp

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        /* c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k); */
        /* std::cout << i << " " << j << " " << k << " " << c << std::endl; */

        x1 = 6 * (ang * c9 * triple / 6.0);
        x2 = r0;
        x3 = r1;
        x4 = r2;
        x5 = pow(r2, 2) - r2;
        energy += (x1 / (((((0.976350945464279 + 0.13084075369818157) *
                            pow(x2 * -2.5275096376155783, 2.0)) /
                           x3) -
                          (-0.13677629695911303 + -1.9927572335514208)) /
                         x3));
      };
    };
  };
  /* printf("ATM energy: %f\n", energy); */
  return energy;
};

double disp_SR_4(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;    // private
  int el1, el2, el3, i, j, k;     // private
  double a1, a2, s9, alph = 16.0; // public
  double x1, x2, x3, x4, x5;
  double energy = 0;
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];
#pragma omp parallel for shared(                                               \
        C6s_ATM, carts, pos, a1, a2, s9,                                       \
            alph) private(el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik,   \
                              r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk, x1,    \
                              x2, x3, x4, x5, dis_ik, triple, c9, ang)         \
    reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        x1 = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        x2 = r0;
        x3 = r1;
        x4 = r2;
        x5 = pow(r2, 2) - r2;
        energy += (((-1.7200158457891235 - x3) / (x2 / square(x1))) / x2);
        /* std::cout << x1 << " " << x2 << " " << x3 << " " << x4 << " " <<
         * energy */
        /*           << std::endl; */
      };
    };
  };
  return energy;
};

double disp_SR_4_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;   // private
  int el1, el2, el3, i, j, k, c; // private
  double a1, a2, s9;             // public
  double x1, x2, x3, x4, x5;
  double energy = 0;
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];
#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, c, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1,    \
            r2, r3, r5, dis_ij, dis_jk, dis_ik, x1, x2, x3, x4, x5, triple,    \
            c9, ang) reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k);
        x1 = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        x2 = r0;
        x3 = r1;
        x4 = r2;
        x5 = pow(r2, 2) - r2;
        vals(c, 0) = x1;
        vals(c, 1) = x2;
        vals(c, 2) = x3;
        vals(c, 3) = x4;
        vals(c, 4) = x5;
        /* energy += (((-1.7200158457891235 - x3) / (x2 / square(x1))) / x2); */
        energy +=
            ((square(square((0.20956607320982998 * x1) * -0.9179628249502935) -
                     0.037416030108050835) *
              x3) *
             -0.9179628249502935);
        /* energy += square( */
        /*     ((square((x1 / 1.2154) * square(-0.078618)) * (x3 - x6)) * x1) /
         */
        /*     square(square(1.0557))); */
      };
    };
  };
  return energy;
};

double disp_SR_5_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;   // private
  int el1, el2, el3, i, j, k, c; // private
  double a1, a2, s9;             // public
  double x0, x1, x2, x3, x4, x5;
  double energy = 0;
  double vdw_i, vdw_j, vdw_k, x6, x7, fmp;
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];
#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, c, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1,    \
            r2, r3, r5, dis_ij, dis_jk, dis_ik, x0, x1, x2, x3, x4, x5,        \
            triple, c9, ang, vdw_i, vdw_j, vdw_k, x6, x7, fmp)                 \
    reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    vdw_i = constants::vdw_ls[el1];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      vdw_j = constants::vdw_ls[el2];
      x5 = vdw_i + vdw_j; // b_IJ = -0.33 (D_IJ) + 4.39
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        vdw_k = constants::vdw_ls[el3];
        x6 = vdw_i + vdw_k; // b_IJ = -0.33 (D_IJ) + 4.39
        x7 = vdw_k + vdw_j; // b_IJ = -0.33 (D_IJ) + 4.39
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k);
        x0 = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        x1 = r0;
        x2 = r1;
        x3 = r2;
        x4 = pow(r2, 2) - r2;
        vals(c, 0) = x0;
        vals(c, 1) = x1;
        vals(c, 2) = x2;
        vals(c, 3) = x3;
        vals(c, 4) = x4;
        /* vals(c, 5) = x5; */
        /* vals(c, 6) = x6; */
        /* vals(c, 7) = x7; */
        fmp = (x2 /
               (square((x3 - ((x1 + 0.3021) + (x1 + 0.12065))) + 0.022277) *
                -0.4173));
        /* fmp = -3.5172e-05; */
        /* fmp = exp((square((square(square((x6 - x5) + x5)) - x5) - */
        /*                   (x3 + 0.3016633254702889)) * */
        /*            -0.9499923155797614) / */
        /*           sqrt(0.07506695261949953)); */
        energy += x0 * fmp;
      };
    };
  };
  return energy;
};

double disp_SR_6_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;   // private
  int el1, el2, el3, i, j, k, c; // private
  double a1, a2, s9;             // public
  double x0, x1, x2, x3, x4, x5;
  double energy = 0;
  double vdw_i, vdw_j, vdw_k, x6, x7, fmp;
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];
#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, c, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1,    \
            r2, r3, r5, dis_ij, dis_jk, dis_ik, x0, x1, x2, x3, x4, x5,        \
            triple, c9, ang, vdw_i, vdw_j, vdw_k, x6, x7, fmp)                 \
    reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    vdw_i = constants::vdw_ls[el1];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      vdw_j = constants::vdw_ls[el2];
      x5 = vdw_i + vdw_j; // b_IJ = -0.33 (D_IJ) + 4.39
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        vdw_k = constants::vdw_ls[el3];
        x6 = vdw_i + vdw_k; // b_IJ = -0.33 (D_IJ) + 4.39
        x7 = vdw_k + vdw_j; // b_IJ = -0.33 (D_IJ) + 4.39
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k);
        x0 = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        x1 = r0;
        x2 = r1;
        x3 = r2;
        x4 = pow(r2, 2) - r2;
        vals(c, 0) = x0;
        vals(c, 1) = x1;
        vals(c, 2) = x2;
        vals(c, 3) = x3;
        vals(c, 4) = x4;
        fmp = (square(square(0.20007955972054262) * x1) * x1);
        energy += x0 * fmp;
      };
    };
  };
  return energy;
};

double disp_SR_7_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;   // private
  int el1, el2, el3, i, j, k, c; // private
  double a1, a2, s9;             // public
  double x0, x1, x2, x3, x4, x5, x6;
  double energy = 0;
  double vdw_i, vdw_j, vdw_k, fmp;
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];
#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, c, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1,    \
            r2, r3, r5, dis_ij, dis_jk, dis_ik, x0, x1, x2, x3, x4, x5, x6,        \
            triple, c9, ang, vdw_i, vdw_j, vdw_k, fmp)                 \
    reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    vdw_i = constants::vdw_ls[el1];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      vdw_j = constants::vdw_ls[el2];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        vdw_k = constants::vdw_ls[el3];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k);
        x0 = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        x1 = r0; // x1 = r0ij * r0ik * r0jk, where r0ij = a1 * sqrt{3 * Q_A * Q_B} + a2
        x2 = r1; // x2 = \sqrt{r2} = sqrt{R_ij * R_ik * R_jk}
        x3 = r2; // x3 = r2 = R_ij * R_ik * R_jk
        x4 = pow(r2, 2) - r2; // x4 = r2**2 - r2
        /* x5 = (vdw_i + vdw_j) / dis_ij * (vdw_i + vdw_k) / dis_ik * (vdw_j + vdw_k) / dis_jk; */
        x5 =  dis_ij / (vdw_i + vdw_j) *  dis_ik / (vdw_i + vdw_k)  * dis_jk / (vdw_j + vdw_k);
        x6 =  pow(Q_A * Q_B * Q_C, 0.5);
        vals(c, 0) = x0;
        vals(c, 1) = x1;
        vals(c, 2) = x2;
        vals(c, 3) = x3;
        vals(c, 4) = x4;
        vals(c, 5) = x5;
        vals(c, 6) = x6;
        fmp = ((neg_exp(x4) - x1) / x4);
        /* fmp = (sqrt(neg_exp(x2)) / -0.3455767973882069); */
        energy += x0 * fmp;
      };
    };
  };
  return energy;
};
double disp_SR_8_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals) {
  int n = pos.size();
  double Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk,
      dis_ik, triple, c9, ang;   // private
  int el1, el2, el3, i, j, k, c; // private
  double a1, a2, s9;             // public
  double x0, x1, x2, x3, x4, x5, x6;
  double energy = 0;
  double vdw_i, vdw_j, vdw_k, fmp;
  // size(eABC) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 6))

  a1 = params_ATM[2];
  a2 = params_ATM[3];
  s9 = params_ATM[4];
#pragma omp parallel for shared(C6s_ATM, carts, pos, a1, a2, s9) private(      \
        el1, el2, el3, i, j, k, c, Q_A, Q_B, Q_C, r0ij, r0ik, r0jk, r0, r1,    \
            r2, r3, r5, dis_ij, dis_jk, dis_ik, x0, x1, x2, x3, x4, x5, x6,        \
            triple, c9, ang, vdw_i, vdw_j, vdw_k, fmp)                 \
    reduction(+ : energy)

  for (i = 0; i < n; i++) {
    el1 = pos[i];
    vdw_i = constants::vdw_ls[el1];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      vdw_j = constants::vdw_ls[el2];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        vdw_k = constants::vdw_ls[el3];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);

        c = (i * (i - 1) * (i - 2) / 6 + j * (j - 1) / 2 + k);
        x0 = 6 * (ang * c9 * triple / 6.0) * hartree_to_kcalmol;
        x1 = r0; // x1 = r0ij * r0ik * r0jk, where r0ij = a1 * sqrt{3 * Q_A * Q_B} + a2
        x2 = r1; // x2 = \sqrt{r2} = sqrt{R_ij * R_ik * R_jk}
        x3 = r2; // x3 = r2 = R_ij * R_ik * R_jk
        x4 = pow(r2, 2) - r2; // x4 = r2**2 - r2
        /* x5 = (vdw_i + vdw_j) / dis_ij * (vdw_i + vdw_k) / dis_ik * (vdw_j + vdw_k) / dis_jk; */
        x5 =  dis_ij / (vdw_i + vdw_j) *  dis_ik / (vdw_i + vdw_k)  * dis_jk / (vdw_j + vdw_k);
        x6 =  pow(Q_A * Q_B * Q_C, 0.5);
        vals(c, 0) = x0;
        vals(c, 1) = x1;
        vals(c, 2) = x2;
        vals(c, 3) = x3;
        vals(c, 4) = x4;
        vals(c, 5) = x5;
        vals(c, 6) = x6;
        fmp = (226.45406572276212 / (square(x1) - x3));
        energy += x0 * fmp;
      };
    };
  };
  return energy;
};
double disp_ATM_2(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
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
        C6s_ATM, carts, params, pos, a1, a2,                                   \
            s9) private(el1, el2, el3, i, j, k, Q_A, Q_B, Q_C, r0ij, r0ik,     \
                            r0jk, r0, r1, r2, r3, r5, dis_ij, dis_jk, dis_ik,  \
                            triple, c9, fdmp, ang) reduction(+ : energy)
  for (i = 0; i < n; i++) {
    el1 = pos[i];
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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
        // TODO: modify fdmp
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
    Q_A = pow(0.5 * pow(el1, 0.5) * constants::r4r2_ls[el1 - 1], 0.5);
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      Q_B = pow(0.5 * pow(el2, 0.5) * constants::r4r2_ls[el2 - 1], 0.5);
      r0ij = a1 * pow(3 * Q_A * Q_B, 0.5) + a2;
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        Q_C = pow(0.5 * pow(el3, 0.5) * constants::r4r2_ls[el3 - 1], 0.5);
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

  if (params_ATM.size() >= 4 && params_ATM[params_ATM.size() - 1] != 0.0) {
    // checking to see if ATM is disabled
    energy += disp_ATM_CHG_dimer(pos, carts, C6s_ATM, pA, cA, C6s_ATM_A, pB, cB,
                                 C6s_ATM_B, params_ATM);
  }
  return energy;
};

double factorial(const int n) {
  double f = 1;
  for (int i = 1; i <= n; ++i) {
    f *= i;
  }
  return f;
}
double f6_TT_summation(double b_ij, double R_ij) {
  double v = 0;
  for (int k = 1; k <= 6; k++) {
    v += pow(b_ij * R_ij, k) / factorial(k);
  }
  return v;
};

double f6_TT_summation(double R_b_ij) {
  double v = 0;
  for (int k = 1; k <= 6; k++) {
    v += pow(R_b_ij, k) / factorial(k);
  }
  return v;
};

double f6_TT(double b_ij, double R_ij) {
  double R_b_ij = b_ij * R_ij;
  return 1 - exp(-R_b_ij) * f6_TT_summation(R_b_ij);
}

double disp_ATM_TT(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                   py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params) {
  double energy = 0;
  int n = pos.size();
  double a1, a2, s9;
  double r1, r2, r3, r5, dis_ij, dis_jk, dis_ik, triple, c9, fdmp,
      ang;                    // private
  int el1, el2, el3, i, j, k; // private
  double vdw_i, vdw_j, vdw_k, b_ij, b_ik, b_jk, f6_ij, f6_ik,
      f6_jk;      // private
  a1 = params[2]; // -0.31
  a2 = params[3]; // 3.43
  s9 = params[4];

#pragma omp parallel for shared(C6s_ATM, carts, params, pos) private(          \
        el1, el2, el3, i, j, k, r1, r2, r3, r5, dis_ij, dis_jk, dis_ik,        \
            triple, c9, fdmp, ang) reduction(+ : energy)
  for (i = 0; i < n; i++) {
    el1 = pos[i];
    vdw_i = constants::vdw_ls[el1];
    for (j = 0; j < i; j++) {
      el2 = pos[j];
      vdw_j = constants::vdw_ls[el2];
      b_ij = a1 * (vdw_i + vdw_j) + a2; // b_IJ = -0.33 (D_IJ) + 4.39
      dis_ij = (carts(i, 0) - carts(j, 0)) * (carts(i, 0) - carts(j, 0)) +
               (carts(i, 1) - carts(j, 1)) * (carts(i, 1) - carts(j, 1)) +
               (carts(i, 2) - carts(j, 2)) * (carts(i, 2) - carts(j, 2));
      /* f6_ij = 1 - exp(-b_ij * dis_ij) * f6_TT_summation(b_ij, dis_ij); // */
      f6_ij = f6_TT(b_ij, dis_ij);
      for (k = 0; k < j; k++) {
        el3 = pos[k];
        vdw_k = constants::vdw_ls[el3];
        b_ik = -a1 * (vdw_i + vdw_k) + a2; // b_IJ = -0.33 (D_IJ) + 4.39
        b_jk = -a1 * (vdw_k + vdw_j) + a2; // b_IJ = -0.33 (D_IJ) + 4.39
        c9 = -s9 * pow(abs(C6s_ATM(i, j) * C6s_ATM(i, k) * C6s_ATM(j, k)), 0.5);
        triple = triple_scale(i, j, k);
        dis_ik = (carts(i, 0) - carts(k, 0)) * (carts(i, 0) - carts(k, 0)) +
                 (carts(i, 1) - carts(k, 1)) * (carts(i, 1) - carts(k, 1)) +
                 (carts(i, 2) - carts(k, 2)) * (carts(i, 2) - carts(k, 2));
        dis_jk = (carts(j, 0) - carts(k, 0)) * (carts(j, 0) - carts(k, 0)) +
                 (carts(j, 1) - carts(k, 1)) * (carts(j, 1) - carts(k, 1)) +
                 (carts(j, 2) - carts(k, 2)) * (carts(j, 2) - carts(k, 2));
        /* f6_ik = 1 - exp(-b_ik * dis_ik) * f6_TT_summation(b_ik, dis_ik); //
         */
        /* f6_jk = 1 - exp(-b_jk * dis_jk) * f6_TT_summation(b_jk, dis_jk); //
         */
        f6_ik = f6_TT(b_ik, dis_ik);
        f6_jk = f6_TT(b_jk, dis_jk);
        r2 = dis_ij * dis_ik * dis_jk;
        if (r2 < 1e-8) {
          std::cout << "r2 too small" << std::endl;
          continue;
        };
        r1 = pow(r2, 0.5);
        r3 = r2 * r1;
        r5 = r3 * r2;
        ang = (0.375 * (dis_ij + dis_jk - dis_ik) * (dis_ij - dis_jk + dis_ik) *
                   (-dis_ij + dis_jk + dis_ik) / r5 +
               1.0 / r3);
        fdmp = f6_ij * f6_ik * f6_jk;
        energy -= 6 * (ang * c9 * triple / 6.0);
        energy *= fdmp;
      };
    };
  };
  return energy;
};

double disp_ATM_TT_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                         py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                         py::EigenDRef<MatrixXd> cA,
                         py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                         py::EigenDRef<MatrixXd> cB,
                         py::EigenDRef<MatrixXd> C6s_ATM_B,
                         Ref<VectorXd> params) {
  double d, a, b;
  d = disp_ATM_TT(pos, carts, C6s_ATM, params);
  a = disp_ATM_TT(pA, cA, C6s_ATM_A, params);
  b = disp_ATM_TT(pB, cB, C6s_ATM_B, params);
  return d - a - b;
};

double disp_2B_BJ_ATM_TT(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
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

  if (params_ATM.size() >= 4 && params_ATM[params_ATM.size() - 1] != 0.0) {
    // checking to see if ATM is disabled
    energy += disp_ATM_TT_dimer(pos, carts, C6s_ATM, pA, cA, C6s_ATM_A, pB, cB,
                                C6s_ATM_B, params_ATM);
  }
  return energy;
};

} // namespace disp

namespace d3 {
double compute_BJ(Ref<VectorXd> params, py::EigenDRef<MatrixXd> d3data) {
  int i;
  double R0, energy = 0.0;
#pragma omp parallel for reduction(+ : energy) private(R0, i)
  for (i = 0; i < d3data.rows(); i++) {
    R0 = pow(d3data(i, 5) / d3data(i, 4), 0.5);
    energy += d3data(i, 4) /
              (pow(d3data(i, 2), 6.0) + pow(params(1) * R0 + params(2), 6.0));
    energy += params(0) * d3data(i, 5) /
              (pow(d3data(i, 2), 8.0) + pow(params(1) * R0 + params(2), 8.0));
  };
  return energy;
};
} // namespace d3
