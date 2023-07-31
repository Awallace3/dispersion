#include "disp.hpp"
#include "r4r2.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
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
  for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];
  return result;
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

double disp_2B_dimer(Ref<VectorXi> pos,
                   py::EigenDRef<MatrixXd> carts,
                   py::EigenDRef<MatrixXd> C6s,
                   Ref<VectorXi> pA,
                   py::EigenDRef<MatrixXd> cA,
                   py::EigenDRef<MatrixXd> C6s_A,
                   Ref<VectorXi> pB,
                   py::EigenDRef<MatrixXd> cB,
                   py::EigenDRef<MatrixXd> C6s_B,
                   Ref<VectorXd> params){
    double d, a, b;
    d = disp_2B(pos, carts, C6s, params);
    a = disp_2B(pA, cA, C6s_A, params);
    b = disp_2B(pB, cB, C6s_B, params);
    return d - a - b;
};



} // namespace disp

