#include "disp.hpp"
#include "r4r2.hpp"
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace disp{

double np_array_sum_test(std::vector<double> &v) {
    double sum = 0;
    for (auto &x : v) {
        sum += x;
    }
    return sum;
};

void np_array_multiply_test(std::vector<std::vector<double>> &v, double &a) {
    for (uint64_t i=0; i < v.size(); i++) {
        for (uint64_t  j=0; j < v[i].size(); j++) {
            v[i][j] *= a;
        }
    }
};

/* pybind11::array_t<double> add_arrays(pybind11::array_t<double> input1, pybind11::array_t<double> input2) { */
/*     auto buf1 = input1.request(), buf2 = input2.request(); */
/*  */
/*     if (buf1.ndim != 1 || buf2.ndim != 1) */
/*         throw std::runtime_error("Number of dimensions must be one"); */
/*  */
/*     if (buf1.shape[0] != buf2.shape[0]) */
/*         throw std::runtime_error("Input shapes must match"); */
/*  */
/*     auto result = pybind11::array(pybind11::buffer_info( */
/*         nullptr, */
/*         sizeof(double), */
/*         pybind11::format_descriptor<double>::value, */
/*         buf1.ndim, */
/*         { buf1.shape[0] }, */
/*         { sizeof(double) } */
/*     )); */
/*  */
/*     auto buf3 = result.request(); */
/*  */
/*     double *ptr1 = (double *) buf1.ptr, */
/*            *ptr2 = (double *) buf2.ptr, */
/*            *ptr3 = (double *) buf3.ptr; */
/*  */
/*     for (size_t idx = 0; idx < buf1.shape[0]; idx++) */
/*         ptr3[idx] = ptr1[idx] + ptr2[idx]; */
/*  */
/*     return result; */

    double disp_2b(std::vector<int> pos, std::vector<std::vector<double>> carts, std::vector<std::vector<double>> C6s, std::vector<double> params){
    double energy = 0;
    int n = pos.size();
    int lattice_points = 1;
    double Q_A, Q_B, rrij, r0ij;
    for (int i=0; i<n; i++){
        r4r2::r4r2_ls

    };

}
}
