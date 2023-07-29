#include "disp.hpp"
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>

double disp::np_array_sum_test(std::vector<double> &v) {
    double sum = 0;
    for (auto &x : v) {
        sum += x;
    }
    return sum;
};

void disp::np_array_multiply_test(std::vector<std::vector<double>> &v, double &a) {
    for (uint64_t i=0; i < v.size(); i++) {
        for (uint64_t  j=0; j < v[i].size(); j++) {
            v[i][j] *= a;
        }
    }
};
