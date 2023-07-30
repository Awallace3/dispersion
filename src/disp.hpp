#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace disp {
    double np_array_sum_test(std::vector<double> &v);
    void np_array_multiply_test(std::vector<std::vector<double>> &, double &);
    pybind11::array_t<double> add_arrays(pybind11::array_t<double> input1, pybind11::array_t<double> input2);
    double disp_2b(std::vector<int> pos, std::vector<std::vector<double>> carts, std::vector<std::vector<double>> C6s, std::vector<double> params);
}
