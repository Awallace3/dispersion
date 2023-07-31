#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

using namespace Eigen;
namespace py = pybind11;

namespace disp {
double np_array_sum_test(std::vector<double> &v);
void np_array_multiply_test(std::vector<std::vector<double>> &, double &);
pybind11::array_t<double> add_arrays(pybind11::array_t<double> input1,
                                     pybind11::array_t<double> input2);
double disp_2b(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
               py::EigenDRef<MatrixXd> C6s, Ref<VectorXd> params);
} // namespace disp
