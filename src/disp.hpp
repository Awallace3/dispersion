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
double disp_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
               py::EigenDRef<MatrixXd> C6s, Ref<VectorXd> params);
double disp_2B_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                     py::EigenDRef<MatrixXd> C6s, Ref<VectorXi> pA,
                     py::EigenDRef<MatrixXd> cA, py::EigenDRef<MatrixXd> C6s_A,
                     Ref<VectorXi> pB, py::EigenDRef<MatrixXd> cB,
                     py::EigenDRef<MatrixXd> C6s_B, Ref<VectorXd> params);

double triple_scale(int i, int j, int k);
double disp_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                    py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params);
double
disp_2B_BJ_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                   py::EigenDRef<MatrixXd> C6s, py::EigenDRef<MatrixXd> C6s_ATM,
                   Ref<VectorXi> pA, py::EigenDRef<MatrixXd> cA,
                   py::EigenDRef<MatrixXd> C6s_A,
                   py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                   py::EigenDRef<MatrixXd> cB, py::EigenDRef<MatrixXd> C6s_B,
                   py::EigenDRef<MatrixXd> C6s_ATM_B, Ref<VectorXd> params);
} // namespace disp
