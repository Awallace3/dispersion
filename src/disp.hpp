#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

using namespace Eigen;
namespace py = pybind11;

namespace disp {

double cube(double x);
double sqrt(double x);
double square(double x);
double neg_exp(double x);
double np_array_sum_test(std::vector<double> &v);

void np_array_multiply_test(std::vector<std::vector<double>> &, double &);

pybind11::array_t<double> add_arrays(pybind11::array_t<double> input1,
                                     pybind11::array_t<double> input2);
void add_arrays_eigen(py::EigenDRef<MatrixXd> v1, py::EigenDRef<MatrixXd> v2);

double disp_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                    py::EigenDRef<MatrixXd> C6s, Ref<VectorXd> params);

double disp_2B(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
               py::EigenDRef<MatrixXd> C6s, Ref<VectorXd> params);

double disp_2B_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                     py::EigenDRef<MatrixXd> C6s, Ref<VectorXi> pA,
                     py::EigenDRef<MatrixXd> cA, py::EigenDRef<MatrixXd> C6s_A,
                     Ref<VectorXi> pB, py::EigenDRef<MatrixXd> cB,
                     py::EigenDRef<MatrixXd> C6s_B, Ref<VectorXd> params);

double disp_2B_C6(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
               py::EigenDRef<MatrixXd> C6s, Ref<VectorXd> params);

double disp_2B_dimer_C6(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                     py::EigenDRef<MatrixXd> C6s, Ref<VectorXi> pA,
                     py::EigenDRef<MatrixXd> cA, py::EigenDRef<MatrixXd> C6s_A,
                     Ref<VectorXi> pB, py::EigenDRef<MatrixXd> cB,
                     py::EigenDRef<MatrixXd> C6s_B, Ref<VectorXd> params);

double triple_scale(int i, int j, int k);

double disp_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                    py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params);

double disp_ATM_2(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                  py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params);

/* void vals_for_SR_ATM(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts, */
/*                    py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params, */
/*                    Ref<VectorXd> eABC); */

void vals_for_SR(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params,
                 py::EigenDRef<MatrixXd> vals);

double disp_SR_1(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM);
// vals [[eABC, r0, r1, r2, alph]]
double disp_SR_2(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM);
// vals [[eABC, r0, r1, r2, r2**2 - r2]]
double disp_SR_3(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM);

double disp_SR_4(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                 py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM);

double disp_SR_4_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals);

double disp_SR_5_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals);

double disp_SR_6_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals);

double disp_SR_7_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals);

double disp_SR_8_vals(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                      py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params_ATM,
                      py::EigenDRef<MatrixXd> vals);

double disp_ATM_CHG_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                          py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                          py::EigenDRef<MatrixXd> cA,
                          py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                          py::EigenDRef<MatrixXd> cB,
                          py::EigenDRef<MatrixXd> C6s_ATM_B,
                          Ref<VectorXd> params);

double disp_2B_BJ_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                          py::EigenDRef<MatrixXd> C6s,
                          py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                          py::EigenDRef<MatrixXd> cA,
                          py::EigenDRef<MatrixXd> C6s_A,
                          py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                          py::EigenDRef<MatrixXd> cB,
                          py::EigenDRef<MatrixXd> C6s_B,
                          py::EigenDRef<MatrixXd> C6s_ATM_B,
                          Ref<VectorXd> params, Ref<VectorXd> params_ATM);

double disp_2B_C6_BJ_ATM_CHG(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                          py::EigenDRef<MatrixXd> C6s,
                          py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                          py::EigenDRef<MatrixXd> cA,
                          py::EigenDRef<MatrixXd> C6s_A,
                          py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                          py::EigenDRef<MatrixXd> cB,
                          py::EigenDRef<MatrixXd> C6s_B,
                          py::EigenDRef<MatrixXd> C6s_ATM_B,
                          Ref<VectorXd> params, Ref<VectorXd> params_ATM);
// START Tang-Toennies damping
// https://pubs.aip.org/aip/jcp/article/132/23/234109/71413
//
double f_n_TT_summation(double b_ij, double R_ij);
double f_n_TT_summation(double R_b_ij);
double f_n_TT(double b_ij, double R_ij);

double disp_ATM_TT(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                   py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXd> params);

double disp_ATM_TT_dimer(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                         py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                         py::EigenDRef<MatrixXd> cA,
                         py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                         py::EigenDRef<MatrixXd> cB,
                         py::EigenDRef<MatrixXd> C6s_ATM_B,
                         Ref<VectorXd> params);

double disp_2B_BJ_ATM_TT(Ref<VectorXi> pos, py::EigenDRef<MatrixXd> carts,
                         py::EigenDRef<MatrixXd> C6s,
                         py::EigenDRef<MatrixXd> C6s_ATM, Ref<VectorXi> pA,
                         py::EigenDRef<MatrixXd> cA,
                         py::EigenDRef<MatrixXd> C6s_A,
                         py::EigenDRef<MatrixXd> C6s_ATM_A, Ref<VectorXi> pB,
                         py::EigenDRef<MatrixXd> cB,
                         py::EigenDRef<MatrixXd> C6s_B,
                         py::EigenDRef<MatrixXd> C6s_ATM_B,
                         Ref<VectorXd> params_2B, Ref<VectorXd> params_ATM);
// END Tang-Toennies damping

} // namespace disp
  //

namespace d3 {
double compute_BJ(Ref<VectorXd> params, py::EigenDRef<MatrixXd> d3data);
}
