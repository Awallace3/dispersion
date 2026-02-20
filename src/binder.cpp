#include "disp.hpp"
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) { return i + j; }

int sum_thread_ids() {
  int sum = 0;
#pragma omp parallel shared(sum)
  {
#pragma omp critical
    sum += omp_get_thread_num();
  }
  return sum;
}

PYBIND11_MODULE(dispersion, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: dispersion

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
  using namespace pybind11::literals;

  m.def("add", &add, R"pbdoc(
        Add two numbers

        Example

    )pbdoc");

  m.def("omp_get_max_threads", &omp_get_max_threads, R"pbdoc(
        Get max omp threads
    )pbdoc");

  m.def("omp_set_num_threads", &omp_set_num_threads, R"pbdoc(
        Set max omp threads
    )pbdoc");

  m.def("sum_thread_ids", &sum_thread_ids, R"pbdoc(
        Sum thread ids
    )pbdoc");

  auto m_d = m.def_submodule("disp", "dispersion submodule");
  m_d.def("factorial", &disp::factorial, R"pbdoc(
        Calculate factorial of a number
        )pbdoc",
          py::arg("n"));
  m_d.def("f_n_TT", &disp::f_n_TT, R"pbdoc(
        calculate f_n for TT damping
        )pbdoc",
          py::arg("b_ij"), py::arg("R_ij"), py::arg("n"));
  m_d.def("f_n_TT_summation", &disp::f_n_TT_summation, R"pbdoc(
        calculate f_n for TT damping summation
        )pbdoc",
          py::arg("R_b_ij"), py::arg("n"));
  m_d.def("np_array_sum_test", &disp::np_array_sum_test, R"pbdoc(
        Add all number in a np.array

        )pbdoc",
          py::arg("v"));

  m_d.def("np_array_multiply_test", &disp::np_array_multiply_test, R"pbdoc(
        multiply np.array by a number

        )pbdoc",
          py::arg("v"), py::arg("n"));

  m_d.def("add_arrays", &disp::add_arrays, R"pbdoc(
        add arrays
        )pbdoc",
          py::arg("input1"), py::arg("input2"));

  m_d.def("add_arrays_eigen", &disp::add_arrays_eigen, R"pbdoc(
        add arrays
        )pbdoc",
          py::arg("input1"), py::arg("input2"));

  m_d.def("disp_2B", &disp::disp_2B, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("params"));

  m_d.def("disp_2B_XDM", &disp::disp_2B_XDM, R"pbdoc(
        calculate XDM dispersion energy from positions, cartesians, C6s, C8s, C10s, RCs, and params
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C8s"),
          py::arg("C10s"), py::arg("RCs"), py::arg("params"));

  m_d.def("disp_2B_dimer", &disp::disp_2B_dimer, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers with C6s and C8s
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("pA"),
          py::arg("cA"), py::arg("C6s_A"), py::arg("pB"), py::arg("cB"),
          py::arg("C6s_B"), py::arg("params"));

  m_d.def("disp_2B_dimer_XDM", &disp::disp_2B_dimer_XDM, R"pbdoc(
        calculate XDM dispersion energy from positions, cartesians, C6s, C8s,
          C10s and params for a dimer broken into two monomers with C6s, C8s,
          and C10s
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C8s"),
          py::arg("C10s"), py::arg("RCs"), py::arg("pA"), py::arg("cA"),
          py::arg("C6s_A"), py::arg("C8s_A"), py::arg("C10s_A"),
          py::arg("RCs_A"), py::arg("pB"), py::arg("cB"), py::arg("C6s_B"),
          py::arg("C8s_B"), py::arg("C10s_B"), py::arg("RCs_B"),
          py::arg("params"));

  m_d.def("disp_2B_NO_DAMPING", &disp::disp_2B_NO_DAMPING, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("params"));

  m_d.def("disp_2B_dimer_NO_DAMPING", &disp::disp_2B_dimer_NO_DAMPING, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers with C6s and C8s
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("pA"),
          py::arg("cA"), py::arg("C6s_A"), py::arg("pB"), py::arg("cB"),
          py::arg("C6s_B"), py::arg("params"));

  m_d.def("disp_2B_dimer_C6", &disp::disp_2B_dimer_C6, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers with only C6s
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("pA"),
          py::arg("cA"), py::arg("C6s_A"), py::arg("pB"), py::arg("cB"),
          py::arg("C6s_B"), py::arg("params"));

  m_d.def("disp_2B_BJ_inter", &disp::disp_2B_BJ_inter, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers with only dimer C6s
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("monAs"),
          py::arg("monBs"), py::arg("params"));

  m_d.def("disp_2B_inter_NO_DAMPING", &disp::disp_2B_inter_NO_DAMPING, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers with only dimer C6s without damping
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("monAs"),
          py::arg("monBs"), py::arg("params"));

  m_d.def("disp_2B_TT_inter", &disp::disp_2B_TT_inter, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers with only dimer C6s
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("monAs"),
          py::arg("monBs"), py::arg("params"));

  m_d.def("vals_for_SR", &disp::vals_for_SR, R"pbdoc(
        calculate values for SR
        size(vals) = np.zeros((int(N * (N - 1) * (N - 2) / 6), 5))
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("params"),
          py::arg("vals"));

  m_d.def("disp_SR_1", &disp::disp_SR_1, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"));

  m_d.def("disp_SR_2", &disp::disp_SR_2, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"));

  m_d.def("disp_SR_3", &disp::disp_SR_3, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"));

  m_d.def("disp_SR_4", &disp::disp_SR_4, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"));

  m_d.def("disp_SR_4_vals", &disp::disp_SR_4_vals, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"), py::arg("vals"));

  m_d.def("disp_SR_5_vals", &disp::disp_SR_5_vals, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"), py::arg("vals"));

  m_d.def("disp_SR_6_vals", &disp::disp_SR_6_vals, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"), py::arg("vals"));

  m_d.def("disp_SR_7_vals", &disp::disp_SR_7_vals, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"), py::arg("vals"));

  m_d.def("disp_SR_8_vals", &disp::disp_SR_7_vals, R"pbdoc(
        Evaluate SR values
        NOTE: dynamically chaning with change of vals_for_SR

        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"),
          py::arg("params_ATM"), py::arg("vals"));

  m_d.def("disp_ATM_CHG", &disp::disp_ATM_CHG, R"pbdoc(
        calculate -D4 ATM Chair and Head-Gordon (CHG) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s_ATM"),
          py::arg("params"));

  m_d.def("disp_ATM_CHG_dimer", &disp::disp_ATM_CHG_dimer, R"pbdoc(
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s_ATM"), py::arg("pA"),
          py::arg("cA"), py::arg("C6s_ATM_A"), py::arg("pB"), py::arg("cB"),
          py::arg("C6s_ATM_B"), py::arg("params"));

  m_d.def("disp_ATM_CHG_trimer_nambe", &disp::disp_ATM_CHG_trimer_nambe,
          R"pbdoc(
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s_ATM"), py::arg("pA"),
          py::arg("pB"), py::arg("pC"), py::arg("params"));

  m_d.def("disp_ATM_TT_trimer_nambe", &disp::disp_ATM_TT_trimer_nambe,
          R"pbdoc(
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s_ATM"), py::arg("pA"),
          py::arg("pB"), py::arg("pC"), py::arg("params"));

  m_d.def("disp_2B_BJ_ATM_CHG", &disp::disp_2B_BJ_ATM_CHG, R"pbdoc(
        calculate -D4 2Body (BJ) ATM (CHG) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C6s_ATM"),
          py::arg("pA"), py::arg("cA"), py::arg("C6s_A"), py::arg("C6s_ATM_A"),
          py::arg("pB"), py::arg("cB"), py::arg("C6s_B"), py::arg("C6s_ATM_B"),
          py::arg("params_2B"), py::arg("params_ATM"));

  m_d.def("disp_2B_C6_BJ_ATM_CHG", &disp::disp_2B_C6_BJ_ATM_CHG, R"pbdoc(
        calculate -D4 2Body (BJ) ATM (CHG) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C6s_ATM"),
          py::arg("pA"), py::arg("cA"), py::arg("C6s_A"), py::arg("C6s_ATM_A"),
          py::arg("pB"), py::arg("cB"), py::arg("C6s_B"), py::arg("C6s_ATM_B"),
          py::arg("params_2B"), py::arg("params_ATM"));

  m_d.def("disp_ATM_TT", &disp::disp_ATM_TT, R"pbdoc(
        calculate -D4 ATM Chair and Head-Gordon (TT) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s_ATM"),
          py::arg("params"));

  m_d.def("disp_ATM_TT_dimer", &disp::disp_ATM_TT_dimer, R"pbdoc(
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s_ATM"), py::arg("pA"),
          py::arg("cA"), py::arg("C6s_ATM_A"), py::arg("pB"), py::arg("cB"),
          py::arg("C6s_ATM_B"), py::arg("params"));

  m_d.def("disp_2B_BJ_ATM_TT", &disp::disp_2B_BJ_ATM_TT, R"pbdoc(
        calculate -D4 2Body (BJ) ATM (TT) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C6s_ATM"),
          py::arg("pA"), py::arg("cA"), py::arg("C6s_A"), py::arg("C6s_ATM_A"),
          py::arg("pB"), py::arg("cB"), py::arg("C6s_B"), py::arg("C6s_ATM_B"),
          py::arg("params_2B"), py::arg("params_ATM"));

  m_d.def("disp_2B_TT_ATM_TT", &disp::disp_2B_TT_ATM_TT, R"pbdoc(
        calculate -D4 2Body (TT) ATM (TT) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C6s_ATM"),
          py::arg("pA"), py::arg("cA"), py::arg("C6s_A"), py::arg("C6s_ATM_A"),
          py::arg("pB"), py::arg("cB"), py::arg("C6s_B"), py::arg("C6s_ATM_B"),
          py::arg("params_2B"), py::arg("params_ATM"));

  m_d.def("disp_2B_TT_ATM_CHG", &disp::disp_2B_TT_ATM_CHG, R"pbdoc(
        calculate -D4 2Body (TT) ATM (CHG) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C6s_ATM"),
          py::arg("pA"), py::arg("cA"), py::arg("C6s_A"), py::arg("C6s_ATM_A"),
          py::arg("pB"), py::arg("cB"), py::arg("C6s_B"), py::arg("C6s_ATM_B"),
          py::arg("params_2B"), py::arg("params_ATM"));

  auto m_d3 = m.def_submodule("d3", "D3 dispersion");
  m_d3.def("compute_BJ", &d3::compute_BJ, R"pbdoc(
        calculate -D3(BJ) dispersion energy from positions, cartesians, C6s, and params
        )pbdoc",
           py::arg("params"), py::arg("d3data"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
