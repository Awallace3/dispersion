#include "disp.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) { return i + j; }

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
  auto m_d = m.def_submodule("disp", "dispersion submodule");
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

  m_d.def("disp_2B", &disp::disp_ATM_CHG, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("params"));

  m_d.def("disp_2B_dimer", &disp::disp_2B_dimer, R"pbdoc(
        calculate 2-body -D4 dispersion energy from positions, cartesians, C6s, and params
        for a dimer broken into two monomers
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("pA"),
          py::arg("cA"), py::arg("C6s_A"), py::arg("pB"), py::arg("cB"),
          py::arg("C6s_B"), py::arg("params"));

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

  m_d.def("disp_2B_BJ_ATM_CHG", &disp::disp_2B_BJ_ATM_CHG, R"pbdoc(
        calculate -D4 2Body (BJ) ATM (CHG) damping dispersion
        )pbdoc",
          py::arg("pos"), py::arg("carts"), py::arg("C6s"), py::arg("C6s_ATM"),
          py::arg("pA"), py::arg("cA"), py::arg("C6s_A"), py::arg("C6s_ATM_A"),
          py::arg("pB"), py::arg("cB"), py::arg("C6s_B"), py::arg("C6s_ATM_B"),
          py::arg("params"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
