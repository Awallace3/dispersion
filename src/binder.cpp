#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "disp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)




namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

/* namespace disp { */
/*     double np_array_test(std::vector<double> &v) { */
/*         double sum = 0; */
/*         for (auto &x : v) { */
/*             sum += x; */
/*         } */
/*         return sum; */
/*     } */
/* } */

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

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Example

    )pbdoc");
    auto m_d = m.def_submodule("disp", "dispersion submodule");
    m_d.def("np_array_sum_test", &disp::np_array_sum_test, R"pbdoc(
        Add all number in a np.array

        )pbdoc");

    m_d.def("np_array_multiply_test", &disp::np_array_multiply_test, R"pbdoc(
        multiply np.array by a number

        )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
