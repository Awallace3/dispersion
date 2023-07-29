#include "disp.hpp"
#include <vector>

double disp::np_array_sum_test(std::vector<double> &v) {
    double sum = 0;
    for (auto &x : v) {
        sum += x;
    }
    return sum;
};

void disp::np_array_multiply_test(std::vector<double> &v, double &a) {

    /* std::vector< std::vector<double> >::const_iterator row; */
    /* std::vector<double>::const_iterator col; */
    /* for (row = v.begin(); row != v.end(); row++) { */
    /*     for (col = row->begin(); col != row->end(); col++){ */
    /*         *col *= a; */
    /*  */
    /*     } */
    /* } */
};
