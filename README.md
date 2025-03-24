# Dispersion

This C++ library uses pybind11 to create functions accessible via python3 for
computing dispersion energies. OpenMP threading is used for parallelization, 
so set the `OMP_NUM_THREADS` environment variable to the number of threads you
want to use.

# Installation
To install this library, clone the repository and run the following commands:

```bash
git submodule init # Initialize submodules for dftd4 and pybind11
git submodule update # acquires submodules
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..
pip install .
```

## Test
After installation, you can run the tests to ensure everything is working correctly:
```bash
cd tests/
python3 ./test_basic.py
cd ..
```

# Objectives
- [x] Compute nambe ATM from qcelemental molecule (for usage, see
`./tests/test_basic.py:test_disp_ATM_CHG_trimer_nambe()`)
- [x] Compute -D4 2-body dispersion energies given dimer coordinates, C6s, and
  parameters
- [X] Compute -D4 2-body and ATM dispersion energies
    - [x] Compute ATM (CHG) dispersion energy
    - [X] Compute ATM (TT) dispersion energy
