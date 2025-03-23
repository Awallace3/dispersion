#!/usr/bin/bash

export numThreads=20
rm compile_commands.json
# rm -rf build
# mkdir -p build
cd build
# cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..
ln -s build/compile_commands.json .
pip install .
cd tests/
python3 ./test_basic.py
cd ..
