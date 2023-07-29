#!/usr/bin/bash

export numThreads=10
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..
pip install .
# rm compile_commands.json
# ln -s build/compile_commands.json .
