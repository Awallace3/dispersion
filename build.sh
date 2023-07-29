#!/usr/bin/bash

export numThreads=10
rm disp
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..
# rm compile_commands.json
# ln -s build/compile_commands.json .
