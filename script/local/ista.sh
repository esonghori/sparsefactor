#!/bin/sh

echo "DVxS"
mpiexec -n 2 ../../bin/istaDVxS ../../data/d ../../data/v_4x6x3_2/v_4x6x3_2 ../../data/x ../../data/y 4 6 3 2 1 0.1 0.01 20 0 1


echo "Ax"
mpiexec -n 2 ../../bin/istaAx ../../data/a_4x6_2/a_4x6_2 ../../data/x ../../data/y 4 6 2 0.1 0.01 20 0 1

