#!/bin/sh

echo "DVxS"
#usage: mpiexec -n npes ./ataDVxS inDfile inVfile_prefix m n l numoffiles localD steps verbose ncpu
mpiexec -n 2 ../../bin/ataDVxS ../../data/d ../../data/v_4x6x3_2/v_4x6x3_2 4 6 3 2 1 10 1 1


echo "Ax"
#usage: mpiexec -n npes ./ataAx m n steps verbose ncpu
mpiexec -n 2 ../../bin/ataAx 6 4 10 1 1
