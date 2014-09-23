#!/bin/sh

echo "Ax"
#usage: mpiexec -n npes ./powerMethodAx infile_prefix m n e_num numoffiles steps verbose ncpu
mpiexec -n 2 ../../bin/powerMethodAx ../../data/a_4x6_2/a_4x6_2 4 6 3 2 100 1 1

echo "DVxS"
#usage: mpiexec -n npes ./powerMethodDVxS inDfile inVfile_prefix m n l e_num numoffiles localD steps verbose ncpu
mpiexec -n 2 /home/am34/GitHub/sparsefactor/bin/powerMethodDVxS /home/am34/GitHub/sparsefactor/data/d /home/am34/GitHub/sparsefactor/data/v_4x6x3_2/v_4x6x3_2 4 6 3 3 2 1 100 1 1
