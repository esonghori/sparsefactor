#!/bin/sh


mkdir -p ../../tmp 

echo "omp"
#usage: mpiexec -n npes ./ompMultiplefile infile_prefix outfile m n lmin lstep lmax kperl epsilon ncpu of QR_batch verbos N
mpiexec -n 2 ../../bin/ompMultiplefile ../../data/a_4x6_2/a_4x6_2 ../../tmp/a_4x6 4 6 2 2 5 1 0.1 1 1 1 1 6


#usage: ./omp infile outfile m n lmin lstep lmax kperl epsilon ncpu of QR_batch verbos N
mpiexec -n 2 ../../bin/omp ../../data/a_4x6_1/a_4x6_1_0 ../../tmp/a_4x6 4 6 2 2 5 1 0.1 1 1 1 1 6


#usage ./adaptive-omperr.out infile outfile m n lmin lstep lmax kperl epsilon ncpu of QR_batch verbos N
mpiexec -n 2 ../../bin/adaptive-omperr ../../data/a_4x6_1/a_4x6_1_0 ../../tmp/a_4x6 4 6 2 2 5 1 0.1 1 1 1 1 6


#usage ./rand-omperr.out infile outfile m n lmin lstep lmax kperl epsilon ncpu of QR_batch verbos N
mpiexec -n 2 ../../bin/rand-omperr ../../data/a_4x6_1/a_4x6_1_0 ../../tmp/a_4x6 4 6 2 2 5 1 0.1 1 1 1 1 6
