#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <ctime>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <mpi.h>

using namespace std;
using namespace Eigen;

typedef SparseMatrix<double> SparseMatrixD;
typedef Eigen::Triplet<double> Trip;

double getTimeMs(const timeval t1,const  timeval t2)
{
	double tE = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	tE += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	
	return tE;
}

int main(int argc, char*argv[])
{
	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	srand(0);
	
	
	if(argc!=5)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./ataAx m n steps ncpu" << endl;
			cout << "using random A"  << endl;
			cout << "ncpu for openmp cores, ncpu==1 when using mpi" << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int steps = atoi(argv[3]);
	int ncpu = atoi(argv[4]);
	
	int myn = (n + npes - 1)/npes;
	
	Eigen::setNbThreads(ncpu);
	Eigen::initParallel();
	
	timeval t1, t2;
	

	MatrixXd A = MatrixXd::Random(m, myn);
	
		
	MatrixXd p_local(m, 1);
	MatrixXd p(m, 1);
	MatrixXd x = MatrixXd::Random(myn, 1);
	
	double xnorm2_local, xnorm2;

	double tpi = 0;
	
	MPI_Barrier(MPI_COMM_WORLD);
	for(int step = 0;step <steps ; step++)
	{
		gettimeofday(&t1, NULL);

		p_local = A*x;
		MPI_Allreduce(p_local.data(), p.data(), p.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
		
		x = A.transpose()*p;
		xnorm2_local = x.norm();
		xnorm2_local = xnorm2_local*xnorm2_local;
		
		MPI_Allreduce(&xnorm2_local, &xnorm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		x = x / sqrt(xnorm2);
		
		
		gettimeofday(&t2, NULL);
		tpi += getTimeMs(t1,t2);
	}
	
	if(!myrank)
		cout << "time(ms) per iter = " << tpi/steps << endl;

	
	MPI_Finalize();
	return 0;
}
