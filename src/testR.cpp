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
	timeval t1, t2;
	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	srand(0);
	
	
	if(argc!=3)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./ata n steps " << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	int n = atoi(argv[1]);
	int steps = atoi(argv[2]);
	
	MatrixXd x = MatrixXd::Random(n, n);
	MatrixXd y = MatrixXd::Random(n, 1);


	double tcomp = 0;
	double tcomm = 0;
	
	
	double tw1, tw2;
	
	
	MPI_Barrier(MPI_COMM_WORLD);
	for(int step = 0;step <steps ; step++)
	{
		if(!myrank)
			cout << step << endl;
			
		MPI_Barrier(MPI_COMM_WORLD);	
		//gettimeofday(&t1, NULL);
		tw1 = MPI_Wtime();
		
		MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
		//gettimeofday(&t2, NULL);
		tw2 = MPI_Wtime();
		tcomm += (tw2 -tw1);//getTimeMs(t1,t2);
	
		
		//gettimeofday(&t1, NULL);
		tw1 = MPI_Wtime();
		
		y = x * y;
			
		//gettimeofday(&t2, NULL);
		tw2 = MPI_Wtime();
		
		tcomp += (tw2 -tw1);//getTimeMs(t1,t2);
		
	}
	
	if(!myrank)
	{
		cout << "tcomm = " << tcomm  << endl;
		cout << "tcomp = " << tcomp  << endl;
		cout << "r = " << tcomm/(tcomp)  << endl;
	}
	
	MPI_Finalize();
	return 0;
}