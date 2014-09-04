#include <sys/time.h>

#include <iostream>
#include <iomanip>
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
	
	
	if(argc!=9)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./powerMethodAx indicfile m n e_num numoffiles steps verbose ncpu" << endl;
			cout << "numoffiles == npes or numoffiles == 1" << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	int m = atoi(argv[2]);
	int n = atoi(argv[3]);
	int e_num = atoi(argv[4]);
	int numoffiles = atoi(argv[5]);
	int steps = atoi(argv[6]);
	int verbose = atoi(argv[7]);
	int ncpu = atoi(argv[8]);
	
	
	assert(numoffiles==npes || numoffiles==1);
	
	Eigen::setNbThreads(ncpu);
	Eigen::initParallel();
	
	timeval t1, t2;
	
	
	int myn = (n + npes -1)/npes;	
		
	MatrixXd A = MatrixXd::Zero(m, myn);
	

	if(numoffiles == 1)
	{
		ifstream fr;
		fr.open(argv[1]);
		if(!fr.is_open())
		{
			if(!myrank)
				cout << "File not found: " << argv[1] << endl;
				
			MPI_Finalize();
			return -1;
		}
		
		if(!myrank && verbose)
			cout<< "Start reading A from " << argv[1] <<endl;
		
		for(int i=0; i< m; i++)
		{
			for(int j=0; j< n; j++)
			{
				double temp;
				fr >> temp;
				if(j%npes == myrank)
					A(i, j/npes) = temp;
			}
		}

		for(int i=0;i<A.cols();i++)
		{
			if(A.col(i).norm() > 1E-3)
			{
				A.col(i) =  A.col(i) / A.col(i).norm();
			}
		}
		
		fr.close();
	}
	else
	{

		stringstream  sofA;
		sofA << argv[1] << "_" << myrank;
		
		
		ifstream fr;
		fr.open(sofA.str().c_str());
		if(!fr.is_open())
		{
			if(!myrank)
				cout << "File not found: " << sofA.str() << endl;
				
			MPI_Finalize();
			return -1;
		}
		
		if(!myrank && verbose)
			cout<< "Start reading A from " << sofA.str() <<endl;
		
		for(int i=0; i< m; i++)
		{
			for(int j=0; j< myn; j++)
			{
				double temp;
				fr >> temp;
				A(i, j) = temp;
			}
		}
		for(int i=0;i<A.cols();i++)
		{
			if(A.col(i).norm() > 1E-3)
			{
				A.col(i) =  A.col(i) / A.col(i).norm();
			}
		}
		
		fr.close();
	}
	
	
	MatrixXd y_local(m, 1);
	MatrixXd y(m, 1);
	MatrixXd x = MatrixXd::Random(myn,1);
	
	double xnorm2_local, xnorm2, xnorm2_old;

	if(!myrank && verbose)
		cout<< "start iterations" << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	for(int e_i=0;e_i<e_num;e_i++)
	{
		gettimeofday(&t1, NULL);
		int step;
		for(step = 0;step <steps ; step++)
		{
			y_local = A*x;
			MPI_Allreduce(y_local.data(), y.data(), y.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);		
			x = A.transpose()*y;
			xnorm2_local = x.norm();
			xnorm2_local = xnorm2_local*xnorm2_local;
			
			MPI_Allreduce(&xnorm2_local, &xnorm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			
			
			if(abs(xnorm2 - xnorm2_old)/xnorm2 < 1E-5)
			{
				A = A - y * x.transpose()/sqrt(xnorm2);

				break;
			}
			
			x = x / sqrt(xnorm2);
			
			if(!myrank && verbose)
				cout<< "lambda" << e_i << " = " <<  sqrt(xnorm2) << endl;
			
			
			xnorm2_old = xnorm2;
		}
		gettimeofday(&t2, NULL);
		if(!myrank)
			cout << e_i << " " << step << " " << sqrt(xnorm2) << " " << getTimeMs(t1,t2) << endl;
	}

	
	MPI_Finalize();
	return 0;
}
