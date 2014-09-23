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

MatrixXd wthresh(const MatrixXd &v, double t)
{
	MatrixXd ret = v;
	for(int i=0;i<v.rows();i++)
	{
		for(int j=0;j<v.cols();j++)
		{
			if(v(i,j) > t)
			{
				ret(i,j) = v(i,j) - t;
			}
			else if(v(i,j) < -1*t)
			{
				ret(i,j) = v(i,j) + t;
			}
			else
			{
				ret(i,j) = 0;			
			}	
		}
	}
	return ret;
}


int main(int argc, char*argv[])
{
	int npes, myrank;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	srand(0);
	
	
	if(argc!=11)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./IstaAx inAfile_perfix outXfile outYfile outAtYfile m n xsparsity numoffiles verbose ncpu" << endl;
			cout << "numoffiles == npes or numoffiles == 1" << endl;
			cout << "if numoffiles > 1 inAfile_perfix: inAfile_0 .. inAfile_{numoffiles-1}" << endl;
			cout << "ncpu for openmp cores, ncpu==1 when using mpi" << endl;
			
		}
		MPI_Finalize();	
		return -1;
	}
	
	const char* inAfile_perfix = argv[1];
	const char* outXfile = argv[2];
	const char* outYfile = argv[3];
	const char* outAtYfile = argv[4];
	int m = atoi(argv[5]);
	int n = atoi(argv[6]);
	double xsparsity = atof(argv[7]);
	int numoffiles = atoi(argv[8]);
	int verbose = atoi(argv[9]);
	int ncpu = atoi(argv[10]);
	
	
	assert(numoffiles==npes || numoffiles==1);
	
	Eigen::setNbThreads(ncpu);
	Eigen::initParallel();
	
	
	int myn = (n + npes -1)/npes;	
		
	MatrixXd A = MatrixXd::Zero(m, myn);

	if(numoffiles == 1)
	{
		ifstream fr;
		fr.open(inAfile_perfix);
		if(!fr.is_open())
		{
			if(!myrank)
				cout << "File not found: " << inAfile_perfix << endl;
				
			MPI_Finalize();
			return -1;
		}
		
		if(!myrank && verbose)
			cout<< "Start reading A from " << inAfile_perfix <<endl;
		
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
		sofA << inAfile_perfix << "_" << myrank;
		
		
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

	///////////////
	MatrixXd y_local(m, 1);
	MatrixXd y(m, 1);
	MatrixXd x(myn,1); 
	
	for(int i=0;i<myn;i++)
	{
		if((rand()%1000000)/1000000.0 < xsparsity)
		{
			x(i, 0) = 1;
		}		
	}
	
	if(!myrank)
	{
		ofstream foutx;
		foutx.open(outXfile);
		
		if(verbose)
			cout << "start writing to " << outXfile << endl;
		
		foutx << x;
		
		foutx.close();
	}

	y_local = A*x;
	MPI_Reduce(y_local.data(), y.data(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(!myrank)
	{
		ofstream fouty;
		fouty.open(outYfile);
		
		if(verbose)
			cout << "start writing to " << outYfile << endl;
		
		fouty << y;
		
		fouty.close();
		
	}
	
	x = A.transpose()*y;
	
	for(int r=0;r<npes;r++)
	{
		if(myrank == r)
		{
			ofstream foutaty;
			
			if(!myrank)
				foutaty.open(outAtYfile, std::ofstream::out);
			else
				foutaty.open(outAtYfile, std::ofstream::out | std::ofstream::app);
			
			if(verbose)
				cout << "start writing to " << outAtYfile << endl;
			
			foutaty << x;
			
			foutaty.close();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
		

	
	MPI_Finalize();
	return 0;
}
