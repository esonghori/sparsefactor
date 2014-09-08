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
	
	
	if(argc!=12)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./IstaAx inAfile inXfile inYfile m n numoffiles gammaN lambda steps verbose ncpu" << endl;
			cout << "numoffiles == npes or numoffiles == 1" << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	const char* inAfile = argv[1];
	const char* inXfile = argv[2];
	const char* inYfile = argv[3];
	int m = atoi(argv[4]);
	int n = atoi(argv[5]);
	int numoffiles = atoi(argv[6]);
	double gammaN = atof(argv[7]);
	double lambda = atof(argv[8]);
	int steps = atoi(argv[9]);
	int verbose = atoi(argv[10]);
	int ncpu = atoi(argv[11]);
	
	
	assert(numoffiles==npes || numoffiles==1);
	
	Eigen::setNbThreads(ncpu);
	Eigen::initParallel();
	
	timeval t1, t2;
	
	
	int myn = (n + npes -1)/npes;	
		
	MatrixXd A = MatrixXd::Zero(m, myn);
	

	if(numoffiles == 1)
	{
		ifstream fr;
		fr.open(inAfile);
		if(!fr.is_open())
		{
			if(!myrank)
				cout << "File not found: " << inAfile << endl;
				
			MPI_Finalize();
			return -1;
		}
		
		if(!myrank && verbose)
			cout<< "Start reading A from " << inAfile <<endl;
		
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
		sofA << inAfile << "_" << myrank;
		
		
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

	MatrixXd yh_local(m, 1);
	MatrixXd yh(m, 1);
	MatrixXd y(m, 1);
	MatrixXd yres(m, 1);
	MatrixXd x(myn,1); 
	MatrixXd deltax(myn,1);
	MatrixXd xold(myn,1);
	
	ifstream fX;
	fX.open(inXfile);
	if(!fX.is_open())
	{
		cout << myrank << " File not found: " << inXfile << endl;		
		MPI_Finalize();
		return -1;
	}
	if(!myrank && verbose)
		cout<< "Start reading X from " << inXfile <<endl;
	
	for(int i=0;i<n;i++)
	{
		double value;
		fX >> value;
		if(i >= myrank*myn && i < (myrank+1)*myn)
		{
			x(i - myrank*myn, 0) = value;
		}
	}
	fX.close();
	
	
	if(!myrank && verbose)
		cout<< "x(myrank=1) = " << endl << x.transpose() <<endl;
	
	
	xold = x;
	
	if(!myrank)
	{
		ifstream fY;
		fY.open(inYfile);
		if(!fY.is_open())
		{
			cout << myrank << " File not found: " << inYfile << endl;		
			MPI_Finalize();
			return -1;
		}
		if(!myrank && verbose)
			cout<< "Start reading Y from " << inYfile <<endl;
		
		for(int i=0;i<m;i++)
		{
			double value;
			fY >> value;
			y(i, 0) = value;
		}
		fY.close();
	}
	
	
	if(!myrank && verbose)
		cout<< "y(myrank=1) = " << endl << y.transpose() <<endl;
	
	
	double diffX2_local, diffX2;
	double xnorm2_local, xnorm2;

	double tpi = 0;
	
	double gamma = (2.0 * gammaN)/n;
	
	double ynorm = y.norm();	
	double diffY2 = 0;
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(!myrank && verbose)
	{
		cout << "start ista with gamma = " << gamma << " lambda = " << lambda << endl; 
		cout << "iter time diffY2 diffX2" << endl;
	}
	
	
	gettimeofday(&t1, NULL);
	int step;
	for(step = 0;step <steps ; step++)
	{
		yh_local = A*x;
		MPI_Reduce(yh_local.data(), yh.data(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(!myrank)
		{
			yres = yh - y;
			
			if(verbose)
			{
				cout <<"yres" << endl << yres.transpose() << endl;
			}
			
			diffY2 = yres.norm();
			diffY2 *= diffY2;
			
		}
		MPI_Bcast(yres.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		
		deltax = A.transpose()*yres;
		
		
		x = wthresh(x - gamma*deltax, gamma*lambda);
		
		diffX2_local = (xold - x).norm();
		diffX2_local *= diffX2_local;
		MPI_Allreduce(&diffX2_local, &diffX2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if(diffX2 < 1E-5)
		{
			if(!myrank)
			{
				cout << "converge at " << step << endl;
			}
			break;
		}
		else if(diffX2 > 1E10)
		{
			if(!myrank)
			{
				cout << "not converge at " << step << endl;
			}
			break;
		}
		
		xold = x;
		
		gettimeofday(&t2, NULL);
		if(!myrank)
		{
			cout << step << " " <<  getTimeMs(t1,t2) << " " << diffY2 << " " << diffX2 << endl;
		}
		tpi += getTimeMs(t1,t2);
	}
	
	if(!myrank)
	{
		cout << "time(ms) per iter = " << tpi/(step+1) << endl;
	}
	
	MPI_Finalize();
	return 0;
}
