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
	
	
	if(argc!=11)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./ataDVxS inDfile inVfile m n l numoffiles localD steps verbose ncpu" << endl;
			cout << "numoffiles % npes = 0" << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	int m = atoi(argv[3]);
	int n = atoi(argv[4]);
	int l = atoi(argv[5]);
	int numoffiles = atoi(argv[6]);
	int localD = atoi(argv[7]);
	int steps = atoi(argv[8]);
	int verbose = atoi(argv[9]);
	int ncpu = atoi(argv[10]);
	
	
	assert(numoffiles%npes == 0);
	
	Eigen::setNbThreads(ncpu);
	Eigen::initParallel();
	
	timeval t1, t2;
	
	
	
	int myn_files = (n + numoffiles -1)/numoffiles;
	int myn = (myn_files*numoffiles + npes - 1)/npes;
	
			
	
	MatrixXd D;

	vector<Trip> tripletV;
	SparseMatrixD V(l, myn);
	
	if(!localD || !myrank)
	{
		D = MatrixXd::Zero(m, l);
		ifstream fD;
		fD.open(argv[1]);
		if(!fD.is_open())
		{
			cout << myrank <<" File not found: " << argv[1] << endl;		
			MPI_Finalize();
			return -1;
		}
		
		
		if(!myrank && verbose)
		{
			if(localD)
			{
				cout<< "local D" <<endl;
			}
			cout<< "Start reading D from " << argv[1] <<endl;
		}
		for(int i=0; i< m; i++)
		{
			for(int j=0; j< l; j++)
			{
				fD >> D(i, j);
			}
		}
		fD.close();
	}
	
	for(int i=0;i<numoffiles/npes;i++)
	{
		int fileID = myrank*(numoffiles/npes) + i;
	
		stringstream  sfV;
		sfV << argv[2] << "_" << fileID;	
		ifstream fV;
		fV.open(sfV.str().c_str());
		if(!fV.is_open())
		{
			cout << myrank <<" File not found: " << sfV.str().c_str() << endl;		
			MPI_Finalize();
			return -1;
		}
		
		if(!myrank && verbose)
			cout<< "Start reading V from " << sfV.str().c_str() <<endl;
		
		while(1)
		{
			unsigned int row,col;
			double value;		
			fV >> row >> col >> value;
			
			if(fV.eof())
				break;

			//if(!myrank)
			//	cout << row << " " << col << " " << value << endl;
			
			
			col += myn_files*i; //shift cols
			
			assert(row>=0);
			assert(row<l);
			assert(col>=0);
			assert(col<myn);
			tripletV.push_back(Trip(row,col,value));
		}
		fV.close();
	}	
	
	
	V.setFromTriplets(tripletV.begin(), tripletV.end());
	
	MatrixXd p_local(l, 1);
	MatrixXd p(l, 1);
	MatrixXd q(l, 1);
	MatrixXd y(m, 1);
	MatrixXd x = MatrixXd::Random(myn,1);
	
	double xnorm2_local, xnorm2;

	double tpi = 0;
	
	MPI_Barrier(MPI_COMM_WORLD);
	for(int step = 0;step <steps ; step++)
	{
		gettimeofday(&t1, NULL);
		
		if(localD)
		{
			p_local = V*x;
			MPI_Reduce(p_local.data(), p.data(), l, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if(!myrank)
			{
				y = D*p;
				q = D.transpose()*y;
			}
			MPI_Bcast(q.data(), l, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		else
		{
			p_local = V*x;
			MPI_Allreduce(p_local.data(), p.data(), l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			y = D*p;
			q = D.transpose()*y;
		}	
		
		
		x = V.transpose()*q;
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
