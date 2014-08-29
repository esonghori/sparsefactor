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
	
	
	if(argc!=12)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./ata inDfile inVfile m n l e_num numoffiles localD steps verbose ncpu" << endl;
			cout << "numoffiles % npes = 0" << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	int m = atoi(argv[3]);
	int n = atoi(argv[4]);
	int l = atoi(argv[5]);
	int e_num = atoi(argv[6]);
	int numoffiles = atoi(argv[7]);
	int localD = atoi(argv[8]);
	int steps = atoi(argv[9]);
	int verbose = atoi(argv[10]);
	int ncpu = atoi(argv[11]);
	
	
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
	
	
	
	MatrixXd W(m, 0);
	MatrixXd Q(0, myn);
	
	
	MatrixXd p_local(l, 1);
	MatrixXd p(l, 1);
	MatrixXd q(l, 1);
	MatrixXd y(m, 1);
	MatrixXd x = MatrixXd::Random(myn,1);
	MatrixXd WQx; 
    MatrixXd Wty;
	MatrixXd Qx_local;
	MatrixXd Qx;
	
	
	double xnorm2_local, xnorm2, xnorm2_old;

	double tpi = 0;
	
	MPI_Barrier(MPI_COMM_WORLD);
	for(int e_i=0;e_i<e_num;e_i++)
	{
		gettimeofday(&t1, NULL);
		int step;
		
		Qx_local = MatrixXd::Zero(e_i,1);
		Qx = MatrixXd::Zero(e_i,1);
		Wty = MatrixXd::Zero(e_i,1);
		
		for(step = 0;step <steps ; step++)
		{
			gettimeofday(&t1, NULL);

			p_local = V*x;
			Qx_local = Q*x;
			MPI_Reduce(p_local.data(), p.data(), p.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(Qx_local.data(), Qx.data(), Qx.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			
			if(!myrank)
			{
				if(e_i>0)
				{
					y = D*p - W*Qx;
				}
				else
				{
					y = D*p;
				}
				
				q = D.transpose()*y;
				Wty = W.transpose()*y;
				
			}
			
			MPI_Bcast(q.data(), q.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(Wty.data(), Wty.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
			
			if(e_i>0)
			{
				x = V.transpose()*q - Q.transpose()*Wty;
			}
			else
			{
				x = V.transpose()*q;
			}
			
			xnorm2_local = x.norm();
			xnorm2_local = xnorm2_local*xnorm2_local;
			
			MPI_Allreduce(&xnorm2_local, &xnorm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			
			if(abs(xnorm2 - xnorm2_old)/xnorm2 < 1E-5)
			{
				if(!myrank)
				{	
					W.conservativeResize(m, e_i+1);
					W.col(e_i) = y;
				}
				
				Q.conservativeResize(e_i+1, myn);
				Q.row(e_i) = x.transpose()/sqrt(xnorm2);
				
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