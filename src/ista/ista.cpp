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
	
	
	if(argc!=15)
	{
		if(!myrank)
		{
			cout << "usage: mpiexec -n npes ./ista inDfile inVfile inXfile inYfile m n l numoffiles localD gammaN lambda steps verbose ncpu" << endl;
			cout << "numoffiles % npes = 0" << endl;
			cout << "gamma == gammaN / n" << endl;
		}
		MPI_Finalize();	
		return -1;
	}
	
	const char *inDfile = argv[1];
	const char *inVfile = argv[2];
	const char *inXfile = argv[3];
	const char *inYfile = argv[4];
	int m = atoi(argv[5]);
	int n = atoi(argv[6]);
	int l = atoi(argv[7]);
	int numoffiles = atoi(argv[8]);
	int localD = atoi(argv[9]);
	double gammaN = atof(argv[10]);
	double lambda = atof(argv[11]);
	int steps = atoi(argv[12]);
	int verbose = atoi(argv[13]);
	int ncpu = atoi(argv[14]);
	
	
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
		fD.open(inDfile);
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
		sfV << inVfile << "_" << fileID;	
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
	MatrixXd yres(m, 1);
	MatrixXd x(myn,1); //= MatrixXd::Random(myn,1); x = x/x.norm();
	MatrixXd deltax(myn,1);
	MatrixXd xold(myn,1);
	
	
	/*int y_ind = rand()%myn;
	if(!localD || !myrank)
	{
		MatrixXd nois = MatrixXd::Random(y.rows(),y.cols());
		nois = 0.1*(nois/nois.norm());
		y = D*V.col(y_ind) + nois;
	}
	if(!localD)
	{
		MPI_Bcast(y.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}*/
	
	
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
	
	if(!localD || !myrank)
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
	if(!localD)
	{
		MPI_Bcast(y.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
		if(localD)
		{
			p_local = V*x;
			MPI_Reduce(p_local.data(), p.data(), l, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if(!myrank)
			{
				yres = D*p - y;
				
				if(verbose)
				{
					cout <<"yres" << endl << yres.transpose() << endl;
				}
				
				diffY2 = yres.norm();
				diffY2 *= diffY2;
				
				q = D.transpose()*yres;
			}
			MPI_Bcast(q.data(), l, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		else
		{
			p_local = V*x;
			MPI_Allreduce(p_local.data(), p.data(), l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			yres = D*p - y;
			
			if(!myrank && verbose)
			{
				cout <<"yres" << endl << yres.transpose() << endl;
			}
			
			diffY2 = yres.norm();
			diffY2 *= diffY2;
			
			q = D.transpose()*yres;
		}	
		
		deltax = V.transpose()*q;
		
		
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
