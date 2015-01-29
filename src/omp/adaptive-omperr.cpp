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


typedef struct errorID
{
	double error;
	unsigned int id;
	
	errorID():
		error(0), id(0)
	{	};
	
	errorID(double error, unsigned int id):
		error(error), id(id)
	{	}; 
	
	static int compare(const errorID& a, const errorID& b)
	{
		return (int)(a.error > b.error);
	}
}errorID;

int verbos = 0;

inline void printPercent(int i, int n)
{
	if(verbos && ( (int)((100.0*(i-1)) / n) < (int)((100.0*i) / n)))
	{
		cout << "\r\033[K"; // erase line
		cout << (int)((100.0*i) / n) << "%"<<std::flush;
	}
}

double getTimeMs(const timeval t1,const  timeval t2)
{
	double tE = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	tE += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	
	return tE;
}


void randperm(unsigned int n, unsigned int perm[])
{
	unsigned int i, j, t;

	for(i=0; i<n; i++)
		perm[i] = i;
	
	for(i=0; i<n; i++) 
	{
		j = rand()%(n-i)+i;
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}


int maxID(const MatrixXd & e)
{
	double max = 0;
	int index = 0;
	if(	e.cols()>1)	
	{
		for(int i=0; i<e.cols(); i++)
		{
			if(e(0,i) >= max)
			{
				index = i;
				max = e(0,i);
			}
		}
	}
	else
	{
		for(int i=0; i<e.rows(); i++)
		{
			if(e(i,0) >= max)
			{
				index = i;
				max = e(i,0);
			}
		}
	}
	
	return index;
}

int OMPQR(MatrixXd& D, MatrixXd& A, int k, double epsilon, vector<Trip>& tripletV)
{
	int npes, myrank;
	timeval t1, t2;
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


	int l = D.cols();
	int m = D.rows();
	int n = A.cols();
	assert(A.rows() == m);
	
	tripletV.clear();
	tripletV.reserve(k * n);
	
	
	MatrixXd Q(m, k);
	MatrixXd R(k, k);
	MatrixXd Dt = D.transpose();
	
	for(int c=0;c<n;c++)
	{
		MatrixXd supp(2, k);
		MatrixXd res;
		
		res = A.col(c);
		double anorm = res.norm();
		double er = 0;
		if(anorm < 1E-9)
			break;
		
		int i=0;
		
		double dt1 = 0, dt2 = 0, dt3 = 0;

		
		for(i=0;i<k;i++)
		{		
			MatrixXd coldot;
			MatrixXd Dtr = Dt*res;

			Dtr = Dtr.cwiseAbs();
			int idx = maxID(Dtr);
			
			supp(0,i) = idx;
		
			MatrixXd newcol = D.col(idx);
	
			
			for(int j=0;j<i;j++)
			{
				coldot = Q.col(j).transpose()*newcol;
				R(j, i) = coldot(0,0);
				newcol = newcol - R(j, i)*Q.col(j);
			}
			R(i, i) = newcol.norm();
			Q.col(i) = newcol/R(i, i);
		


			coldot = Q.col(i).transpose()* res;
			res = res - Q.col(i)*(coldot(0,0));
			
			er = res.norm()/anorm;
		
			supp(1,i) = er;
			if(er < epsilon)
			{
				
				i++;
				break;
			}
		}

		MatrixXd vom = R.block(0,0,i,i).triangularView<Eigen::Upper>().solve(Q.block(0,0,m,i).transpose()*A.col(c));
		
		if(!myrank)
		{
			printPercent(c, n);
		}
		
		//cout << "supp ";
		for(int j=0;j<i;j++)
		{
			tripletV.push_back(Trip(supp(0,j), c, vom(j)));
			//cout << supp(0,j) << ", " << supp(1,j) << "  ";
		}
		//cout << endl;
	}

	return 0;
}

int OMPBATCHLLT(MatrixXd& D, MatrixXd& A, int k, double epsilon, vector<Trip>& tripletV)
{
	int npes, myrank;
	timeval t1, t2;
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


	int l = D.cols();
	int m = D.rows();
	int n = A.cols();
	assert(A.rows() == m);
	
	tripletV.clear();
	tripletV.reserve(k * n);
	
	
	
	
	MatrixXd DtA = D.transpose()*A;
	MatrixXd G = D.transpose()*D;
	
	for(int c=0;c<n;c++)
	{
		double anorm = A.col(c).norm();
		
		if(anorm < 1E-9)
			break;
				
		assert((anorm - 1) < 1E-3);
	
		
		int i=0;
		
		double dt1 = 0, dt2 = 0, dt3 = 0;

		double delta, delta_old = 0;
		double er, er_old = anorm*anorm;
		
		MatrixXd supp(2, k);
		MatrixXd alpha_0 = DtA.col(c);
		MatrixXd alpha = alpha_0;
		MatrixXd alpha_I(k,1);
		MatrixXd G_I(l,k);
		MatrixXd G_II(k,k);
		MatrixXd beta;
		MatrixXd beta_I;
		MatrixXd L = MatrixXd::Zero(k,k);
		L(0,0) = 1;
		MatrixXd w;
		MatrixXd gamma_I;
		MatrixXd Linvalpha;
		for(i=0;i<k;i++)
		{
			int idx = maxID(alpha.cwiseAbs());
			supp(0,i) = idx;
			alpha_I(i, 0) =  alpha_0(idx, 0);
			
			if(i>0)
			{
				w = L.block(0,0,i,i).triangularView<Eigen::Lower>().solve(G_I.block(idx,0,1,i).transpose());
				L.block(i,0,1,i) = w.transpose();
				double wnorm = w.norm();
				L(i,i) = sqrt(1 - wnorm*wnorm);
			}
			G_I.col(i) = G.col(idx);
			

			

			Linvalpha = L.block(0,0,i+1,i+1).triangularView<Eigen::Lower>().solve(alpha_I.block(0,0,i+1,1));
			gamma_I = L.block(0,0,i+1,i+1).transpose().triangularView<Eigen::Upper>().solve(Linvalpha);
				
			
			beta = G_I.block(0,0,l,i+1)*gamma_I;
			
			beta_I = MatrixXd::Zero(i+1,1);
			for(int j=0;j<=i;j++)
			{
				beta_I(j,0) = beta(supp(0,j),0);
			}
			
			
			alpha = alpha_0 - beta;
			
			
			delta = (gamma_I.transpose()*beta_I)(0,0);
			er = (er_old - delta + delta_old); 
		
		
			supp(1,i) = sqrt(er)/anorm;
			if(er/(anorm*anorm) < epsilon*epsilon)
			{
				i++;
				break;
			}
			
			er_old = er;
			delta_old = delta;
		}
		
		
		if(!myrank)
		{
			printPercent(c, n);
		}
		
		//cout << "supp ";
		for(int j=0;j<i;j++)
		{
			tripletV.push_back(Trip(supp(0,j), c, gamma_I(j)));
			//cout << supp(0,j) << ", " << supp(1,j) << "  ";	
		}
		//cout << endl;
	}

	return 0;
}

void findIdx(int n, MatrixXd &A, MatrixXd& D, unsigned int * idx)
{
	int npes, myrank;
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	int myn = A.cols();
	
	
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	
	errorID test;
	
	MPI_Datatype ERROR_ID_type;
	int array_of_blocklengths[2] = {1, 1};
	MPI_Aint array_of_displacements[2];
	array_of_displacements[0] = 0;
	array_of_displacements[1] = sizeof(double);
	MPI_Datatype array_of_types[2] = {MPI_DOUBLE, MPI_UNSIGNED};
	MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &ERROR_ID_type);
	MPI_Type_commit(&ERROR_ID_type);
		
	std::vector<errorID> vError_local(myn);
	std::vector<errorID> vError(n);
	
	MatrixXd GD = (D.transpose()*D);
	MatrixXd E =  D*GD.inverse()*D.transpose()*A - A;	
	

	for(unsigned  j=0;j<A.cols();j++)
	{
		vError_local[j] = errorID(E.col(j).norm(), npes*j+myrank); 
		//cout << myrank << ": " << vError_local[j].id << "-> " <<  vError_local[j].error << endl;	
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(&vError_local.front(), myn, ERROR_ID_type,
               &vError.front(), myn, ERROR_ID_type,
               0, MPI_COMM_WORLD);
	           
    if(!myrank)
    {           
		sort(vError.begin(), vError.end(), errorID::compare);
		//cout << "errors" << endl;
		for(unsigned i = 0; i < n; i++)
		{
			idx[i] = vError[i].id;
			//cout << vError[i].id << ": " <<  vError[i].error << endl;
		}
	}
	MPI_Bcast(idx, n, MPI_INT, 0, MPI_COMM_WORLD);
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
			cout << "usage: ./omp infile outfile m n lmin lstep lmax kperl epsilon ncpu of QR_batch verbos N" << endl;
			cout << "QR_batch: QR 0, batch 1" << endl;
			cout << "N>n for subset omp, N==n for normal omp" << endl;
			cout << "ncpu for openmp cores, ncpu==1 when using mpi" << endl;
			cout << "kperl= k / l" << endl;
			cout << "lmin <= l < lmax" << endl;
			cout << "of==1 for writing outputs" << endl;
		}
			
		MPI_Finalize();	
		return -1;
	}
	
	int m = atoi(argv[3]);
	int n = atoi(argv[4]);
	int lmin = atoi(argv[5]);
	int lstep = atoi(argv[6]);
	int lmax = atoi(argv[7]);
	double kperl = atof(argv[8]);
	double epsilon = atof(argv[9]);
	int ncpu = atoi(argv[10]);
	int of = atoi(argv[11]);
	int batch = atoi(argv[12]);
	verbos = atoi(argv[13]);
	int N = atoi(argv[14]);
	
	
	Eigen::setNbThreads(ncpu);
	Eigen::initParallel();

	
	timeval t1, t2;
	
	
	
	
	
	
	
	int myn = (n + npes -1)/npes;
			
	
	MatrixXd A = MatrixXd::Zero(m, myn);
	MatrixXd D;
	vector<Trip> tripletV;
	
	
	unsigned int *idx = new unsigned int[n];	
	unsigned int *idx_read = new unsigned int[N];	
	if(!myrank)
	{
		randperm(n, idx);
		randperm(N, idx_read);
	}
	MPI_Bcast(idx, n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(idx_read, N, MPI_INT, 0, MPI_COMM_WORLD);

	
	
	ifstream fr;
	fr.open(argv[1]);
	if(!fr.is_open())
	{
		if(!myrank)
			cout << "File not found: " << argv[1] << endl;
			
		MPI_Finalize();
		return -1;
	}
	
	if(!myrank)
		cout<< "Start reading A from " << argv[1] <<endl;
	
	for(int i=0; i< m; i++)
	{
		for(int j=0; j< N; j++)
		{
			double temp;
			fr >> temp;
			if(idx_read[j]%npes == myrank && idx_read[j] < n)
				A(i, idx_read[j]/npes) = temp;
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

	
	int lold = 0;
	
	D = MatrixXd::Zero(m, 0);
	
	if(!myrank)
	{
		cout << "npes = " << npes << endl; 
		cout << "start omp on " << m << "x" << n << endl;
		cout << "kperl = " << kperl << " epsilon = " << epsilon << endl;
		cout << "(l, nnz(V), time) = " << endl;
	}
	
	
	//for(int l=lmin;l<lmax;l+=lstep)
	double global_error_D = 100;
	int l = lmin;
	while (global_error_D>epsilon && l<lmax+1)
	{	
		if(l!=lmin)
		{
			findIdx(n, A, D, idx); 
		}
		
		D.conservativeResize(m, l);
		
		for(int i=lold;i<l;i++)
		{
			//cout << idx[i] << endl;
			if(idx[i-lold]%npes == myrank)
			{
				D.col(i) = A.col(idx[i-lold]/npes)/A.col(idx[i-lold]/npes).norm();
				
				assert((D.col(i).norm() - 1) < 1E-3);
			}
			int root = idx[i-lold]%npes;
			MPI_Bcast(D.col(i).data(), D.col(i).size(), MPI_DOUBLE, root, MPI_COMM_WORLD);
		}
	    double error_D = 0;
	  
		/*for (int coli = 0; coli<myn; coli++)
		{
			VectorXd acol = A.col(coli);
			VectorXd x_acol = D.jacobiSvd(ComputeThinU | ComputeThinV).solve(acol);
			error_D = error_D + (D*x_acol-acol).norm(); 
		}
		error_D = error_D/myn;
		* */
		
		MatrixXd x_a = D.jacobiSvd(ComputeThinU | ComputeThinV).solve(A);
		error_D = (D*x_a-A).norm();
		
		error_D = (error_D*error_D)/myn;
		
		MPI_Allreduce(&error_D, &global_error_D, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		global_error_D = global_error_D/npes;
		
		lold = l;
		l = l + lstep;
		//if(!myrank)
		//	cout << "l = "<< lold<<endl;
	}
	int k = kperl*D.cols();
		gettimeofday(&t1, NULL);
		if(batch ==1)
		{
			if(!myrank && verbos)
			{
				cout << "Avegare error per col = " << global_error_D << endl;
				cout << "OMPBATCHLLT on  l = " << D.cols() << " myn = " << A.cols() << endl;
			}
			OMPBATCHLLT(D, A, k, sqrt(epsilon), tripletV);
			if(!myrank && verbos)
				cout << endl;
		}
		else 
		{
			if(!myrank && verbos)
				cout << "OMPQR on  l = " << D.cols() << " myn = " << A.cols() << endl;
			OMPQR(D, A, k, sqrt(epsilon), tripletV);
			if(!myrank && verbos)
				cout << endl;
		}
		gettimeofday(&t2, NULL);
		
		
		
		int Vnnz = tripletV.size();
		int Vnnzt = 0;
		MPI_Reduce(&Vnnz, &Vnnzt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if(!myrank)
			cout << D.cols() << " " << Vnnzt << " " << getTimeMs(t1,t2) << endl;
			
		
		
		if(of)
		{
			if(!myrank)
			{
				
				stringstream  sofD;
				sofD << argv[2] << "x" << D.cols() << "_D";
				
				ofstream foutD;
				foutD.open(sofD.str().c_str());
				
				cout << "start writing to " << sofD.str() << endl;
				
				foutD << D;
				
				foutD.close();
			}
		
			stringstream  sof;
			sof << argv[2] << "x" << D.cols()  << "_" << npes << "_" << myrank;
		
			ofstream fout;
			fout.open(sof.str().c_str());
			
			if(!myrank)
				cout << "start writing to " << sof.str() << endl;
		
			for(int i=0;i<tripletV.size();i++)
			{
				fout << tripletV[i].row() << " "<<  tripletV[i].col() << " " << tripletV[i].value() << endl;		
			}
			fout.close();
		}
		
	
	
	
	
	delete[] idx;
	MPI_Finalize();
	return 0;
}

