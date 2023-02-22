#include <fstream> 
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <cstdint>
#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include <mkl.h>


using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> T;

void CleanGeno(const string & genfile){
	int      nanim = 0;
	int      nsnp  = 0;
	size_t   start = 0;
	size_t 	 end   = 0;
	size_t   nump  = 0;
	string   sep   = " ";
	string   line;
	string   ID;
	ifstream iun01;
	ofstream oun01("CleanGeno.dat");
	ofstream oun02("Dimensions.dat");
	ofstream oun03("GenotypeID.dat");
	iun01.open(genfile.c_str());
	if(!iun01){
		cout << "Error! File: " << genfile << " could not be opened." << endl;
		exit(EXIT_SUCCESS);
	}
	auto tini = chrono::steady_clock::now();
	while(getline(iun01, line)){
		nanim++;
		if(nanim > 1){
			start   = line.find(sep);
			end     = line.size();
			nump    = line.find_first_of("0123456789", start);
			ID      = line.substr(0, start);
			if(nanim == 2){
				nsnp = end - nump;
			}
			for(int j = start + 1; j < line.length(); j++){
				if(line[j] == '3' || line[j] == '4'){
					line[j] = '1';
				}
			}
			oun01 << line.substr(nump, line.length()) << "\n";
			oun03 << ID << "\n";
		}
	}
	nanim--;
	oun02 << nanim << " " << nsnp << "\n";
	auto tend = chrono::steady_clock::now();
	chrono::duration<double> ttime = tend - tini;
	iun01.close();
	oun01.close();
	oun02.close();
	oun03.close();
	cout << "--------------------------------------------------------------------\n";
	cout << "Genotype file:                   " << genfile << "\n";
	cout << "Number of genotyped individuals: " << nanim << "\n";
	cout << "Number of markers:               " << nsnp << "\n";
	cout << "Elapsed time:                    " << ttime.count() << " seconds.\n";
	cout << "--------------------------------------------------------------------\n";
}

MatrixXd GetZ(const string & genfile, const string & dimfile){
	ifstream     iun01(genfile);
	ifstream     iun02(dimfile);
	string       line;
	string       gline;
	if(!iun01){
		cout << "Error! File: " << genfile << " could not be opened." << endl;
		exit(EXIT_SUCCESS);
	}
	if(!iun02){
		cout << "Error! File: " << dimfile << " could not be opened." << endl;
		exit(EXIT_SUCCESS);
	}
	/*
	* Get dimensions from file
	*/
	getline(iun02, line);
	stringstream ss(line);
	int          nGeno = 0;
	int          nsnp  = 0;
	int          i     = 0;
	double       k     = 0.0;
	ss >> nGeno;
	ss >> nsnp;
	/*
	* Declare Z matrix and initialize with zeros
	*/
	MatrixXd Z    = MatrixXd::Zero(nGeno, nsnp);
	VectorXd phat = VectorXd::Zero(nsnp);
	i             = 0;
	auto tini     = chrono::steady_clock::now();
	while(getline(iun01, line)){
		if(nsnp == line.length()){
			for(int j = 0; j < nsnp; j++){
				gline   = line[j];
				Z(i, j) = stod(gline);
			}
			i++;
		}
	}
	iun01.close();
	iun02.close();
	phat      = Z.colwise().mean().transpose()/2.0;
	int nfix  = 0;
	ofstream iunphat("p_hat.dat");
	for(int j = 0; j < nsnp; j++){
		if(phat(j) < 0.05){
			nfix++;
		}
		iunphat << phat(j) << "\n";
	}
	iunphat.close();
	k         = 2.0*(phat.array()*(1.0 - phat.array()).array()).sum();
	Z         = Z.rowwise() - 2.0*phat.transpose();
	Z        /= sqrt(k);
	auto tend = chrono::steady_clock::now();
	chrono::duration<double> ttime = tend - tini;
	cout << "--------------------------------------------------------------------\n";
	cout << "Z matrix centered and scaled.               \n";
	cout << "sum(2pq):                        " << k << "\n";
	cout << "Elapsed time:                    " << ttime.count() << " seconds.\n";
	cout << "--------------------------------------------------------------------\n";
	return(Z);
}


void PrepareData(const string & datafile, const int & n1 = 13, const int & n2 = 15){
	ifstream     iun01;
	ofstream     oun01("PedFile.dat");
	ofstream     oun02("PhenoFile.dat");
	ofstream     oun03("Inbrfile.dat");
	ofstream     oun04("EBVfile.dat");
	string       line;
	string       str;
	stringstream ss;
	int          nanim = 0;
	int          ncol  = 3;
	int 		 sire  = 0;
	int          dam   = 0;
	int          id    = 0;
	//int          sex   = 0;
	int          gen   = 0;
	int          nmp   = 0;
	int          nfp   = 0;
	string       sex;
	double       ff    = 0.0;
	double       hh    = 0.0;
	double       pheno = 0.0;
	double       res   = 0.0;
	double       poly  = 0.0;
	double       qtl   = 0.0;
	iun01.open(datafile.c_str());
	while(getline(iun01, line)){
		if(nanim > 0){
			stringstream ss(line);
			/*
			* Get Pedigree 
			*/
			ss >> id;
			ss >> sire;
			ss >> dam;
			/*
			* Get remaining ints
			*/
			ss >> sex;
			ss >> gen;
			ss >> nmp;
			ss >> nfp;
			/*
			* Get doubles
			*/
			ss >> str;
			stringstream(str) >> ff;
			ss >> str;
			stringstream(str) >> hh;
			ss >> str;
			stringstream(str) >> pheno;
			ss >> str;
			stringstream(str) >> res;
			ss >> str;
			stringstream(str) >> poly;
			ss >> str;
			stringstream(str) >> qtl;
			/*
			* Write pedigree
			*/
			oun01 << id << " " << sire << " " << dam << "\n";
			/*
			* Write data file (id + pheno)
			*/
			if(gen > 0 && gen < n2){
				oun02 << id << " " << 1 << " " <<  pheno + 5.0 << "\n";
			}
			/*
			* Write inbreeding
			*/
			if(gen >= n1){
				oun03 << id << " " << id << " " << ff << "\n";
			}
			/*
			* Write EBV
			*/
			oun04 << id << " " << qtl << "\n";
		}
		nanim++;
	}
	iun01.close();
	oun01.close();
	oun02.close();
	oun03.close();
}

MatrixXi GetPedigree(const string & filename){
  	ifstream infile(filename);
  	string       line;
  	int          nAnim = 0;
  	int          i     = 0;
	int          id;
	int          sire;
	int          dam;
  	while(getline(infile, line)){
		nAnim++;
    }
	infile.clear();
  	infile.seekg(0);
  	MatrixXi Ped(nAnim, 3);
	while(getline(infile, line)){
		stringstream ss(line);
		ss >> id;
		ss >> sire;
		ss >> dam;
		Ped(i, 0) = id;
		Ped(i, 1) = sire;
		Ped(i, 2) = dam;
		i++;
    }
	infile.close();
  	return(Ped);
}

VectorXi GetGenotypeID(const string & filename){
	ifstream infile(filename);
  	string       line;
  	int          nAnim = 0;
  	int          i     = 0;
	int          id    = 0;
  	while(getline(infile, line)){
		nAnim++;
    }
	infile.clear();
  	infile.seekg(0);
  	VectorXi ID(nAnim);
	while(getline(infile, line)){
		stringstream ss(line);
		ss >> id;
		ID(i) = id;
		i++;
    }
	infile.close();
  	return(ID);
}

VectorXd GetLi(const MatrixXd & U){
  	VectorXd L(2);
  	double det = U(1, 1)*U(2, 2) - U(2, 1)*U(1, 2);
  	L(0)       = (U(2, 2)*U(1, 0) - U(2, 0)*U(1, 2))/det;
  	L(1)       = (U(1, 1)*U(2, 0) - U(1, 0)*U(1, 2))/det;    
  	return(L);
}

double GetDii(const MatrixXd & U){
  	double detu = 1.0/(U(1, 1)*U(2, 2) - U(1, 2)*U(1, 2));
  	double var  = U(0, 0) - U(0, 1)*(U(2, 2)*U(1, 0) - U(2, 0)*U(1, 2))*detu - U(0, 2)*(U(1, 1)*U(2, 0) - U(1, 0)*U(1, 2))*detu;
  	return(var);
}

MatrixXd GetGi(const MatrixXd & M, const int & posi, const int & poss, const int & posd, const int & m){
  	MatrixXd Z = MatrixXd::Zero(3, m);
  	if(posi != -1){
    	Z.row(0) = M.row(posi);
  	}
  	if(poss != -1){
    	Z.row(1) = M.row(poss);
  	}
  	if(posd != -1){
    	Z.row(2) = M.row(posd);
  	}
  	MatrixXd XpX(MatrixXd(3, 3).setZero().selfadjointView<Lower>().rankUpdate(Z));
  	if(posi == -1){
    	XpX(0, 0) = 1.0;
  	}
  	if(poss == -1){
    	XpX(1, 1) = 1.0;
  	}
  	if(posd == -1){
    	XpX(2, 2) = 1.0;
  	}
  	return(XpX);
}

VectorXd MatD(const MatrixXi & Ped){
  int nanim = Ped.rows();
  int S     = 0;
  int D     = 0;
  int t     = 0;
  int i     = 0;
  int j     = 0;
  
  VectorXi Anc;  
  VectorXi Lap; 
  VectorXd F;
  VectorXd B;
  VectorXd L;
  
  Anc.setZero(nanim + 1);
  Lap.setZero(nanim + 1);
  F.setZero(nanim + 1);
  B.setZero(nanim + 1);
  L.setZero(nanim + 1);
  
  F(0)   =-1; 
  Lap(0) =-1;
  for(i = 1, t = -1; i <= nanim; i++) { 
    S = Ped(i - 1, 1); 
    D = Ped(i - 1, 2);
    Lap(i) = ((Lap(S) < Lap(D)) ? Lap(D) : Lap(S)) + 1;
    if (Lap(i) > t) t = Lap(i);
  }
  VectorXi St;
  VectorXi Mi;
  
  St.setZero(t + 1);
  Mi.setZero(t + 1);
  for(i = 1; i <= nanim; i++) {
    S    = Ped(i - 1, 1); 
    D    = Ped(i - 1, 2);
    B(i) = 0.5 - 0.25*(F(S) + F(D)); 
    for (j = 0; j < Lap(i); j++) {
      ++St(j); 
      ++Mi(j);
    } 
    if (S == 0 || D == 0) {
      F(i) = L(i) = 0; 
      continue;
    }
    if(S == Ped(i - 2, 1) && D == Ped(i - 2, 2)) {
      F(i) = F(i - 1); 
      L(i) = L(i - 1); 
      continue;
    }
    F(i)         = -1; 
    L(i)         = 1; 
    t            = Lap(i); 
    Anc(Mi(t)++) = i; 
    while(t > -1) {
      j = Anc(--Mi(t)); 
      S = Ped(j - 1, 1); 
      D = Ped(j - 1, 2); 
      if (S) {
        if (!L(S)){
          Anc(Mi(Lap(S))++) = S;
        }  
        L(S) += 0.5*L(j); 
      }
      if (D) {
        if (!L(D)){
          Anc(Mi(Lap(D))++) = D;
        } 
        L(D) += 0.5*L(j);
      }
      F(i) += L(j)*L(j)*B(j);
      L(j) = 0;
      if (Mi(t) == St(t)) --t;
    } 
  }
  return(B.tail(nanim));
}

VectorXi SetDiff(const VectorXi & Full, const VectorXi & Small){
  VectorXi Subset(Full.size());
  auto it = std::set_difference(Full.data(), Full.data() + Full.size(), 
                                Small.data(), Small.data() + Small.size(), 
                                Subset.data());
  Subset.conservativeResize(std::distance(Subset.data(), it));
  return(Subset);
}

void GetA22(const string & pedfile, const string & genidfile, const bool & OrigID){
	int    nAnim;
	int    nGeno;
	int    nSubs;
	int    row;
	int    col;
	int    neq;
	double val;
	chrono::duration<double> ttime;  	
	/*
	* Pedigree
	*/
	auto s1 = chrono::steady_clock::now();
  	MatrixXi Pedigree = GetPedigree(pedfile);
  	nAnim             = Pedigree.rows();
  	auto e1   = chrono::steady_clock::now();
  	ttime     = e1 - s1;
	/*
	* Genotype ID
	*/
	VectorXi  GenID  = GetGenotypeID(genidfile);
	VectorXi  FullID = Pedigree.col(0); 
  	nGeno            = GenID.rows();
	/*
	* Some variables
	*/
	VectorXi SubID    = SetDiff(FullID, GenID);
	VectorXd D        = MatD(Pedigree);
	VectorXi pednode  = VectorXi::Zero(3);
	VectorXi rownode  = VectorXi::Zero(3);
	VectorXi colnode  = VectorXi::Zero(3);
	VectorXd L        = VectorXd::Zero(3);
	nSubs             = SubID.rows();
	/*
	* Unordered_map to find individuals
	*/
	std::unordered_map<int, int> RowID;
  	std::unordered_map<int, int> ColID;
  	std::unordered_map<int, int>::iterator colit;
  	std::unordered_map<int, int>::iterator rowit;
  	auto tini = chrono::steady_clock::now();
  	for(int i = 0; i < nGeno; i++){
    	rowit = RowID.find(GenID(i));
    	if(rowit == RowID.end()){
      		RowID.insert({GenID(i), i});
    	}
  	}
  	for(int i = 0; i < nSubs; i++){
    	colit = ColID.find(SubID(i));
    	if(colit == ColID.end()){
      		ColID.insert({SubID(i), i});
    	}
  	}
	if(OrigID == true){
		neq = nAnim;
	} else if(OrigID == false){
		neq = nGeno;
	}
	SparseMatrix<double> A22(neq, neq); 
  	vector<T> A22COO;
  	A22COO.reserve(neq*10);
  	L(0) = 1.0;
  	L(1) = -0.5;
  	L(2) = -0.5;
	/*
   	* A**22 
   	*/
  	for(int i = 0; i < nAnim; i++){
    	pednode(0) = Pedigree(i, 0);
    	pednode(1) = Pedigree(i, 1);
    	pednode(2) = Pedigree(i, 2);
    	for(int j = 0; j < 3; j++){
      		for(int k = 0; k < 3; k++){
        		rowit = RowID.find(pednode(j));
        		colit = RowID.find(pednode(k));
        		if(((pednode(j) != 0) && (pednode(k) != 0)) && ((rowit != RowID.end()) && (colit != RowID.end()))){
					if(OrigID == true){
						row = pednode(j) - 1;
						col = pednode(k) - 1;
					} else if (OrigID == false){
						row = rowit->second;
          				col = colit->second;
					}
          			val = L(j)*L(k)/D(i);
          			A22COO.push_back(T(row, col, val));
        		}
      		}
    	}
  	}
	/*
   	* Setting the matrix from coordinates
   	*/
  	A22.setFromTriplets(A22COO.begin(), A22COO.end());
  	auto tend   = chrono::steady_clock::now();
  	ttime       = tend - tini;
	cout << "--------------------------------------------------------------------\n";
  	cout << "A**22 matrix (genotyped animals)   \n";
	cout << "Animals in pedigree:             " << nAnim << "\n";
  	cout << "Genotyped animals:               " << nGeno << "\n";
  	cout << "Total time:                      " << ttime.count() << " seconds.\n";
  	cout << "trace(A**22):                    " << A22.diagonal().sum() << "\n";
	cout << "Number of nonzero elements:      " << A22.nonZeros() << "\n";
	cout << "--------------------------------------------------------------------\n";
	/*
	* Writing matrix to file
	*/
	ofstream oun01("A22.dat");
	for(int k = 0; k < A22.outerSize(); k++){
		for(SparseMatrix<double>::InnerIterator iit(A22, k); iit; ++iit){
			if(iit.row() <= iit.col()){
				oun01 << iit.row() + 1 << " " << iit.col() + 1 << " " << iit.value() << "\n";
			}
		}
	}
	oun01.close();
}

void GInvLDL(const string & pedfile, const string & genidfile, const MatrixXd & Z, const bool & OrigID){
  	/*
   	* Integer variables
   	*/
  	int    nAnim   = 0;
  	int    nGeno   = 0;
  	int    m       = 0;
  	int    iid     = 0;
  	int    sid     = 0;
  	int    did     = 0;
  	int    gii     = 0;
  	int    gis     = 0;
  	int    gid     = 0;
  	int    row     = 0;
  	int    col     = 0;
  	int    neq     = 0;
  	/*
   	* Double precision variables
   	*/
  	double val    = 0.0;
  	double logdet = 0.0;
  	double nval   = 0.0;
  	double trace  = 0.0;
  	chrono::duration<double> ttime;
  	/*
   	* Vectors (integer, doubles) and matrices (doubles, and sparse)
   	*/
  	VectorXi pednode(3);
  	VectorXi gennode(3);
  	VectorXd L(3);
  	L(0) = 1.0;
  	L(1) = -0.5;
  	L(2) = -0.5;
	VectorXd tmp;
  	/*
   	* Get pedigree from file
   	*/
  	auto s1 = chrono::steady_clock::now();
  	MatrixXi Pedigree = GetPedigree(pedfile);
  	nAnim             = Pedigree.rows();
  	auto e1   = chrono::steady_clock::now();
  	ttime     = e1 - s1;
  	cout << "--------------------------------------------------------------------\n";
  	cout << "Pedigree processed in:           " << ttime.count() << " seconds.\n";
  	cout << "Animals in pedigree:             " << nAnim << "\n";
  	/*
   	* get genotype ID, allocate D vector, triplet vector and Ginv
   	*/
  	VectorXi  GenID = GetGenotypeID(genidfile);
  	nGeno           = GenID.rows();
	m               = Z.cols();
  	if(nGeno != Z.rows()){
		cout << "Dimensions of Genotype ID and Z matrix do not match. Check those files.\n";
		exit(EXIT_FAILURE);
  	}
  	VectorXd  D     = VectorXd::Zero(nGeno);
  	vector<T> GinvCOO;
  	GinvCOO.reserve(nGeno*10);
	if(OrigID == true){
		neq = nAnim;
	} else if(OrigID == false){
		neq = nGeno;
	}
  	SparseMatrix<double> Ginv(neq, neq);
	/*
   	* Recoding Genotyped IDs to find rows of Z
   	*/
  	std::unordered_map<int, int> NewID;
  	std::unordered_map<int, int>::iterator it;
  	for(int i = 0; i < nGeno; i++){
    	it = NewID.find(GenID(i));
    	if(it == NewID.end()){
      		NewID.insert({GenID(i), i});
    	}
  	}
	/*
	* Read A**22 and find trace
	*/
	ifstream iuna22("A22.dat");
	string   line;
	double   trA22 = 0.0;
	while(getline(iuna22, line)){
		stringstream ss(line);
		ss >> row;
		ss >> col;
		ss >> val;
		if(row == col){
			trA22 += val;
		}
	}
	iuna22.close();
	/*
	* Computing G inverse
	* Main loop.
	*/
	auto start = chrono::steady_clock::now();
	for(int i = 0; i < nGeno; i++){
		// Finding if individual and parents are genotyped
		iid = GenID(i); 
    	it  = NewID.find(iid);
    	if(it != NewID.end()){
      		gii = it->second;
    	} else if(it == NewID.end()){
      		gii = -1;
    	}
    	sid = Pedigree(GenID(i) - 1, 1);
    	it  = NewID.find(sid);
    	if(it != NewID.end()){
      		gis = it->second;
    	} else if(it == NewID.end()){
      		gis = -1;
    	}
    	did = Pedigree(GenID(i) - 1, 2);
    	it  = NewID.find(did);
    	if(it != NewID.end()){
      		gid = it->second;
    	} else if(it == NewID.end()){
      		gid = -1;
    	}
		// Calculating Gi, Li, and Di from manuscript
    	MatrixXd XpX = GetGi(Z, gii, gis, gid, m);
    	tmp          = GetLi(XpX);
		// Normalizing
    	if(gis != -1 && gid != -1){
			L.tail(2) = -tmp/tmp.sum();
		}
		D(i) = GetDii(XpX); // TODO: see if normalizing makes difference.
     	// Pedigree information for animal i. This gives the positions of the nonzero elements. 
    	pednode(0) = GenID(i);
    	pednode(1) = Pedigree(GenID(i) - 1, 1);
    	pednode(2) = Pedigree(GenID(i) - 1, 2);
     	// Genomic node for animal i.
    	gennode(0) = gii;
    	gennode(1) = gis;
    	gennode(2) = gid;
		for(int j = 0; j < 3; j++){
      		for(int k = 0; k < 3; k++){
        		if((pednode(j) != 0) && (pednode(k) != 0) && (gennode(j) != -1) && (gennode(k) != -1)){
					if(OrigID == true){
						// row and column correspond to pedigree id
						row = pednode(j) - 1;
            			col = pednode(k) - 1;
					} else if(OrigID == false){
						// row and column begin at 0
						it  = NewID.find(pednode(j));
            			row = it->second;
            			it  = NewID.find(pednode(k));
            			col = it->second;
					}
					// Value
					val = L(j)*L(k)/D(i);
          			GinvCOO.push_back(T(row, col, val));
      			}
    		}
		}
	}
	/*
	* Setting G inverse from vector of triplets
	*/
	Ginv.setFromTriplets(GinvCOO.begin(), GinvCOO.end());
	double trscal = trA22/Ginv.diagonal().sum();
	Ginv         *= trscal;
  	auto   end    = chrono::steady_clock::now();
  	ttime         = end - start;
	cout << "--------------------------------------------------------------------\n";
	cout << "Vectorized Cholesky method.\n";
	cout << "Scaling done by trace of A**22.\n";
  	cout << "G inverse completed in:          " << ttime.count() << " seconds\n";
  	cout << "Number of genotyped animals:     " << nGeno << "\n";
  	cout << "Number of markers used:          " << m << "\n";
  	cout << "trace(Ginv):                     " << Ginv.diagonal().sum() << "\n";
	cout << "trace(A**22):                    " << trA22 << "\n";
	cout << "Number of nonzero elements:      " << Ginv.nonZeros() << "\n";
	cout << "--------------------------------------------------------------------\n";
	/*
	* Writing matrix to file
	*/
	ofstream oun01("GInvLDLt.dat");
	for(int k = 0; k < Ginv.outerSize(); k++){
		for(SparseMatrix<double>::InnerIterator iit(Ginv, k); iit; ++iit){
			if(iit.row() <= iit.col()){
				oun01 << iit.row() + 1 << " " << iit.col() + 1 << " " << iit.value() << "\n";
			}
		}
	}
	oun01.close(); 
}

int getindex_ivec(const VectorXi & x, const int & val){
  	int ans = -1;
  	for(int i = 0; i < x.rows(); i++){
    	if(x(i) == val){
      		ans = i;
    	}
  	}
  	return(ans);
}

void GInvLZU(const string & pedfile, const string & genidfile, const MatrixXd & Z, const bool & OrigID){
  	/*
   	* Integer variables
   	*/
  	int    nAnim   = 0;
  	int    nGeno   = 0;
  	int    m       = 0;
  	int    iid     = 0;
  	int    pid     = 0;
  	int    gid     = 0;
  	int    row     = 0;
  	int    col     = 0;
  	int    neq     = 0;
  	int    one     = 1;
  	int    rval    = 0;
  	int    tmp1    = 0;
  	int    tmp2    = 0;
  	int    maxval  = 0;
  	int    minval  = 0;
  	/*
   	* Double precision variables
   	*/
  	double val    = 0.0;
  	double mnod   = 0.0;
  	double nval   = 0.0;
  	chrono::duration<double> ttime;  	
	/*
   	* Vectors (integer, doubles) and matrices (doubles, and sparse)
   	*/
  	VectorXi pednode(3);
  	VectorXi gennode(3);
  	/*
   	* Get pedigree from file
   	*/
  	auto s1 = chrono::steady_clock::now();
  	MatrixXi Pedigree = GetPedigree(pedfile);
  	nAnim             = Pedigree.rows();
  	auto e1   = chrono::steady_clock::now();
  	ttime     = e1 - s1;
  	cout << "--------------------------------------------------------------------\n";
  	cout << "Pedigree processed in:           " << ttime.count() << " seconds.\n";
  	cout << "Animals in pedigree:             " << nAnim << "\n";
  	/*
   	* get genotype ID, allocate D vector, triplet vector and Ginv
   	*/
  	VectorXi  GenID = GetGenotypeID(genidfile);
  	nGeno           = GenID.rows();
	m               = Z.cols();
  	if(nGeno != Z.rows()){
		cout << "Dimensions of Genotype ID and Z matrix do not match. Check those files.\n";
		exit(EXIT_FAILURE);
  	}
  	vector<T> GinvCOO;
  	GinvCOO.reserve(nGeno*10);
	if(OrigID == true){
		neq = nAnim;
	} else if(OrigID == false){
		neq = nGeno;
	}
  	SparseMatrix<double> Ginv(neq, neq);
	SparseMatrix<double> Ginvt(neq, neq);
	/*
   	* Recoding Genotyped IDs to find rows of Z
   	*/
  	std::unordered_map<int, int> NewID;
  	std::unordered_map<int, int>::iterator it;
  	for(int i = 0; i < nGeno; i++){
    	it = NewID.find(GenID(i));
    	if(it == NewID.end()){
      		NewID.insert({GenID(i), i});
    	}
  	}
	auto sm = chrono::steady_clock::now();
	unordered_map<uint64_t, int>           Graph;
	unordered_map<uint64_t, int>::iterator uit;
	uint64_t 		                       ijpos;
	int                                    curr = 0;
	vector<vector<int>> 	               AdjMat(nAnim);
	for(int i = 0; i < nAnim; i++){
		for(int j = 0; j < 3; j++){
			for(int k = 0; k < 3; k++){
				if(Pedigree(i, j) != 0 && Pedigree(i, k) != 0){
					uint32_t rij = Pedigree(i, j) - 1;
  					uint32_t cij = Pedigree(i, k) - 1;
					ijpos        = (uint64_t) rij << 32 | cij;
					uit          = Graph.find(ijpos);
					if(uit == Graph.end()){
						Graph.emplace(ijpos, curr);
						AdjMat[cij].push_back(rij);
						curr++;
					}
				}
			}
		}
	}
	tmp1 = 0;
	tmp2 = 1000;
	for(int i = 0; i < nAnim; i++){
		sort(AdjMat[i].begin(), AdjMat[i].end());
		rval      = AdjMat[i].size();
    	maxval    = max(tmp1, rval);
    	minval    = min(tmp2, rval);
    	mnod     += rval;
    	tmp1      = maxval;
    	tmp2      = minval;
	}
  	mnod    /= static_cast<double>(nAnim);
  	auto em  = chrono::steady_clock::now();
  	ttime    = em - sm;
  	cout << "--------------------------------------------------------------------\n";
  	cout << "Graph constructed in:            " << ttime.count() << " seconds.\n";
  	cout << "Max number of nodes/row:         " << maxval << " animals.\n";
  	cout << "Min number of nodes/row:         " << minval << " animals.\n";
  	cout << "Mean number of nodes/row:        " << mnod << " animals.\n";
	/*
	* Read A**22 and find trace
	*/
	ifstream iuna22("A22.dat");
	string   line;
	double   trA22 = 0.0;
	while(getline(iuna22, line)){
		stringstream ss(line);
		ss >> row;
		ss >> col;
		ss >> val;
		if(row == col){
			trA22 += val;
		}
	}
	iuna22.close();
	/*
   	 *  G inverse. Main loop
   	*/
  	auto start = chrono::steady_clock::now();
  	int ni = 0;
  	vector<int> colnode;
  	for(int i = 0; i < nGeno; i++){
		// i is a column of G inverse. 
		// Find which rows of column i are not zero.
    	gid     = GenID(i) - 1;
    	colnode = AdjMat[gid];
    	ni      = colnode.size(); 
    	VectorXi Genlist(ni);
		// Find if individuals corresponding to those rows are genotyped
    	for(int j = 0; j < ni; j++){
      		iid = Pedigree(colnode[j], 0);
      		it  = NewID.find(iid);
      		if(it != NewID.end()){
        		Genlist(j) = it->second;
      		} else if(it == NewID.end()){
        		Genlist(j) = -1;
      		}
    	}
		// Genotypes for the rows of column i 
    	MatrixXd Zi  = MatrixXd::Zero(ni, m);
    	for(int k = 0; k < ni; k++){
      		if(Genlist(k) != -1){
        		Zi.row(k) = Z.row(Genlist(k));
      		}
    	}
		// Gstar in manuscript
    	MatrixXd XpX(MatrixXd(ni, ni).setZero().selfadjointView<Lower>().rankUpdate(Zi));
    	for(int k = 0; k < ni; k++){
      		if(Genlist(k) == -1){
        		XpX(k, k) = 1.0;
      		}
    	}
		// ej and qj vectors in manuscript
    	int      pos = getindex_ivec(Genlist, i);
    	VectorXd ej  = VectorXd::Zero(ni);
    	VectorXd qj  = VectorXd::Zero(ni);
    	ej(pos)      = 1;
		// solving using Cholesky decomposition
    	qj           = XpX.ldlt().solve(ej);
    	if(OrigID == false){
      		col = i;
      		for(int j = 0; j < ni; j++){
        		if(Genlist(j) != -1){
          			row = Genlist(j);
          			val = qj(j);
          			GinvCOO.push_back(T(row, col, val));
        		}
      		}
    	} else if(OrigID == true){
      		col = GenID(i) - 1;
      		for(int j = 0; j < ni; j++){
        		if(Genlist(j) != -1){
          			row = GenID(Genlist(j)) - 1;
          			val = qj(j);
          			GinvCOO.push_back(T(row, col, val));
        		}
      		}
    	}
  	}
	Ginv.setFromTriplets(GinvCOO.begin(), GinvCOO.end());
  	Ginvt         = Ginv.transpose();
  	Ginv         += Ginvt;
  	Ginv         *= 0.5;
	double trscal = trA22/Ginv.diagonal().sum();
	Ginv         *= trscal;
  	auto end      = chrono::steady_clock::now();
  	ttime         = end - start;
	cout << "--------------------------------------------------------------------\n";
  	cout << "Le and Zhong (2021) method.  \n";
	cout << "Scaled by trace(A**22).\n";
  	cout << "G inverse completed in:          " << ttime.count() << " seconds.\n";
  	cout << "Number of genotyped animals:     " << nGeno << "\n";
  	cout << "Number of markers used:          " << m << "\n";
	cout << "trace(A**22):                    " << trA22 << "\n";
  	cout << "trace(Ginv):                     " << Ginv.diagonal().sum() << "\n";
	cout << "Number of nonzero elements:      " << Ginv.nonZeros() << "\n";
  	cout << "--------------------------------------------------------------------\n";
	/*
	* Writing matrix to file
	*/
	ofstream oun01("GInvLZ21.dat");
	for(int k = 0; k < Ginv.outerSize(); k++){
		for(SparseMatrix<double>::InnerIterator iit(Ginv, k); iit; ++iit){
			if(iit.row() <= iit.col()){
				oun01 << iit.row() + 1 << " " << iit.col() + 1 << " " << iit.value() << "\n";
			}
		}
	}
	oun01.close(); 
}


void GInvEPS(const string & genidfile, const MatrixXd & Z, const bool & OrigID){
	/*
   	* Integer variables
   	*/
  	int    nGeno   = 0;
  	int    m       = 0;
  	int    row     = 0;
  	int    col     = 0;
  	int    neq     = 0;
  	/*
   	* Double precision variables
   	*/
  	double val  = 0.0;
	double eps  = 0.01; 
  	chrono::duration<double> ttime;
	/*
   	* Get genotype ID, allocate D vector, triplet vector and Ginv
   	*/
  	VectorXi  GenID = GetGenotypeID(genidfile);
  	nGeno           = GenID.rows();
	m               = Z.cols();
  	if(nGeno != Z.rows()){
		cout << "Dimensions of Genotype ID and Z matrix do not match. Check those files.\n";
		exit(EXIT_FAILURE);
  	}
	/*
   	* Recoding Genotyped IDs to find rows of Z
   	*/
  	std::unordered_map<int, int> NewID;
  	std::unordered_map<int, int>::iterator it;
  	for(int i = 0; i < nGeno; i++){
    	it = NewID.find(GenID(i));
    	if(it == NewID.end()){
      		NewID.insert({GenID(i), i});
    	}
  	}
	/*
	* First compute full G.
	*/
	auto s1   = chrono::steady_clock::now();
	MatrixXd G(MatrixXd(nGeno, nGeno).setZero().selfadjointView<Lower>().rankUpdate(Z));
	G.diagonal().array() += eps;
	auto s2   = chrono::steady_clock::now();
	ttime     = s2 - s1;
	cout << "--------------------------------------------------------------------\n";
	cout << "Inverting using Cholesky decomposition (Eigen LDLt.solve()).\n";
  	cout << "Dense G calculated in:           " << ttime.count() << " seconds.\n";
	cout << "Epsilon value:                   " << eps << "\n";
	auto start    = chrono::steady_clock::now();
	MatrixXd Ginv = G.ldlt().solve(MatrixXd::Identity(nGeno, nGeno));
	auto end      = chrono::steady_clock::now();
	cout << "G inverse calculated in:         " << ttime.count() << " seconds.\n";
	cout << "Number of genotyped animals:     " << nGeno << "\n";
  	cout << "Number of markers used:          " << m << "\n";
	cout << "trace(Ginv):                     " << Ginv.diagonal().sum() << "\n";
	cout << "--------------------------------------------------------------------\n";
	/*
	* Writing matrix to file
	*/
	ofstream oun01("GInvEPSv.dat");
	for(int i = 0; i < nGeno; i++){
		for(int j = 0; j <= i; j++){
			if(OrigID == true){
				oun01 << GenID(i) << " " << GenID(j) << " " << Ginv(i, j) << "\n"; 
			} else if(OrigID == false){
				oun01 << i << " " << j << " " << Ginv(i, j) << "\n";
			}
		}
	}
	oun01.close();
}

void MinusDiag(const string & filename){
	ifstream iun01(filename);
	ofstream oun01("MinusDiagA22.dat");
	string   line;
	//string   minus = "-";
	int      id;
	double   val;
	while(getline(iun01, line)){
		stringstream ss(line);
		ss >> id;
		ss >> val;
		oun01 << id << " " << -val << "\n";
	}
}



int main(int argc, char* argv[]){
	vector<string> parameters(argc);
	for(int i = 0; i < argc; i++){
		parameters[i] = argv[i];
	}
	string pedfile = parameters[1];
	string genfile = parameters[2];
	/*
	* 1. Clean genotype file
	*/
	CleanGeno(genfile);
	/*
	* 2. Get Z matrix
	*/
	MatrixXd Z = GetZ("CleanGeno.dat", "Dimensions.dat");
	/*
	* 3. Prepare phenotype file, inbreeding file, and pedigree file
	*/
	PrepareData(pedfile, 13, 15);// Need to be cautious about this. 
	/*
	* 4. Get A**22
	*/
	GetA22("PedFile.dat", "GenotypeID.dat", true);
	/*
	* 5. Get G inverse (LDLt, Le and Zhong 2022)
	*/
	//GInvLDL("PedFile.dat", "GenotypeID.dat", Z, true);
	//GInvLZU("PedFile.dat", "GenotypeID.dat", Z, true);
	/*
	* 6. Call Hginv
	*/

	/*
	* Run cal_diag_A22 from Ismo Strandén and change sign 
	*/
	string calcA22inv = "./calc_diag_iA22_v34_para_64bit_Linux -MC 1000";
	string gfiles     = "PedFile.dat GenotypeID.dat";
	string ofiles     = "DiagA22inv.dat";
	string command    = calcA22inv + " " + gfiles + " " +  ofiles;
	const char *comms = command.c_str();
	int a = system (comms);
	if(a != 0){
		cout << "Error! Command: " << command << " not executed succesfully.\n";
		exit(EXIT_FAILURE);
	}
	MinusDiag("DiagA22inv.dat");
	cout << "Program finished successfully.\n";
	return(1);
}