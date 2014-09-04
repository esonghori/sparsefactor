.PHONY: bin cleanmake file 

#EIGEN=/home/em24/eigen/
MPICPP=mpic++ -O3 -g  -std=c++11 -fopenmp





all: checkEigen bin bin/omp bin/ompMultiplefile bin/ompRand bin/ompOutFile bin/ompSubset bin/powerMethodDVxS bin/powerMethodAx bin/ataDVxS bin/ataAx bin/ista

checkEigen:
ifeq ($(EIGEN),)
	$(error EIGEN is not set correctly.)
endif

bin:
	mkdir -p bin

bin/omp: src/omp/omp.cpp
	${MPICPP} -o bin/omp src/omp/omp.cpp  -I${EIGEN}
	
bin/ompMultiplefile: src/omp/ompMultiplefile.cpp
	${MPICPP} -o bin/ompMultiplefile src/omp/ompMultiplefile.cpp -I${EIGEN}
	
bin/ompRand: src/omp/ompRand.cpp
	${MPICPP} -o bin/ompRand src/omp/ompRand.cpp -I${EIGEN}

bin/ompOutFile: src/omp/ompOutFile.cpp
	${MPICPP} -o bin/ompOutFile src/omp/ompOutFile.cpp -I${EIGEN}

bin/ompSubset: src/omp/ompSubset.cpp
	${MPICPP} -o bin/ompSubset src/omp/ompSubset.cpp -I${EIGEN}
	


bin/powerMethodDVxS: src/powerMethod/powerMethodDVxS.cpp
	${MPICPP} -o bin/powerMethodDVxS src/powerMethod/powerMethodDVxS.cpp -I${EIGEN}
	
bin/powerMethodAx: src/powerMethod/powerMethodAx.cpp
	${MPICPP} -o bin/powerMethodAx src/powerMethod/powerMethodAx.cpp -I${EIGEN}
	
	
	
bin/ataDVxS: src/ata/ataDVxS.cpp
	${MPICPP} -o bin/ataDVxS src/ata/ataDVxS.cpp -I${EIGEN}	
	
bin/ataAx: src/ata/ataAx.cpp
	${MPICPP} -o bin/ataAx src/ata/ataAx.cpp -I${EIGEN}


bin/ista: src/ista/ista.cpp
	${MPICPP} -o bin/ista src/ista/ista.cpp -I${EIGEN}


clean:
	rm -rf bin
