.PHONY: checkEigen bin


MPICPP=mpic++ -O3 -g  -std=c++11 -fopenmp


all: checkEigen bin bin/omp bin/ompMultiplefile bin/powerMethodDVxS bin/powerMethodAx bin/ataDVxS bin/ataAx bin/istaAx bin/istaDVxS bin/mksample

checkEigen:
ifeq ($(EIGEN),)
	$(error EIGEN is not set.)
endif

bin:
	mkdir -p bin

bin/omp: src/omp/omp.cpp
	${MPICPP} -o bin/omp src/omp/omp.cpp  -I${EIGEN}
	
bin/ompMultiplefile: src/omp/ompMultiplefile.cpp
	${MPICPP} -o bin/ompMultiplefile src/omp/ompMultiplefile.cpp -I${EIGEN}
	

bin/powerMethodDVxS: src/powerMethod/powerMethodDVxS.cpp
	${MPICPP} -o bin/powerMethodDVxS src/powerMethod/powerMethodDVxS.cpp -I${EIGEN}
	
bin/powerMethodAx: src/powerMethod/powerMethodAx.cpp
	${MPICPP} -o bin/powerMethodAx src/powerMethod/powerMethodAx.cpp -I${EIGEN}
	
	
	
bin/ataDVxS: src/ata/ataDVxS.cpp
	${MPICPP} -o bin/ataDVxS src/ata/ataDVxS.cpp -I${EIGEN}	
	
bin/ataAx: src/ata/ataAx.cpp
	${MPICPP} -o bin/ataAx src/ata/ataAx.cpp -I${EIGEN}


bin/istaDVxS: src/ista/istaDVxS.cpp
	${MPICPP} -o bin/istaDVxS src/ista/istaDVxS.cpp -I${EIGEN}
	
bin/istaAx: src/ista/istaAx.cpp
	${MPICPP} -o bin/istaAx src/ista/istaAx.cpp -I${EIGEN}

bin/mksample: src/ista/mksample.cpp
	${MPICPP} -o bin/mksample src/ista/mksample.cpp -I${EIGEN}


clean:
	rm -rf bin
