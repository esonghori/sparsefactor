.PHONY: bin

EIGEN=/home/em24/eigen/
MPICPP=mpic++ -O3 -g  -std=c++11 -fopenmp

all: bin bin/ompqr bin/ompqr_multiplefile bin/ata bin/ompqr_ofile bin/ista bin/ata_base bin/ompqr_rand bin/power_method bin/power_method_base bin/ompqr_subset

bin:
	mkdir -p bin

bin/ompqr: src/ompqr.cpp
	${MPICPP} -o bin/ompqr src/ompqr.cpp  -I${EIGEN}
	
bin/ompqr_rand: src/ompqr_rand.cpp
	${MPICPP} -o bin/ompqr_rand src/ompqr_rand.cpp -I${EIGEN}
	
bin/ompqr_multiplefile: src/ompqr_multiplefile.cpp
	${MPICPP} -o bin/ompqr_multiplefile src/ompqr_multiplefile.cpp -I${EIGEN}

bin/ompqr_ofile: src/ompqr_ofile.cpp
	${MPICPP} -o bin/ompqr_ofile src/ompqr_ofile.cpp -I${EIGEN}

bin/ompqr_subset: src/ompqr_subset.cpp
	${MPICPP} -o bin/ompqr_subset src/ompqr_subset.cpp -I${EIGEN}
	
bin/ata: src/ata.cpp
	${MPICPP} -o bin/ata src/ata.cpp -I${EIGEN}

bin/power_method: src/power_method.cpp
	${MPICPP} -o bin/power_method src/power_method.cpp -I${EIGEN}
	
bin/power_method_base: src/power_method_base.cpp
	${MPICPP} -o bin/power_method_base src/power_method_base.cpp -I${EIGEN}
	
	
bin/ata_base: src/ata.cpp
	${MPICPP} -o bin/ata_base src/ata_base.cpp -I${EIGEN}
	
bin/ista: src/ista.cpp
	${MPICPP} -o bin/ista src/ista.cpp -I${EIGEN}

clean:
	rm -f bin/ompqr bin/ompqr_multiplefile bin/ata ompqr_ofile bin/ista ata_base bin/ompqr_rand bin/power_method bin/power_method_base bin/ompqr_subset
