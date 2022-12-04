export LD_LIBRARY_PATH=../src/thundersvm/kernel:$LD_LIBRARY_PATH
export  OMP_NUM_THREADS=16
../build/bin/thundersvm-train -s 0 -t 2 -g 1 -c 10 -o 64 ../data/data1
