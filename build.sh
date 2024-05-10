source /opt/intel/oneapi/setvars.sh
mkdir build
icpx -fsycl ./src/lab1.cpp -o ./build/lab1.out
icpx -fsycl ./src/lab2.cpp -o ./build/lab2.out
icpx -fsycl ./src/lab3.cpp -o ./build/lab3.out -qopenmp && ./build/lab3.out 1000 0.0001 100
