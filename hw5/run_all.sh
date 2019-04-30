make -j
mpirun -np $1 ./int_ring
mpirun -np $1 ./arr_ring
make clean
