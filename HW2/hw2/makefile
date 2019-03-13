CC = g++
CFLAGS = -Wall -O3 -fopenmp -march=native
CAREFULCFLAGS = -Wall -fopenmp 

default:
	$(CC) $(CFLAGS) val_test01_solved.cpp -o val_test01_solved
	$(CC) $(CFLAGS) val_test02_solved.cpp -o val_test02_solved
	$(CC) $(CFLAGS) MMult1.cpp -o MMult1
	$(CC) $(CFLAGS) omp_solved2.c -o omp_solved2
	$(CC) $(CFLAGS) omp_solved3.c -o omp_solved3
	$(CC) $(CFLAGS) omp_solved4.c -o omp_solved4
	$(CC) $(CFLAGS) omp_solved5.c -o omp_solved5
	$(CC) $(CAREFULCFLAGS) omp_solved6.c -o omp_solved6
	$(CC) $(CFLAGS) jacobi2D-omp.c -o jacobi2D-omp

jacobi:
	$(CC) $(CFLAGS) jacobi2D-omp.c -o jacobi2D-omp

clean:
	rm -f val_test01_solved
	rm -f val_test02_solved
	rm -f MMult1
	rm -f omp_solved2
	rm -f omp_solved3
	rm -f omp_solved4
	rm -f omp_solved5
	rm -f omp_solved6
	rm -f jacobi2D-omp
