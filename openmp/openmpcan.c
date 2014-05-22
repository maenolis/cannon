#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <time.h> 
#include <math.h> 
#include <omp.h>
#include "openmpcan.h"

int main(int argc, char *argv[]) {
	int i, j, k, nlocal, threads_num, dim, procdim;
	int **topArray, coordsA[2], coordsB[2], coordsC[2];
	double **finalA, **finalB, **finalC, ***A, ***Aa, ***B, ***Bb, ***C;
	double start, end;

	start = omp_get_wtime();
	srand(time(NULL));

	//Topology size
	procdim = convert_to_int(argv[1]);

	//Matrix size
	dim = convert_to_int(argv[2]);

	//Number of processes
	threads_num = convert_to_int(argv[3]);

	if (dim == -1 || procdim == -1 || threads_num == -1 || dim % procdim != 0) {
		printf("Check your arguments again.\n");
	}
	//Submatrix size
	nlocal = dim / procdim;

	A = init_array(procdim, nlocal, TRUE);
	B = init_array(procdim, nlocal, TRUE);
	C = init_array(procdim, nlocal, FALSE);

	print_array(A, procdim, nlocal, dim);
	print_array(B, procdim, nlocal, dim);

	//These will be A and B after the first alignment	
	Aa = malloc(procdim * sizeof(double**));
	if (Aa == NULL) {
		return EXIT_FAILURE;
	}
	Bb = malloc(procdim * sizeof(double**));
	if (Bb == NULL) {
		return EXIT_FAILURE;
	}
	for (i = 0; i < procdim; i++) {
		Aa[i] = malloc(procdim * sizeof(double*));
		if (Aa[i] == NULL) {
			return EXIT_FAILURE;
		}
		Bb[i] = malloc(procdim * sizeof(double*));
		if (Bb[i] == NULL) {
			return EXIT_FAILURE;
		}
	}

	//Initial alignment. 
	for (i = 0; i < procdim; i++) {
		for (j = 0; j < procdim; j++) {
			/*Trexouses suntetagmenes*/
			coordsA[0] = i; //row
			coordsB[0] = i; //row
			coordsA[1] = j; //column
			coordsB[1] = j; //column
			if (coordsA[1] - coordsA[0] < 0) //Check if twist needed
				coordsA[1] = coordsA[1] - coordsA[0] + procdim;	//emulate the periodicity of the array by adding the size of the array
			else
				coordsA[1] = coordsA[1] - coordsA[0];
			if (coordsB[0] - coordsB[1] < 0)
				coordsB[0] = coordsB[0] - coordsB[1] + procdim;
			else
				coordsB[0] = coordsB[0] - coordsB[1];

			Aa[coordsA[0]][coordsA[1]] = A[i][j];
			Bb[coordsB[0]][coordsB[1]] = B[i][j];
		}
	}

	//Free the arrays of pointers but not the data
	for (i = 0; i < procdim; i++) {
		free(A[i]);
		free(B[i]);
	}
	free(A);
	free(B);

	topArray = malloc(procdim * sizeof(double*));
	for (i = 0; i < procdim; i++) {
		topArray[i] = malloc(procdim * sizeof(double));
		for (j = 0; j < procdim; j++) {
			topArray[i][j] = -1;
		}
	}
	/*******************************************/
#pragma omp parallel shared( Aa, Bb, C, topArray, procdim, nlocal, dim ) private( i, j, k, coordsA, coordsB, coordsC ) num_threads(threads_num)
	{
#pragma omp critical
		{
			//create topology
			for (i = 0; i < procdim; i++) {
				for (j = 0; j < procdim; j++) {
					if (topArray[i][j] < 0) {
						topArray[i][j] = omp_get_thread_num();
						;
						coordsA[0] = i;
						coordsA[1] = j;
						coordsB[0] = i;
						coordsB[1] = j;
						coordsC[0] = i;
						coordsC[1] = j;
						i = procdim;
						break;
					}
				}
			}
		}

		for (i = 0; i < procdim; i++) {
			matrix_multiply(nlocal, Aa[coordsA[0]][coordsA[1]],
					Bb[coordsB[0]][coordsB[1]], C[coordsC[0]][coordsC[1]]);
			if (coordsA[1] - 1 < 0)
				coordsA[1] = coordsA[1] - 1 + procdim;
			else
				coordsA[1] = coordsA[1] - 1;
			if (coordsB[0] - 1 < 0)
				coordsB[0] = coordsB[0] - 1 + procdim;
			else
				coordsB[0] = coordsB[0] - 1;
		}
	}

	for (i = 0; i < procdim; i++) {
		free(topArray[i]);
	}
	free(topArray);

	print_array(C, procdim, nlocal, dim);

	dispose_array(Aa, procdim);
	dispose_array(Bb, procdim);
	dispose_array(C, procdim);

	free(Aa);
	free(Bb);
	free(C);

	end = omp_get_wtime();

	printf("\nDuration is: %f\n", end - start);
}

int convert_to_int(char *buffer) {
	int i, temp;
	if (strlen(buffer) > 0) {
		for (i = 0; i < strlen(buffer); i++) {
			//TODO
			if (buffer[i] < '0' || buffer[i] > '9')
				return -1;
		}
		temp = atoi(buffer);
		return temp;
	}
	return -1;
}

void matrix_multiply(int n, double *a, double *b, double *c) {
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < n; k++) {
				c[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

double fRand() {
	return rand() / (((double) RAND_MAX + 1) / MAX_VALUE_FRAND);
}

double *** init_array(int procdim, int nlocal, int random) {
	//2D array that holds the submatrices.
	double ***arr;
	int i, j, k;

	arr = malloc(procdim * sizeof(double**));
	for (i = 0; i < procdim; i++) {
		arr[i] = malloc(procdim * sizeof(double*));
		for (j = 0; j < procdim; j++) {
			arr[i][j] = malloc(nlocal * nlocal * sizeof(double));
			if (random)
				for (k = 0; k < nlocal * nlocal; k++) {
					arr[i][j][k] = fRand();
				}
			else
				for (k = 0; k < nlocal * nlocal; k++) {
					arr[i][j][k] = 0.00;
				}
		}
	}
	return arr;
}

void dispose_array(double ***Arr, int procdim) {

	int i, j;

	for (i = 0; i < procdim; i++) {
		for (j = 0; j < procdim; j++) {
			free(Arr[i][j]);
		}
		free(Arr[i]);
	}
}

void print_array(double ***Arr, int procdim, int nlocal, int dim) {
	int rowC = 0, columnC = 0;
	int i, j, k;
	double **finalArray;

	finalArray = malloc(dim * sizeof(double*));
	for (i = 0; i < dim; i++) {
		finalArray[i] = malloc(dim * sizeof(double));
	}
	printf("Matrix is:\n");
	for (i = 0; i < procdim; i++) {
		for (j = 0; j < procdim; j++) {
			rowC = i * nlocal;
			columnC = j * nlocal;
			for (k = 0; k < nlocal * nlocal; k++) {
				finalArray[rowC][columnC] = Arr[i][j][k];
				if ((k + 1) % (nlocal) == 0) {
					rowC++;
					columnC = j * nlocal;
				} else {
					columnC++;
				}
			}
		}
	}
	printf("-------------------------------------------------------\n");
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			printf("%f ", finalArray[i][j]);
		}
		printf("\n");
	}

	for (i = 0; i < dim; i++) {
		free(finalArray[i]);
	}
	free(finalArray);
}
