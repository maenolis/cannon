#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "mpican.h"

#define MAX_VALUE_FRAND 100

int main(int argc, char *argv[]) {
	int dim, npes, procdim, myrank, i;
	double start, end; 
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes); 
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	start = MPI_Wtime(); 
	srand(time(NULL));	

	
	procdim = convert_to_int(argv[1]);	//Topology size
	if(procdim == -1 || sqrt(npes) != procdim) {
		print_error_abort("Check your first argument again. It must be an integer and equal to the argument you pass on MPI about the number of processes\n", myrank);		
	}

	dim = convert_to_int(argv[2]);	//Matrix size
	if(dim == -1 || dim%procdim != 0) {
		print_error_abort("Check your second argument again. It must be an integer and dividable by the number of processes\n", myrank);
	}
	
	execute_matrix_multiply(dim, MPI_COMM_WORLD, npes, myrank);
	
	end = MPI_Wtime(); 
	MPI_Finalize();
		
	if(myrank == 0) printf("\nDuration is: %f\n", end - start);
	return 0;
}

void execute_matrix_multiply(int n, MPI_Comm comm, int npes, int myrank) { 
    int nlocal, i; 
    int dims[2], periods[2]; 
    int my2drank, mycoords[2]; 
    int uprank, downrank, leftrank, rightrank; 
    int shiftsource, shiftdest;
	double *a, *b, *c, *recvC, *recvA, *recvB;	
    MPI_Status status; 
    MPI_Comm comm_2d; 
 
    //Dimensions of topology 
    dims[0] = dims[1] = sqrt(npes); 
 
    //Periodicity of topology
    periods[0] = periods[1] = 1; 
     
	//Dimensions of submatrix
    nlocal = n/dims[0];
	
	create_submatrix(nlocal, myrank, &a, TRUE);
	create_submatrix(nlocal, myrank, &b, TRUE);
	create_submatrix(nlocal, myrank, &c, FALSE);

    //Create cartesian topology rank reordering
    MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d); 

    //Rank of the process relative to the topology
    MPI_Comm_rank(comm_2d, &my2drank); 
    
	//Coordinates of the process relative to the topology
    MPI_Cart_coords(comm_2d, my2drank, 2, mycoords); 

	if(myrank == 0) {
		recvA = malloc(npes * nlocal * nlocal * sizeof(double));
		recvB = malloc(npes * nlocal * nlocal * sizeof(double));
	}
	
	MPI_Gather(a, nlocal * nlocal, MPI_DOUBLE, recvA, nlocal * nlocal, MPI_DOUBLE, 0, comm);
	MPI_Gather(b, nlocal * nlocal, MPI_DOUBLE, recvB, nlocal * nlocal, MPI_DOUBLE, 0, comm);
	
	if(myrank == 0) {
		print_matrix(n, npes, nlocal, recvA, dims);
		print_matrix(n, npes, nlocal, recvB, dims);		
			
		free(recvA);
		free(recvB);	
	}	
		
    //Gives the rank for send and receive. Used later on MPI_Sendrecv_replace
    MPI_Cart_shift(comm_2d, 1, -1, &rightrank, &leftrank); 
    MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank); 

	//Initial alignment
	MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest); 	
    MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_2d, &status); 

    MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest); 
    MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_2d, &status); 

    for (i = 0; i < dims[0]; i++) { 
		matrix_multiply(nlocal, a, b, c); //c=c+a*b
		
		//Shift matrices by one
		MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, leftrank, 1, rightrank, 1, comm_2d, &status); 
		MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE, uprank, 1, downrank, 1, comm_2d, &status); 
    } 
	
	if(myrank == 0) {
		recvC = malloc(npes * nlocal * nlocal * sizeof(double));
	}
	
	MPI_Gather(c, nlocal * nlocal, MPI_DOUBLE, recvC, nlocal * nlocal, MPI_DOUBLE, 0, comm);
	
	if(myrank == 0) {
		print_matrix(n, npes, nlocal, recvC, dims);
		free(recvC);
	}	
	
	free(a);
	free(b);
	free(c);
    MPI_Comm_free(&comm_2d); 
} 

void print_error_abort(char *message, int myrank) {
    if(myrank == 0) 
        printf("%s", message);
	MPI_Abort(MPI_COMM_WORLD, 1);	
}

int convert_to_int(char *buffer) {
    int i,temp;
    if(strlen(buffer) > 0){
        for(i = 0; i < strlen(buffer); i++){
                if(buffer[i] != '0' && 
				   buffer[i] != '1' && 
				   buffer[i] != '2' && 
				   buffer[i] != '3' && 
				   buffer[i] != '4' && 
				   buffer[i] != '5' && 
				   buffer[i] != '6' && 
				   buffer[i] != '7' && 
				   buffer[i] != '8' && 
				   buffer[i] != '9')	return -1;
        }
        temp = atoi(buffer);
        return temp;
    }
    return -1;
}

double fRand(int max) {							
    return rand()/(((double)RAND_MAX + 1) / max);
}

void create_submatrix(int nlocal, int myrank, double **arr, int random) {
	int i, size;
	
	size = nlocal * nlocal;
	(*arr) = malloc(size * sizeof(double));

	if(random) 
		for(i = 0; i < size; i++) { (*arr)[i] = fRand(MAX_VALUE_FRAND); } 
	else 
		for(i = 0; i < size; i++) { (*arr)[i] = 0.00; }
}

void matrix_multiply(int n, double *a, double *b, double *c) { 
	int i, j, k; 

	for (i = 0; i < n; i++) { 
		for (j = 0; j < n; j++) { 
			for (k = 0; k < n; k++) {
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}		
}

void print_matrix(int n, int npes, int nlocal, double *recv, int *dims) {
	
	int i, j, row, column, rowTwoDim, columnTwoDim;
	double **finalArray;
	
	//Two dimensional array for easy visualization of the data. Not needed for fuctionality
	finalArray = malloc(n * sizeof(double*));
	for (i = 0; i < n; i++) {
		finalArray[i] = malloc(n * sizeof(double));
	}		
	printf("\nMatrix is:\n");
	row = 0;	//Row of current location
	column = 0;	
	rowTwoDim = 0;	//Row in which it will be located in the two dimensional
	columnTwoDim = 0;	
	
	for (i = 0; i < npes * nlocal * nlocal; i++) { 
		
		finalArray[rowTwoDim][columnTwoDim] = recv[i];	
		if((i + 1)%(nlocal*nlocal) == 0) {	//End of submatrix reached
			if((column + 1) % dims[0] == 0 && column != 0) {	//Last column of submatrix reached
				row++;	
				column = 0; 
			}
			else {
				column++;	
			}
			
			rowTwoDim = row * nlocal;	
			columnTwoDim = column * nlocal;	
		}				
		else if( (i + 1)%(nlocal) != 0 ) {
			columnTwoDim++;		
		}	
		else {
			rowTwoDim++;		
			columnTwoDim = column * nlocal;	
		}
	}
	printf("-------------------------------------------\n");
	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			printf("%f ", finalArray[i][j]);
		}
		printf("\n");
	}
	for(i = 0; i < n; i++) {
		free(finalArray[i]);
	}
	free(finalArray);
}



