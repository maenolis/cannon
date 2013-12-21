#ifndef MPICAN_H
#define MPICAN_H

#define TRUE 1
#define FALSE 0
#define MAX_VALUE_FRAND 100

//converts a string to int
int convert_to_int(char *buffer);	

//prints the message if root and aborts.
void print_error_abort(char *message, int myrank); 

//creates submatrix with random values or 0.00 
void create_submatrix(int nlocal, int myrank, double **arr, int random);

//actual computation of multiplication
void matrix_multiply(int n, double *a, double *b, double *c);

//begins the parallel operation
void execute_matrix_multiply(int n, MPI_Comm comm, int npes, int myrank); 

//return double[0 , MAX_VALUE_FRAND)
double fRand();

//prints matrix in console
void print_matrix(int n, int npes, int nlocal, double *recv, int *dims);

#endif 
