#ifndef OPENMPCAN_H
#define OPENMPCAN_H

#define TRUE 1
#define FALSE 0
#define MAX_VALUE_FRAND 100

//converts a string to int
int convert_to_int(char *buffer);	

//actual computation of multiplication
void matrix_multiply(int n, double *a, double *b, double *c); 

//return double[0 , MAX_VALUE_FRAND)
double fRand();

//allocates memory for 3D array and returns it
double *** init_array(int procdim, int nlocal, int random);

//free resources for Arr
void dispose_array(double ***Arr, int procdim);

//Print array
void print_array(double ***Arr, int procdim, int nlocal, int dim);

#endif 
