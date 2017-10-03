/*
Tim Whitaker
CSCI 551 Numerical Methods & Parallel Programming
Gaussian Elimination with Partial Pivoting and OpenMP in C++
*/

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
using namespace std;

double **allocate_matrix(int size);
double *allocate_vector(int size);
void randomize_matrix(int size, double **matrix);
void randomize_vector(int size, double *vector);
void print_matrix(int size, double **matrix);
void print_vector(int size, double *vector);

void gauss(int size, double **matrix, double *vector, double *solution);
void partial_pivot(int size, int pivot_row, double **matrix, double *vector);
double norm(double **a, double *b, double *solution, int size);

int main(int argc, char *argv[])
{
	int size = 8000;
	int tid;

	double **a_matrix = allocate_matrix(size);
	double *b_vector = allocate_vector(size);
	double *solution_vector = allocate_vector(size);

	srand(time(NULL));
	randomize_matrix(size, a_matrix);
	randomize_vector(size, b_vector);

	cout << "Number of cores: " << omp_get_num_procs() << endl;
	#pragma omp parallel
	{
		tid = omp_get_thread_num();

		if (tid == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}
	}

	double start = omp_get_wtime();

	gauss(size, a_matrix, b_vector, solution_vector);

	double end = omp_get_wtime();

	double time_elapsed = end - start;

	cout << "Time: " << time_elapsed << " seconds.\n";

	cout << "L-Squared Norm = " << norm(a_matrix, b_vector, solution_vector, size) << ".\n";

	free(a_matrix[0]);
	free(a_matrix);
	free(b_vector);
	free(solution_vector);

	return 0;
}

/*
This function is used to allocate a contiguous block of memory for our matrix.
*/
double **allocate_matrix(int size)
{
	double *data = (double *)malloc(size * size * sizeof(double));
	double **matrix = (double **)malloc(size * sizeof(double*));
	for (int i = 0; i < size; i++)
		matrix[i] = &(data[size*i]);

	memset(matrix[0], 0, size*size*sizeof(double));


	return matrix;
}

/*
This function is used to allocate a contiguous block of memory for our vectors.
*/
double *allocate_vector(int size)
{
	double *vector = (double *)malloc(size * sizeof(double));
	memset(vector, 0, size*sizeof(double));
	return vector;
}

/*
This function randomizes our matrix.
*/
void randomize_matrix(int size, double **matrix)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix[i][j] = drand48() * 200000 - 100000;
		}
	}
}

/*
This function randomizes our vectors.
*/
void randomize_vector(int size, double *vector)
{
	for (int i = 0; i < size; i++)
	{
		vector[i] = drand48() * 200000 - 100000;
	}
}

/*
This is a function that prints out a matrix. 
Great for debugging.
*/
void print_matrix(int size, double **matrix)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

/*
This is a function that prints out a vector. 
Great for debugging.
*/
void print_vector(int size, double *vector)
{
	for (int i = 0; i < size; i++)
	{
		cout << vector[i] << endl;
	}
}

/*
This function is the meat of the program.
Gauss elimination. Partial pivot is abstracted out, but forward elimination
and back substitution are both coded in this function.
*/
void gauss(int size, double **matrix, double *vector, double *solution)
{
	double multiplier, temp, result;

	for (int k = 0; k < size; k++)
	{
		partial_pivot(size, k, matrix, vector);
		#pragma omp parallel for shared(matrix, vector, size, k) private(multiplier)
			//forward elimination
			for (int i = k+1; i < size; i++)
			{
				multiplier = matrix[i][k]/matrix[k][k];
				for (int j = k+1; j < size; j++)
				{
					matrix[i][j] -= (multiplier * matrix[k][j]);
				}
				matrix[i][k] = 0;
				vector[i] -= (multiplier * vector[k]);
			}
	}

	//back substitution
	for (int i = size-1; i >= 0; i--)
	{
		result = vector[i];
		for (int j = size-1; j > i; j--)
		{
			result -= (matrix[i][j] * solution[j]);
		}
		solution[i] = result / matrix[i][i];
	}
}

/*
This function performs partial pivoting.
It checks the max abs value from the correct column and row down
and then swaps the max row with the working row.
Rather than use an augmented matrix, I elected to keep my vector and matrix
seperate and swap the appropriate vector values when the rows switch.
*/
void partial_pivot(int size, int pivot_row, double **matrix, double *vector)
{
	double max = fabs(matrix[pivot_row][pivot_row]);
	int max_row = 0;
	double temp = 0;

	//find max val
	for (int i = pivot_row; i < size; i++)
	{
		if (fabs(matrix[i][pivot_row]) > max)
		{
			max = matrix[i][pivot_row];
			max_row = i;
		}
	}
	//swap those rows

	if (max_row > pivot_row)
	{
		for (int i = 0; i < size; i++)
		{
			temp = matrix[max_row][i];
			matrix[max_row][i] = matrix[pivot_row][i];
			matrix[pivot_row][i] = temp;
		}
		temp = vector[max_row];
		vector[max_row] = vector[pivot_row];
		vector[pivot_row] = temp;
	}
}

/* This function checks if our solution is right.
It create a residual vector AX-b = 0 and then
it computes the L^2 norm. It is the square root of the sum of squares of all 
the values in the residual vector.
*/
double norm(double **a, double *b, double *solution, int size)
{
	//residual
	double *residual = allocate_vector(size);
	double val = 0;

	for (int i = 0; i < size; i++)
	{
		residual[i] = 0;
		for (int j = 0; j < size; j++)
		{
			residual[i] += (a[i][j] * solution[j]);
		}
		residual[i] -= b[i];
	}

	//l^2
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += pow(residual[i], 2);
	}
	return sqrt(sum);
}