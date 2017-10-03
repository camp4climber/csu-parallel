/*
Tim Whitaker
CSCI 551
Parallel matrix multiplication.
*/

#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
using namespace std;

void get_input(int rank, int *form, char *flag, int *size);
int **allocate_matrix(int size);
void init_matrix(char flag, int size, int **a, int **b);
void multiply_matrix(int form, int start, int interval, int size, int **a, int **b, int **c);
void print_matrix(int size, int **matrix);

int main()
{
	char flag;
	int form, matrix_size, interval, interval_remainder, local_start;
	int rank, comm_sz; 
	double start_time, end_time;
	int **a_matrix;
	int **b_matrix;
	int **c_matrix;

	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	get_input(rank, &form, &flag, &matrix_size);

    a_matrix = allocate_matrix(matrix_size);
	b_matrix = allocate_matrix(matrix_size);
	c_matrix = allocate_matrix(matrix_size);

	if (rank == 0) init_matrix(flag, matrix_size, a_matrix, b_matrix);

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) start_time = MPI_Wtime();

	MPI_Bcast(&(b_matrix[0][0]), matrix_size*matrix_size, MPI_INT, 0, MPI_COMM_WORLD);

	interval = matrix_size / comm_sz;
    interval_remainder = matrix_size % comm_sz;
    local_start = rank * interval;

    MPI_Scatter(&a_matrix[0][0], matrix_size * interval, MPI_INT, &a_matrix[local_start][0], matrix_size * interval, MPI_INT, 0, MPI_COMM_WORLD);
    multiply_matrix(form, local_start, interval, matrix_size, a_matrix, b_matrix, c_matrix);
    MPI_Gather(&c_matrix[local_start][0], matrix_size * interval, MPI_INT, &c_matrix[0][0], matrix_size * interval, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
    	//handle remainder
    	if (interval_remainder != 0)
    	{
    		multiply_matrix(form, ((comm_sz-1)*interval)+interval, interval_remainder, matrix_size, a_matrix, b_matrix, c_matrix);
    	}

		end_time = MPI_Wtime();

		cout << "Running on " << comm_sz << " processors.\n";
		cout << "Elapsed time = " << scientific << setprecision(6) << end_time - start_time << " seconds.\n";
		if (flag == 'i' || flag == 'I') print_matrix(matrix_size, c_matrix);
    }

	MPI_Finalize();

	free(a_matrix[0]);
	free(a_matrix);
	free(b_matrix[0]);
	free(b_matrix);
	free(c_matrix[0]);
	free(c_matrix);

	return 0;
}

//Get input values.
//We pass in the rank to check which process we're in and ensure only process 0 does i/o.
void get_input(int rank, int *form, char *flag, int *size)
{
	if (rank == 0) 
    {
    	int matrix_size;
    	char matrix_flag;
    	string matrix_form;

    	cout << "Form: ";
    	cin >> matrix_form;
    	
    	while (matrix_form != "ijk" &&
    		matrix_form != "ikj" &&
    		matrix_form != "kij")
    	{
    		cout << "Invalid form. Choose ijk, ikj, or kij.\n";
    		cout << "Form: ";
    		cin >> matrix_form;
    	}
    	
    	cout << "Flag: ";
    	cin >> matrix_flag;
    	
    	while (matrix_flag != 'R' && 
    		   matrix_flag != 'I')
    	{
    		cout << "Invalid flag. Choose R or I.\n";
    		cout << "Flag: ";
    		cin >> matrix_flag;
    	}

    	cout << "Matrix Size: ";
    	cin >> matrix_size;

    	if (matrix_form == "ijk") *form = 1;
    	else if (matrix_form == "ikj") *form = 2;
    	else if (matrix_form == "kij") *form = 3;
    	*flag = matrix_flag;
    	*size = matrix_size;
    }
    MPI_Bcast(form, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(size, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

//Allocate contiguous memory for our arrays. Contiguous memory will allow us to scatter/gather.
int **allocate_matrix(int size)
{
	int *data = (int *)malloc(size*size*sizeof(int));
    int **matrix = (int **)malloc(size*sizeof(int*));
    for (int i=0; i<size; i++)
        matrix[i] = &(data[size*i]);

    //fast way to initialize matrix to 0 instead of garbage left from malloc.
	memset(matrix[0], 0, size*size*sizeof(int));

    return matrix;
}

//Initialize our matrices. If flag is set to R, we randomise. If I, get input.
void init_matrix(char flag, int size, int **a, int **b)
{
	srand(time(NULL));
	if (flag == 'R')
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				a[i][j] = rand() % 10;
				b[i][j] = rand() % 10;
			}
		}
	}
	else if (flag == 'I')
	{
		cout << "A Matrix: \n";
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				cin >> a[i][j];
			}
		}

		cout << "B Matrix: \n";
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				cin >> b[i][j];
			}
		}
	}
}

//our matrix multplication function. form 1 == ijk, 2 == ikj and 3 == kij.
void multiply_matrix(int form, int start, int interval, int size,
                     int **a, int **b, int **c)
{
	if (form == 1)
	{
		for (int i = start; i < start+interval; i++)
		{
			for (int j = 0; j < size; j++)
			{
				for (int k = 0; k < size; k++)
				{
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
	else if (form == 2)
	{
		for (int i = start; i < start+interval; i++)
		{
			for (int k = 0; k < size; k++)
			{
				for (int j = 0; j < size; j++)
				{
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
	else if (form == 3)
	{
		for (int k = 0; k < size; k++)
		{
			for (int i = start; i < start+interval; i++)
			{
				for (int j = 0; j < size; j++)
				{					
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
}

void print_matrix(int size, int **matrix)
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
