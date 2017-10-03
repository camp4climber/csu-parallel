/*
Tim Whitaker
CSCI 551
Parallel integral approximation using a whole bunch of trapezoids.
*/

#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <math.h>
#include <sstream>
using namespace std;

void get_input(int rank, int comm_sz, double* lower_bound_p, double *upper_bound_p, int *num_traps_p);
double get_area(double local_lower_bound, double local_upper_bound, int traps_per_process, double trap_width);
double evaluate_height(double x);
string calculate_error(string calculated);

int main()
{
	int rank, comm_sz, num_traps, traps_per_process, process;
	double lower_bound, upper_bound, local_lower_bound, local_upper_bound; 
	double local_integral, total_integral, trap_width;
	double start_time, end_time;
	double error;
	stringstream ss;
	string calculated;

	MPI_Init(NULL, NULL);

	//rank processes
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	//get lower and upper bounds and number of trapezoids
	get_input(rank, comm_sz, &lower_bound, &upper_bound, &num_traps);

	//for timing
	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime();

	//figure out trapezoid width and number of trapezoids per process
	trap_width = (upper_bound-lower_bound)/num_traps;
	traps_per_process = num_traps/comm_sz;

	//get areas for local values
	local_lower_bound = lower_bound + rank * traps_per_process * trap_width;
	local_upper_bound = local_lower_bound + traps_per_process * trap_width;
	local_integral = get_area(local_lower_bound, local_upper_bound, traps_per_process, trap_width);

	if (rank != 0)
	{
		//send local_integral to process 0
		MPI_Send(&local_integral, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		//sum up all local integrals into total integral
		total_integral = local_integral;
		for (process = 1; process < comm_sz; process++)
		{
			MPI_Recv(&local_integral, 1, MPI_DOUBLE, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			total_integral += local_integral;
		}

		//use stringstream to keep high precision.
		ss << setprecision(20) << total_integral;
		calculated = ss.str();

		//for timing
		end_time = MPI_Wtime();
	}

	//output
	if (rank == 0)
	{
		cout << "Running on " << comm_sz << " processors.\n";
		cout << "Elapsed time = " << scientific << setprecision(6) << end_time - start_time << " seconds.\n";
		cout << "With n = " << num_traps << " trapezoids.\n";
		cout << "Our estimate of the integral from " << fixed << lower_bound << " to " << upper_bound << " = " << scientific << setprecision(14) << total_integral << ".\n";
		cout << "Absolute relative true error = " << calculate_error(calculated) << endl;
	}


	MPI_Finalize();

	return 0;
}

//Get input values for lower bound, upper bound and number of trapezoids.
//We pass in the rank to check which process we're in and ensure only process 0 does i/o.
void get_input(int rank, int comm_sz, double *lower_bound_p, double *upper_bound_p, int *num_traps_p) 
{
	int destination;
	if (rank == 0) 
    {
    	printf("Enter a, b, and n\n");
        scanf("%lf %lf %d", lower_bound_p, upper_bound_p, num_traps_p);
        for (destination = 1; destination < comm_sz; destination++)
        {
        	MPI_Send(lower_bound_p, 1, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD);
        	MPI_Send(upper_bound_p, 1, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD);
        	MPI_Send(num_traps_p, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
    	MPI_Recv(lower_bound_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(upper_bound_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	MPI_Recv(num_traps_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

//This function is used to get the total area for a local range for a given process.
double get_area(double local_lower_bound, double local_upper_bound, int traps_per_process, double trap_width)
{
	double estimate, x;
	double h1 = evaluate_height(local_lower_bound);
	double h2 = evaluate_height(local_upper_bound);
	estimate = (h1 + h2)/2.0;
	for (int i = 1; i <= traps_per_process-1; i++)
	{
		x = local_lower_bound + i * trap_width;
		estimate += evaluate_height(x);
	}
	estimate = estimate * trap_width;
	return estimate;
}

//This is the given function we're estimating an integral for.
//By passing in a value, we can get the height for an endpoint of the trapezoid.
double evaluate_height(double x)
{
	return (cos(x/3)-(2*cos(x/5))+(5*sin(x/4))+8);
}

//calculate our true relative error
string calculate_error(string calculated)
{
	const string actual = "4003.7209001513268265";
	stringstream error;
	int total = 0;
	for (int i = 0; i < 21; i++)
	{
		if (calculated[i] == actual[i])
			total++;
		else
			break;
	}

	error << calculated[total] << "." << calculated.substr(total+1, 2) << "e-" << total;
	return error.str();
}