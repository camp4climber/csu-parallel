/*
Tim Whitaker
CSCI 551
Serial program to find the number of trapezoids to accurately approximate a crazy function.
*/

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

double get_area(double lower_limit, double upper_limit);
double evaluate_height(double x);
int number_of_digits_correct(string calculated);

int main(int argv, char* argc[])
{
	int trap_number = 1;
	int range_to_test = 100;
	int big_increment = 100000;
	bool found_number = false;
	double trap_width, sum, current;
	int number_correct, winning_number;
	string calculated;

	while (1)
	{
		//This loop will test numbers in blocks based on range_to_test and big_increment.
		//e.g. 0-100. 100001-100100. 200001-200100.
		for (int i = trap_number; i < (trap_number + range_to_test); i++)
		{
			//our integral is from 100-600.
			//trap width will be this number divided by how many trapezoids we have.
			trap_width = 500.0/i;
			sum = 0;
			current = 100;
			number_correct = 0;
			stringstream ss;
			stringstream error;

			//this will get the area for our approximation by summing all of our trapezoids.
			for (int j=0; j<i; j++)
			{
				sum += get_area(current, current+trap_width);
				current = current+trap_width;
			}

			//need to use stringstream to compare the numbers and keep precision.
			//passing doubles to functions loses precision.
			ss << setprecision(20) << sum; 
			calculated = ss.str();
			number_correct = number_of_digits_correct(calculated);
			cout << i << ": " << number_correct << endl;
			//our goal case. since we compare strings, the decimal is "counted"
			//we make our number correct 15 or higher to account for the decimal.
			if (number_correct > 14)
			{
				winning_number = i;
				found_number = true;

				//outputting our found info.
				cout << "Number of Trapezoids: " << winning_number << endl;
				cout << "Found Value: " << setprecision(20) << fixed << calculated << endl;
				cout << "Relative True Error: ";
				cout << calculated[number_correct] << "." 
				     << calculated.substr(number_correct+1, 2) << "e-" << number_correct << endl;
				break;
			}
		}
		if (found_number) break;
		trap_number += big_increment;
	}
	return 0;
}

int number_of_digits_correct(string calculated)
{
	const string actual = "4003.7209001513268265";
	int total = 0;
	for (int i = 0; i < 21; i++)
	{
		if (calculated[i] == actual[i])
			total++;
		else
			break;
	}
	return total;
}

double get_area(double lower_limit, double upper_limit)
{
	double width = upper_limit - lower_limit;
	double h1 = evaluate_height(lower_limit);
	double h2 = evaluate_height(upper_limit);
	return (((h1 + h2)/2.0) * (width));
}

double evaluate_height(double x)
{
	return (cos(x/3)-(2*cos(x/5))+(5*sin(x/4))+8);
}