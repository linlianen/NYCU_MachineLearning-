#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <cmath>
#include "../matplotlibcpp.h"
using namespace std;
namespace plt = matplotlibcpp;

vector<double> data22;

vector<double> temp_des;



void tokenize(string s, string del = " ")
{
	int count = 0;
	int start=0, end = -1 * del.size();

	do
	{
		start = end + del.size();
		end = s.find(del, start);

		if (end == -1)
		{   

			temp_des.push_back(std::stof(s.substr(start, end - start).c_str()));
		}else{
			data22.push_back(std::stof(s.substr(start, end - start).c_str()));

			continue;
		}

		count ++;

	}  while (end != -1);
}


//Transepose Procedure
vector<vector<double>> Transposematrix(vector<vector<double>> &m1)
{

	vector<vector<double>> transposeMatrix(m1[0].size(), vector<double>(m1.size(), 0));

	for (size_t i = 0; i < m1.size(); i++)
	{
		for (size_t j = 0; j < m1[0].size(); j++)
		{
			transposeMatrix[j][i] = m1[i][j];
		}vector<double> Inv_matrix_row;
	}

	return transposeMatrix;
}

// Inverse procedure (Gaussian Elimination)
vector<vector<double>> makeInvermatrix(vector<vector<double>> &matrix1)
{
	vector<vector<double>> Inv_matrix;

	vector<double> GE_matrix_row;
	vector<vector<double>> GE_matrix;
	vector<double> Inv_matrix_row;
	int dim = matrix1.size();
	double temp_value1;

	for (int r1 = 0; r1 < dim; r1++)
	{
		for (int c1 = 0; c1 < dim; c1++)
		{
			if (r1 == c1)
			{
				matrix1[r1].push_back(1);
			}
			else
			{
				matrix1[r1].push_back(0);
			}
		}
	}

	double ratio;
	for (int c2 = 0; c2 < dim; c2++)
	{
		if (matrix1[c2][c2] == 0.0)
		{
			cout << "Mathematical Error!";
			exit(0);
		}

		for (int r2 = 0; r2 < dim; r2++)
		{
			if (c2 != r2)
			{
				ratio =matrix1[r2][c2] / matrix1[c2][c2];
				for (int ct = 0; ct < 2 * dim; ct++)
				{
					matrix1[r2][ct] = matrix1[r2][ct] - ratio * matrix1[c2][ct];
				}
			}
		}
	}

	for (int r3 = 0; r3 < dim; r3++)
	{
		double div = matrix1[r3][r3];
		for (int c3 = 0; c3 < 2 * dim; c3++)
		{

			matrix1[r3][c3]= (double)matrix1[r3][c3] / div;


		}

	}

	for (int c4 = 0; c4 < dim; c4++)
	{

		for (int r4 = 0; r4 <  dim; r4++)
		{
			if(c4 != r4 && matrix1[r4][c4] != 0){
				ratio =matrix1[r4][c4];
				for (int ct = 0; ct < 2 * dim; ct++)
				{
					matrix1[r4][ct] = matrix1[r4][ct] - ratio * matrix1[c4][ct];
				}
			}


		}

	}

	for (size_t s = 0; s < matrix1.size(); s++)
	{
		for (size_t v = matrix1.size(); v < matrix1[s].size(); v++)
		{
			Inv_matrix_row.push_back(matrix1[s][v]);
		}

		Inv_matrix.push_back(Inv_matrix_row);
		Inv_matrix_row.clear();
	}

	return Inv_matrix;
}

//Make an identity Matrix
vector<vector<double>> IdentityMatrix(int dim, int lambda_sub)
{
	vector<vector<double>> identity(dim, vector<double>(dim, 0));

	for (int l = 0; l < dim; l++)
	{
		for (int u = 0; u < dim; u++)
		{
			for (int v = 0; v < dim; v++)
			{
				if (u == v)
				{
					identity[u][v] = lambda_sub;
				}
				else
				{
					identity[u][v] = 0;
				}
			}
		}
	}

	return identity;
}


//Matrix added
vector<vector<double>> addMatrix(vector<vector<double>> &m1, vector<vector<double>> &m2)
{
	vector<vector<double>> add_Matrix(m1[0].size(), vector<double>(m1.size(), 0));

	for (size_t i = 0; i < m1.size(); i++)
	{
		for (size_t j = 0; j < m1[0].size(); j++)
		{
			add_Matrix[i][j] = m1[i][j] + m2[i][j];
		}
	}

	return add_Matrix;
}


//Matrix multiplied
vector<vector<double>> multipleMatrix(vector<vector<double>> &m1, vector<vector<double>> &m2)
{
	vector<vector<double>> matrix_value;
	vector<double> tempMatrix;
	double temp_matrix_value;
	int dim1 = m1.size();

	int dim2 = m2[0].size();

	int dim3 = m2.size();

	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim2; j++)
		{
			for (int k = 0; k < dim3; k++)
			{
				temp_matrix_value += m1[i][k] * m2[k][j];
			}
			tempMatrix.push_back(temp_matrix_value);

			temp_matrix_value = 0;
		}

		matrix_value.push_back(tempMatrix);
		tempMatrix.clear();
	}

	return matrix_value;
}

int main(int argc, char const *argv[])
{

	ifstream file;
	string line;
	int dim = 3;
	int lambda = 0;
	vector<vector<double>> des_vector;
	vector<vector<double>> transposeMatrix_main;
	vector<vector<double>> inverseMatrix_main;
	vector<vector<double>> matrix;
	vector<double> temp_ex;
	cout << "Please input the dimension and lamba first" << endl;
	cout << "dimension :";
	cin >> dim;
	cout << endl;

	cout << "lambda: ";
	cin >> lambda;
	cout << endl;
	// // 資料處理
	file.open("testfile.txt", ios::in);

	if (file.is_open())
	{    
		cout<<"1239999"<<endl;

		while (getline(file, line))
		{

			tokenize(line, ",");
		}
		file.close();

	}
	vector<double> tempMatrix_des;

	for (size_t i = 0; i < data22.size(); i++)
	{
		for (int k = 0; k < dim; k++)
		{

			temp_ex.push_back(pow(data22[i], k));
		}

		tempMatrix_des.push_back(temp_des[i]);

		matrix.push_back(temp_ex);

		des_vector.push_back(tempMatrix_des);

		temp_ex.clear();

		tempMatrix_des.clear();

	}

	temp_ex.shrink_to_fit();
	tempMatrix_des.shrink_to_fit();

	//(ATA+lambda*I)x= ATb
	//------------------------------------

	//Transpose->AT
	transposeMatrix_main = Transposematrix(matrix);
	//ATA
	vector<vector<double>> multipleMatrix1 = multipleMatrix(transposeMatrix_main, matrix);
	//lambda*I
	vector<vector<double>> Identity_main = IdentityMatrix(multipleMatrix1.size(), lambda);
	//ATA+lamda*I
	vector<vector<double>> Lagrange_Matrix = addMatrix(multipleMatrix1, Identity_main);

	vector<vector<double>> solution;
	//(ATA+lambda*I)^(-1)
	vector<vector<double>> Lag_Inverse_Matrix = makeInvermatrix(Lagrange_Matrix);


	//(ATA+lambda*I)^(-1)*AT
	vector<vector<double>> temp_operate_matrix = multipleMatrix(Lag_Inverse_Matrix, transposeMatrix_main);

	//(ATA+lambda*I)^(-1)*AT*b
	solution = multipleMatrix((temp_operate_matrix), des_vector);


	cout << "dimension: "<<dim<<","<<"lambda: " << lambda << endl <<endl;

	cout << "LSE:" ;
	for (int vd = 0; vd < solution.size(); vd++)
	{   

		cout << solution[vd][0] ;
		if (vd != 0)
			cout << "x^" << vd ;
		if (vd != solution.size()-1)
			cout << " + ";

	}

	//LSE Error
	vector<float> function_y;
	double temp_y = 0.0;

	for (size_t j = 0; j < data22.size(); j++)
	{

		for (size_t k = 0; k < solution.size(); k++)
		{
			temp_y = temp_y + solution[k][0] * pow(data22[j], k);
		}
		function_y.push_back(temp_y);
		temp_y = 0;
	}

	float error_LSE = 0;

	for (size_t z = 0; z < function_y.size(); z++)
	{
		error_LSE += pow(temp_des[z] - function_y[z], 2);
	}

	cout << endl;
	cout<< "LSE_Error: "<<error_LSE <<endl;
	cout<<"------------"<<endl;
	//-------------------------

	//Newton_method
	vector<vector<double>> Newton_solution;
	vector<vector<double>> Newton_Inverse_Matrix = makeInvermatrix(multipleMatrix1);
	vector<vector<double>> temp_newton_matrix = multipleMatrix(Newton_Inverse_Matrix, transposeMatrix_main);
	vector<vector<double>> N_solution = multipleMatrix((temp_newton_matrix), des_vector);
	Newton_solution = multipleMatrix(matrix, N_solution);

	vector<float> newton_y;

	cout << "Newton_method: " ; 
	for (int nd = 0; nd < N_solution.size(); nd++)
	{   

		cout << N_solution[nd][0] ;
		if (nd != 0)
			cout << "x^" << nd ;
		if (nd != N_solution.size()-1)
			cout << " + ";

	}

	cout << endl;

	float error_Newton = 0;
	for (size_t n_i = 0; n_i < Newton_solution.size(); n_i++)
	{
		error_Newton += pow(Newton_solution[n_i][0] - temp_des[n_i], 2);
	}

	cout << "Newton_Error:" << error_Newton << endl;
	int n = 20;
	double plot_LSE = 0;
	std::vector<double> x(n), y(n,2), z(n), w(n);
	for(int i=0; i<n; ++i) {
		plot_LSE = 0;
		x.at(i) = i-10;

		for (size_t k = 0; k < solution.size(); k++)
		{   


			plot_LSE = plot_LSE + solution[k][0] * pow( i-10, k);


		}

		y.at(i) = plot_LSE;

	}

	//Plot LSE_image
	plt::figure_size(1200, 780);

	plt::plot(x,y, "r");
	plt::plot(data22, temp_des,"ob");

	plt::xlim(-10,10);
	plt::title("LSE figure");
	plt::save("./LSE.png");

	plt::show();

	//Plot NewtonImage
	plt::figure_size(1200, 780);
	double plot_Newton; 
	for(int i=0; i<n; ++i) {
		plot_Newton = 0;
		z.at(i) = i-10;

		for (size_t k = 0; k < N_solution.size(); k++)
		{   


			plot_Newton = plot_Newton + N_solution[k][0] * pow( i-10, k);


		}


		w.at(i) = plot_Newton;


	}

	plt::plot(z,w, "r");
	plt::plot(data22, temp_des,"ob");

	plt::xlim(-10,10);
	// // Add graph title
	plt::title("Newton figure");
	plt::save("./Newton.png");
	plt::show();

	return 0;
}
