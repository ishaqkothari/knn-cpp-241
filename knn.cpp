#include "eigen3/Eigen/Geometry"
#include <iostream>
#include <vector>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include "utils.hpp"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

template<typename T> T load_csv(const std::string & sys_path)
{

  /* Returns csv file input as an Eigen matrix or vector. */

  std::ifstream in;
  in.open(sys_path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(in, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      while (std::getline(lineStream, cell, ',')) {
          values.push_back(std::stod(cell));
      }
      rows = rows + 1;
  }

  return Eigen::Map<const Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime, T::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

double EuclideanDistance(Eigen::VectorXd a, Eigen::VectorXd b, int length)
{

    /* Returns the Euclidean Distance between two vectors of the same feature length, a and b. */

    double sum = 0;

    for(int i = 0; i < length; i++)
    {
        sum += pow(a.coeff(i) - b.coeff(i),2);
    }

    return sqrt(sum);
}

double ManhattanDistance(Eigen::VectorXd a, Eigen::VectorXd b, int length)
{

    /* Returns the Manhattan Distance between two vectors of the same feature length, a and b. */

    double sum = 0;

    for(int i = 0; i < length; i++)
    {
        sum = sum + abs(a.coeff(i) - b.coeff(i));
    }

    return sum;

}

double ChebyshevDistance(Eigen::VectorXd a, Eigen::VectorXd b, int length)
{

	/* Returns the Chebyshev Distance between two vector of the same feature length, a and b. */

	Eigen::VectorXd ret(length);

	for(int i = 0; i < length; i++)
	{
		ret(i) = abs(a.coeff(i) - b.coeff(i));
	}

	return ret.maxCoeff();

}

std::vector<double> distances(Eigen::VectorXd vector, int vector_length, Eigen::MatrixXd X, int X_size, double (*distance_function) (Eigen::VectorXd a, Eigen::VectorXd b, int length))
{

    /* Returns the distances for one input vector to every point in matrix X vector where the last entry in each vector is its classification. */

    std::vector<double>distances_list = { };

    for(int i = 0; i < X_size; i++)
    {
        Eigen::VectorXd x = X.row(i);
        distances_list.push_back(distance_function(vector.tail(vector_length-1),x.tail(vector_length-1),vector_length-1));
    }

    return distances_list;
}

std::vector<int> knn(Eigen::MatrixXd M1, int M1_size, Eigen::MatrixXd M2, int M2_size, int K, double (*distance_function) (Eigen::VectorXd a, Eigen::VectorXd b, int length))
{

    /* Returns classifications of all instances in one dataset, M1, using another dataset, M2. */

    std::vector<int> predictions = { };

    for(int i = 0; i < M1_size; i++) // Computes the distances from one M1 row, x, to all vectors in M2 and store the classifications of the K smallest vectors.
    {
        std::vector<int> k_smallest_classifications = { };

        Eigen::VectorXd x = M1.row(i);
        std::vector<double> dists = distances(x,x.size()-1,M2,M2_size,*&distance_function);

      	std::vector<int> k_smallest = argpartition(dists, K); // Find the indicies of the K smallest distances
      	double k_smallest_arr[k_smallest.size()];
        std::copy(k_smallest.begin(), k_smallest.end(), k_smallest_arr);

        std::vector<int> labels = { }; // Create list of labels of K shortest distances.

        for(int j = 0; j < K; j++)
        {
            Eigen::VectorXd k_closest_vector = M2.row(k_smallest_arr[j]);
            labels.push_back(k_closest_vector.coeff(0));
        }

        int classification = plurality_class(labels); // Find the most commonly occuring label in list
        predictions.push_back(classification);
    }

    return predictions;
}

void driver(std::string sys_path_test, std::string sys_path_train, int K, double (*distance_function) (Eigen::VectorXd a, Eigen::VectorXd b, int length), bool verbose)
{

  /* Driver for a k nearest neighbors classifier. */

  Eigen::MatrixXd test = load_csv<Eigen::MatrixXd>(sys_path_test);

  if(verbose == true)
  {
      std::cout << "Test Data: " << sys_path_test << "\n";
      std::cout << test << "\n\n";
  }

  Eigen::MatrixXd train = load_csv<Eigen::MatrixXd>(sys_path_train);

  if(verbose == true)
  {
      std::cout << "Train Data: " << sys_path_train << "\n";
      std::cout << train << "\n\n";
  }

  std::vector<int> predictions = knn(test, test.rows(), train, train.rows(), K, *&distance_function);

  if(verbose == true)
  {
    int count = 0;
    for(auto v : predictions)
    {
        std::cout << "Vector " << count << " Classification = " << v << "\n";
        count++;
    }
    std::cout << "\n";
  }
}

int main()
{

    /* [Iris-virginica] => 0 [Iris-setosa] => 1 [Iris-versicolor] => 2 */

    driver("./data/iristest.csv", "./data/iris.csv", 11, &ManhattanDistance, true);
    driver("./data/iristest.csv", "./data/iris.csv", 11, &ChebyshevDistance, true);
    driver("./data/iristest.csv", "./data/iris.csv", 11, &EuclideanDistance, true);

    //driver("./data/S1test.csv", "./data/S1train.csv", 131, &EuclideanDistance, true);
    //driver("./data/S2test.csv", "./data/S2train.csv", 85, &EuclideanDistance, true);


    return 0;
}
