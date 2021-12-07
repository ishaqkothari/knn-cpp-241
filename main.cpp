#include <iostream>
#include <vector>
#include <cmath>
#include "includes/eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include "includes/utils.hpp"
#include "includes/knn.hpp"

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
