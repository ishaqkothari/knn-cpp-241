#include <iostream>
#include <vector>
#include <cmath>
#include "includes/eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include "includes/utils.h"
#include "includes/knn.h"

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

bool valid_filepath(const std::string & sys_path)
{
  std::ifstream test(sys_path);
  if(!test)
  {
    return false;
  }

  return true;
}

int main(int argc, char **argv)
{

if(argc == 2)
  {
    if(argv[1][0]== '-' && argv[1][1]== 'h' && argv[1][2] == '\0')
      {
        std::cout << "Help Menu \n";
	std::cout << "KNN cli (2021 Dec 8, compiled " << __TIMESTAMP__ << " " << __TIME__ << ")\n\n";
        std::cout << "usage: ./knn-cli [train] [test] [options ..]    read in train csv and test csv files from filesystem\n";
        std::cout << "   or: ./knn-cli -h                             displays help menu\n\n";
        std::cout << "Arguments:\n";
        std::cout << "   -h     Displays help menu\n";
        std::cout << "   -v     Displays output in verbose mode\n";
	return 0;
      }
    else
      {
        std::cout << "error.usage ./knn-cli[train][test][options...]\n";
	std::cout << "enter ./knn-cli -h for a help menu \n";
	return 1;
      }

  }
else if(argc == 4)
  {
    if(argv[3][0] == '-' && argv[3][1] == 'v' && argv[3][2] == '\0')
      {
        if(valid_filepath(argv[1]) && valid_filepath(argv[2]))
          {
            driver(argv[2],argv[1],5,&EuclideanDistance,true);
            return 0;
          }
        else if(!valid_filepath(argv[1]))
          {
            std::cout << "Invalid filepath: " << argv[1] << "\n";
            return 1;
          } 
        else if(!valid_filepath(argv[2]))
          {
            std::cout << "Invalid filepath: " << argv[2] << "\n";
            return 1;
          }	
      }
    else
      {
        std::cout << "error.usage ./knn-cli[train][test][options...]\n";
        std::cout << "enter ./knn-cli [train] [test] -v to run program in verbose\n";
	return 1;
      }
  }
else if(argc == 3)
  {
    if(valid_filepath(argv[1]) && valid_filepath(argv[2]))
      {
        driver(argv[2],argv[1],5,&EuclideanDistance,false);
        return 0;
      }
    else if(!valid_filepath(argv[1]))
      {
        std::cout << "Invalid filepath: " << argv[1] << "\n";
        return 1;
      }
    else if(!valid_filepath(argv[2]))
      {
        std::cout << "Invalid filepath: " << argv[2] << "\n";
        return 1;
      }
    else
      {
        std::cout << "error.usage ./knn-cli[train][test][options...]\n";
        std::cout << "enter valid files in the form: knn-cli [train] [test]\n";
	return 1;
      }
  }
else
  {
    std::cout << "error.usage ./knn-cli[train][test][options...]\n";
    std::cout << "input is not recognized -- please enter in the form: knn-cli[train][test][options...]\n";
    return 1;
  }

    /* [Iris-virginica] => 0 [Iris-setosa] => 1 [Iris-versicolor] => 2 */

    driver("./data/iristest.csv", "./data/iris.csv", 11, &ManhattanDistance, true);
    driver("./data/iristest.csv", "./data/iris.csv", 11, &ChebyshevDistance, true);
    driver("./data/iristest.csv", "./data/iris.csv", 11, &EuclideanDistance, true);

    //driver("./data/S1test.csv", "./data/S1train.csv", 131, &EuclideanDistance, true);
    //driver("./data/S2test.csv", "./data/S2train.csv", 85, &EuclideanDistance, true);


    return 0;
}
