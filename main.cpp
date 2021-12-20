#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "includes/eigen3/Eigen/Dense"
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

  Eigen::MatrixXd train = load_csv<Eigen::MatrixXd>(sys_path_train);
  Eigen::MatrixXd test = load_csv<Eigen::MatrixXd>(sys_path_test);

  if(verbose == true)
  {
    std::cout << "K = " << K << "\n";
  }

  if(K > train.rows())
  {
    std::cout << "Invalid K value. K must not exceed train dataset length.\n";
    exit(1);
  }

  if(verbose == true)
  {
      std::cout << "Train data filepath: " << sys_path_train << "\n";
      //std::cout << "Train Data: " << sys_path_train << "\n";
      //std::cout << train << "\n\n";
  }

  if(verbose == true)
  {
      std::cout << "Test data filepath: " << sys_path_test << "\n";
      //std::cout << "Test Data: " << sys_path_test << "\n";
      //std::cout << test << "\n\n";
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

int main(int argc, char **argv)
{

  /* Default parameters are K = 1, distance function is set to Euclidean Distance, and verbose is set to false. */

  int K = 1;
  auto distance_function = &EuclideanDistance;
  bool verbose = false;

  if(argc == 1)
  {
    std::cout << "No arguments supplied.\n";
    std::cout << "Usage: ./knn-cli [train] [test] [options ..]\n";
    std::cout << "More info with: \"./knn-cli -h\"\n";
    return 1;
  }

  int counter = 1;

  while(counter < argc)
  {
    if(argv[counter][0] == '-' && argv[counter][1] == 'h') //&& argv[counter][2] == '\0'
    {
      std::cout << "K Nearest Neighbors Cli (2021 Dec 9, compiled " << __TIMESTAMP__ << " " << __TIME__ << ")\n\n";
      std::cout << "usage: ./knn-cli [train] [test] [options ..]    read in train csv and test csv files from filesystem\n";
      std::cout << "   or: ./knn-cli -h                             displays help menu\n\n";
      std::cout << "Arguments:\n";
      std::cout << "   -h     Displays help menu\n";
      std::cout << "   -v     Displays output in verbose mode\n";
      std::cout << "   -e     Runs algorithm using the Euclidean Distance formula\n";
      std::cout << "   -m     Runs algorithm using the Manhattan Distance formula\n";
      std::cout << "   -c     Runs algorithm using the Chebyshev Distance formula\n";
      return 0;
    } else if(counter == 1 && !(valid_filepath(argv[1])))
    {
      std::cout << "Invalid filepath: " << argv[1] << "\n";
      std::cout << "Usage: ./knn-cli [train] [test] [options ..]\n";
      return 1;
    } else if(counter == 2 && !(valid_filepath(argv[1])))
    {
      std::cout << "Invalid filepath: " << argv[1] << "\n";
      std::cout << "Usage: ./knn-cli [train] [test] [options ..]\n";
      return 1;
    } else if(counter == 1 && (valid_filepath(argv[1])))
    {
      //std::cout << "Train filepath: " << argv[1] << "\n";
    } else if(counter == 2 && (valid_filepath(argv[2])))
    {
      //std::cout << "Test filepath: " << argv[2] << "\n";
    } else if(counter >= 3)
    {

      if(argv[counter][0] == '-' && argv[counter][1] == 'v' && argv[counter][2] == '\0')
      {
        verbose = true;
      } else if(argv[counter][0] == '-' && argv[counter][1] == 'e' && argv[counter][2] == '\0')
      {
        distance_function = &EuclideanDistance;
      } else if(argv[counter][0] == '-' && argv[counter][1] == 'm' && argv[counter][2] == '\0')
      {
        distance_function = &ManhattanDistance;
      } else if(argv[counter][0] == '-' && argv[counter][1] == 'c' && argv[counter][2] == '\0')
      {
        distance_function = &ChebyshevDistance;
      } else if(argv[counter][0] == '-' && argv[counter][1] == 'k' && argv[counter][2] == '\0')
      {
        if(argv[counter + 1] != NULL)
        {
          if(is_digits(argv[counter + 1]))
          {
            std::stringstream converter(argv[counter + 1]);
            K = 0;
            converter >> K;
            counter = counter + 1;
          } else
          {
            std::cout << "Unknown K value: " << argv[counter + 1] << "\n";
            std::cout << "More info with: \"./knn-cli -h\"\n";
            return 1;
          }
        } else
        {
          std::cout << "Unknown K value: \n";
          std::cout << "More info with: \"./knn-cli -h\"\n";
          return 1;
        }
      } else
      {
        std::cout << "Unknown option argument: " << argv[counter] << "\n";
        std::cout << "More info with: \"./knn-cli -h\"\n";
        return 1;
      }
    } else
    {
      std::cout << "Unknown option argument: " << argv[counter] << "\n";
      std::cout << "More info with: \"./knn-cli -h\"\n";
      return 1;
    }



    counter = counter + 1;
  }



  driver(argv[2],argv[1],K,distance_function,verbose);

  /* [Iris-virginica] => 0 [Iris-versicolor] => 1 [Iris-setosa] => 2 */

  //driver("./data/iristest.csv", "./data/iris.csv", 11, &ManhattanDistance, true);
  //driver("./data/iristest.csv", "./data/iris.csv", 11, &ChebyshevDistance, true);
  //driver("./data/iristest.csv", "./data/iris.csv", 11, &EuclideanDistance, true);

  //driver("./data/S1test.csv", "./data/S1train.csv", 131, &EuclideanDistance, true);
  //driver("./data/S2test.csv", "./data/S2train.csv", 85, &EuclideanDistance, true);


  return 0;
}
