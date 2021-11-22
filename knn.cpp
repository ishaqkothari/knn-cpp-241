#include <iostream>
#include <list>
#include <cmath>
#include <eigen3/Eigen/Dense>

double EuclideanDistance(Eigen::VectorXf a, Eigen::VectorXf b, int length)
{

    /* Computes the Euclidean Distance between two vectors of the same feature length, a and b. */

    double sum = 0;
    Eigen::VectorXf ret(length);

    for(int i = 0; i < length; i++)
    {
        ret(i) = pow(a.coeff(i) - b.coeff(i),2);
    }

    for(int i = 0; i < length; i++)
    {
        sum = sum + ret(i);
    }

    return sqrt(sum);
}

double ManhattanDistance(Eigen::VectorXf a, Eigen::VectorXf b, int length)
{

    /* Computes the Manhattan Distance between two vectors of the same feature length, a and b. */

    double sum = 0;

    for(int i = 0; i < length; i++)
    {
        sum = sum + abs(a.coeff(i) - b.coeff(i));
    }

    return sum;

}

std::list<double> EuclideanDistances(Eigen::VectorXf vector, int vector_length, Eigen::MatrixXf X, int X_size)
{

    /* Computes the Euclidean Distances for one input vector to every training vector. */

    std::list<double>distances = { };
    int count = 0;
    for(int i = 0; i < X_size; i++)
    {
        Eigen::VectorXf x = X.row(i);
        distances.push_back (EuclideanDistance(vector,x,vector_length));
    }

    return distances;
}

std::list<double> ManhattanDistances(Eigen::VectorXf vector, int vector_length, Eigen::MatrixXf X, int X_size)
{

    /* Computes the Manhattan Distances for one input vector to every training vector. */

    std::list<double>distances = { };
    int count = 0;
    for(int i = 0; i < X_size; i++)
    {
        Eigen::VectorXf x = X.row(i);
        distances.push_back (ManhattanDistance(vector,x,vector_length));
    }

    return distances;
}


int main()
{
    // Testing
    int length = 3;
    int size = 6;
    Eigen::VectorXf v1(length);
    Eigen::VectorXf v2(length);
    Eigen::MatrixXf m1(size,length);

    v1 << 2,
          3,
          4;

    v2 << 5,
          9,
          5;

    m1 << 5, 6, 7,
          8, 9, 3,
          4, 7, 4,
          3, 2, 0,
          1, 1, 1, 
          6, 5, 8;

    double result = EuclideanDistance(v1,v2,length);
    std::cout << result << "\n";
    result = ManhattanDistance(v1,v2,length);
    std::cout << result << "\n";

    std::list<double> distances = EuclideanDistances(v1,length,m1,size);
    for (auto v : distances)
        std::cout << v << "\n";
    
    return 0;
}

