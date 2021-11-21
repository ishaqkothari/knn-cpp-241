#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>
#include <numeric>

using namespace std;

template <typename T>
double EuclideanDistance(const std::vector<T>& a, const std::vector<T>& b)
{

    /* Computes the Euclidean Distance between two vectors of the same feature length, a and b. */

	std::vector<double>	ret;

	std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(ret),
	[](T feature_1, T feature_2) {
        return pow((feature_1 - feature_2),2);
    });

	return std::sqrt(std::accumulate(ret.begin(), ret.end(), 0.0));
} 

template <typename T>
double ManhattanDistance(const std::vector<T>& a, const std::vector<T>& b)
{

    /* Computes the Manhattan Distance between two vectors of the same feature length, a and b. */

    std::vector<double> ret;

    std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(ret),
	[](T feature_1, T feature_2) {
        return abs (feature_1 - feature_2);
    });

    return std::accumulate(ret.begin(), ret.end(), 0.0);
}

int main()
{

    std::vector<double> vector_1 = {4,4,6};
    std::vector<double> vector_2 = {100,7,-9};
    std::cout << EuclideanDistance(vector_1,vector_2) << "\n";
    std::cout << ManhattanDistance(vector_1,vector_2) << "\n";

    return 0;

}
