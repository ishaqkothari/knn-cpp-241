#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include "eigen3/Eigen/Geometry"

template <typename T> T * to_array(std::vector<T>, int);
int plurality_class(std::vector<int> &);
std::vector<int> argpartition(std::vector<double>, int);

#endif
