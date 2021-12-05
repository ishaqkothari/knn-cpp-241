#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include "processed.cleveland.data"
#include <fstream>
#include <algorithm>
#include <armadillo>

template <typename M>
M load_csv_arma (const std::string & path) {
    arma::mat X;
    X.load(path, arma::csv_ascii);
    return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}
int main(){
  Eigen::VectorXd name = load_csv_arma("processed.cleveland.data");
  std::cout<<(name);
}
