#include <iostream>
#include <vector>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include <cstdlib>
void crossValidate(Eigen::VectorXd matrix,int matrixsize,int K){
  std::vector<int> perm;
PermutationMatrix<Dynamic,Dynamic> perm(matrixsize);
perm.setIdentity();
std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
perm = matrix * perm; // permute columns
shuffled_matrix = matrix * A; // permute rows
std::cout << shuffled_matrix; 
int main(){
  static Eigen::Matrix4d foo = (Eigen::Matrix4d() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16).finished();

  crossValidate(foo,4,3)

}
