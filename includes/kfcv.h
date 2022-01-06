#ifndef UTILS_H
#define UTILS_H

double misclassification_rate(std::vector<int> labels, std::vector<int> ground_truth_labels);
std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > split(Eigen::MatrixXd dataset, int K);
double kfcv(Eigen::MatrixXd dataset, int K, (*classifier) (Eigen::MatrixXd train, Eigen::MatrixXd validation, int optimal_parameter), (*error_function) (std::vector<int> labels, std::vector<int> ground_truth_labels));

#endif
