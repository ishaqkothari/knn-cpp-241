#include <iostream>
#include <vector>
#include <iterator>
#include "includes/eigen3/Eigen/Dense"
#include "includes/eigen3/Eigen/StdVector"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

double misclassification_rate(std::vector<int> labels, std::vector<int> ground_truth_labels)
{

  /* Takes an array of labels and an array of ground truth labels and calculates the misclassification rate. */

  int incorrect = 0;

  std::vector<int>::iterator labels_it = labels.begin();
  std::vector<int>::iterator ground_truth_labels_it = ground_truth_labels.begin();

  for(; labels_it != labels.end() && ground_truth_labels_it != ground_truth_labels.end(); ++labels_it, ++ground_truth_labels_it)
  {
      if(*labels_it != *ground_truth_labels_it)
      {
        incorrect += 1;
      }
  }

  return incorrect / labels.size();
}

std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > split(Eigen::MatrixXd dataset, int K)
{
	
  	/* Returns list of K folds split from input dataset. */
  
	int place = 0;

	std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > list; // does not like not regular ints as arguments, e.g. row len and fold len	

	for(int i = 0; i < K; i++)
	{
		Eigen::MatrixXd fold(dataset.rows() / K,dataset.cols());

		for(int j = 0; j < fold_len; j++)
		{
			Eigen::VectorXd x = dataset.row(place++);
			fold.row(j) = x;
		}

		list.push_back(fold);
	}

	return list;
}

