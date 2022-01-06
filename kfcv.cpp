#include <iostream>
#include <vector>
#include <iterator>
#include <random>
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
	
  	/* Returns shuffled list of K Eigen::MatrixXd folds, split from input dataset. */

	int place = 0;
	
	//create temporary std::vector to hold rows of dataset
	std::vector<Eigen::Vector<double,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Vector<double,Eigen::Dynamic> > > temp;
	for(int i = 0; i < dataset.rows(); i++)
	{
		temp.push_back(dataset.row(i));
	}

	//shuffle std::vector
	auto random_number_generator = std::default_random_engine {};
	std::shuffle(std::begin(temp), std::end(temp), random_number_generator);

	//write shuffled rows a new shuffled matrix
	Eigen::MatrixXd shuffled(dataset.rows(),dataset.cols());
	for( auto v : temp )
	{
		shuffled.row(place++) = v;
	}

	place = 0;
	
	std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > list; // does not like not regular ints as arguments, e.g. row len and fold len	

	for(int i = 0; i < K; i++)
	{
		Eigen::MatrixXd fold(dataset.rows() / K,dataset.cols());

		for(int j = 0; j < dataset.rows() / K; j++)
		{
			Eigen::VectorXd x = shuffled.row(place++);
			fold.row(j) = x;
		}

		list.push_back(fold);
	}

	return list;
}

// double (*distance_function) (Eigen::VectorXd a, Eigen::VectorXd b, int length),
// std::vector<int> predictions = knn(test, test.rows(), train, train.rows(), K, *&distance_function);

double kfcv(Eigen::MatrixXd dataset, int K, std::vector<int> (*classifier) (Eigen::MatrixXd train, Eigen::MatrixXd validation, int optimal_parameter), double (*error_function) (std::vector<int> labels, std::vector<int> ground_truth_labels))
{
	/* Returns std::vector of error statistics from run of cross validation using given error function and classification function. */

	std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > > folds = split(dataset,K);

	double total_error = 0;

	int idx = 0;

	for(int i = 0; i < K; i++)
	{
		int length = dataset.rows() / K;
		int place = 0;

		Eigen::MatrixXd validation(length * 1,dataset.cols());
		Eigen::MatrixXd train(length * (K-1),dataset.cols());

		for(auto v : folds)
		{
			if(idx != K)
			{
				for(int i = 0; i < length; i++)
				{
					train.row(place++) = v.row(i);
				}
			}

			if(idx == K)
			{
				for(int i = 0; i < length; i++)
				{
					validation.row(place++) = v.row(i);
				}
			}

			idx = idx + 1;
		}

		std::vector<int> truth_labels;

		idx = 0;

		for(int i = 0; i < validation.rows(); i++)
		{
			truth_labels.push_back(validation.coeff(i,0));
		}	

		std::vector<int> predictions = classifier(train,validation,K);
		double error = error_function(predictions,truth_labels);
		total_error += error;
	}

	return total_error / K;
}

// compute_best_k in knn


