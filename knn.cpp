#include <iostream>
#include <list>
#include <cmath>
#include <vector>
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

double ChebyshevDistance(Eigen::VectorXf a, Eigen::VectorXf b, int length)
{

	/* Computes the Chebyshev Distance between two vector of the same feature length, a and b. */

	Eigen::VectorXf ret(length);

	for(int i = 0; i < length; i++)
	{
		ret(i) = abs(a.coeff(i) - b.coeff(i));
	}

	return ret.maxCoeff();

}

std::list<double> distances(Eigen::VectorXf vector, int vector_length, Eigen::MatrixXf X, int X_size, double (*distance_function) (Eigen::VectorXf a, Eigen::VectorXf b, int length))
{

    /* Computes the distances for one input vector to every point in matrix X vector where the last entry in each vector is its classification. */
 
    std::list<double>distances_list = { };
    int count = 0;

    for(int i = 0; i < X_size; i++)
    {
        Eigen::VectorXf x = X.row(i);
        distances_list.push_back(distance_function(vector.head(vector_length-1),x.head(vector_length-1),vector_length-1));
    }

    return distances_list;
}

int plurality_class(std::list<int> class_list, int K)
{

    /* Calculates the most common classification in list containing K classifications. */


    /* Convert classification list to array. */

    int arr[K];
    int idx = 0;

    for(int const &i: class_list) 
    {
        arr[idx++] = i;
    }

    /* Find max occurence in classification array. */

    int most_frequent = arr[0];
    int max_count = 0;

    for (int i=0; i<K; i++)
    {
        int count=1;

        for (int j=i+1;j<K;j++)
        {
            if (arr[i]==arr[j])
            {
                count++;
            }
        }

        if (count>max_count)
        {
            max_count = count;
        }
    }

    for (int i=0;i<K;i++)
    {
        int count=1;

        for (int j=i+1;j<K;j++)
        {
            if (arr[i]==arr[j])
            {
                count++;
            }
        }

        if (count==max_count)
        {
            most_frequent = arr[i];
        }
    }

    return most_frequent;
}

std::list<int> argpartition(std::list<double> list, int N)
{
    
    /* Returns an array of the N smallest indicies of a list. Based on the NumPy argpartition() function. */

    std::list<int> indicies;

    int arr_size = list.size();
    double list_arr[arr_size];
    std::copy(list.begin(), list.end(), list_arr);  

    double mins [N];
    double min = list_arr[0];
    int min_idx = 0;
    int count = 0;

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < arr_size; j++)
        {
            if(list_arr[j] < min)
            {
                min = list_arr[j];
                min_idx = j;
            }
        } 

        mins[count] = list_arr[min_idx];

        for (int i = min_idx; i < arr_size; ++i)
        {
            list_arr[i] = list_arr[i + 1];
        }

        arr_size = arr_size - 1;
        min = list_arr[0];
        min_idx = 0;
        count++;
        
    }

    
    std::copy(list.begin(), list.end(), list_arr);

    int visited [list.size()-1];
    
    for(int i = 0; i < list.size(); i++)
    {
        visited[i] = 0;
    }
    
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < list.size(); j++)
        {
            if(mins[i] == list_arr[j] && visited[j] == 0)
            {
                indicies.push_back(j);
                visited[j] = 1;
                break;
            }
        }
    }

    return indicies;
}

std::list<int> knn(Eigen::MatrixXf input, Eigen::MatrixXf dataset, int dataset_size, int K, double (*distance_function) (Eigen::VectorXf a, Eigen::VectorXf b, int length))
{

    /* Classifies all instances of one dataset using another dataset. */


    std::list<int> predictions = { };


    /* Error Catching */

    if(K > dataset_size)
    {
        std::cout << "K must not be greater than the size of your dataset of size " << dataset_size << "\n";
        return predictions;
    }

    if(K <= 0)
    {
        std::cout << "K must be greater than 0.\n";
        return predictions;
    }

    for(int i = 0; i < dataset_size; i++)
    {

        std::list<int> k_smallest_classifications = { };


        /* Compute the distances from one input vector to all points in dataset. */

        Eigen::VectorXf x = input.row(i);
        std::list<double> dists = distances(x,x.size()-1,dataset,dataset_size,*&distance_function);


        /* Find the indicies of the K smallest distances. */

        std::list<int> k_smallest = argpartition(dists, K);
        double k_smallest_arr[k_smallest.size()];
        std::copy(k_smallest.begin(), k_smallest.end(), k_smallest_arr);


        /* Create list of labels of K shortest distances. */

        int counter = 0;
        std::list<int> labels = { };
        for(int j = 0; j < K; j++)
        {
            Eigen::VectorXf k_closest_vector = dataset.row(k_smallest_arr[j]);
            labels.push_back(k_closest_vector.coeff(k_closest_vector.size()-1));
        }

        int classification = plurality_class(labels, K);
        predictions.push_back(classification);
    }

    return predictions;
}



int main()
{

    /* Testing */    

    int length = 4;
    int size = 6;
    Eigen::VectorXf v1(length);
    Eigen::VectorXf v2(length);
    Eigen::MatrixXf m1(size,length);
    Eigen::MatrixXf m2(size,length);
    
    v1 << 2, 3, 4, 0;

    v2 << 5, 9, 5, 0;

    /* Input Matrix */ 

    m1 << 5, 6, 7, 1,
          8, 9, 3, 1,
          4, 7, 4, 1,
          3, 2, 0, 1,
          1, 1, 1, 1,
          6, 5, 8, 1;

    /* Dataset Matrix */ 

    m2 << 5, 6, 7, 0,
          8, 9, 3, 0,
          4, 7, 4, 1,
          3, 2, 0, 1,
          1, 1, 1, 1,
          6, 5, 8, 1;
  
    int K = 3;
    std::list<int> predictions = knn(m1, m2, size, K, &EuclideanDistance);

    int count = 1;
    for(auto v : predictions)
    {
        std::cout << "Classification: " << "Vector" << count << ": " << v << "\n";
        count++;
    }

    return 0;
}

