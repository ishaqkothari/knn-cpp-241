#include "eigen3/Eigen/Geometry"
#include <iostream>
#include <vector>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include "utils.hpp"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

template <typename T> T * to_array(std::vector<T> list, int length)
{

    /* Returns the input vector, list, converted to an array. */

    T * array = (T *) malloc(sizeof(T) * (length));
    std::copy(list.begin(), list.end(), array);
    return array;
}

int plurality_class(std::vector<int> &classifications)
{

    /* Returns the most common classification in vector. */

    if (classifications.empty())
    {
        return -1;
    }

    sort(classifications.begin(), classifications.end());

    auto last = classifications.front();
    auto most_frequent = classifications.front();

    int max_frequency = 0;
    int current_frequency = 0;

    for (const auto &i : classifications)
    {
        if (i == last)
        {
            ++current_frequency;
        } else {
            if (current_frequency > max_frequency)
            {
                max_frequency = current_frequency;
                most_frequent = last;
            }

            last = i;
            current_frequency = 1;
        }
    }

    if (current_frequency > max_frequency)
    {
        max_frequency = current_frequency;
        most_frequent = last;
    }

    return most_frequent;
}


std::vector<int> argpartition(std::vector<double> list, int N)
{

    /* Returns an array of the N smallest indicies of a list with numerical entries. */

    std::vector<int> indicies = { };

    double mins [N];

    double * list_arr;
    list_arr = to_array(list, 1 + list.size());
    int arr_size = list.size();

    double min = list_arr[0];
    int min_idx = 0;

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

        mins[i] = list_arr[min_idx];

        for (int j = min_idx; j < arr_size; ++j) // Remove min element and shrink array.
        {
            list_arr[j] = list_arr[j + 1];
        }

        arr_size = arr_size - 1;

        min = list_arr[0];
        min_idx = 0;
    }

    double * list_arr_copy = to_array(list, 1+list.size()); // Reset array to original.

    int visited [list.size()];

    for(int i = 0; i < list.size(); i++)
    {
        visited[i] = 0;
    }

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < list.size(); j++)
        {
            if(mins[i] == list_arr_copy[j] && visited[j] == 0)
            {
                indicies.push_back(j);
                visited[j] = 1;
                break;
            }
        }
    }

    free(list_arr);
    free(list_arr_copy);

    return indicies;
}
