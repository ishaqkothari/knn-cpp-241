#include "eigen3/Eigen/Geometry"
#include <iostream>
#include <vector>
#include <cmath>
#include "eigen3/Eigen/Dense"
#include <fstream>
#include <algorithm>
#include "utils.hpp"

/* Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021) */

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
