# K Nearest Neighbors Classifier Algorithm

## Authors
Nathan Englehart, Xuhang Cao, Samuel Topper, Ishaq Kothari (Autumn 2021)

## Usage
To clone and run this classifier so that it can be run on a dataset, please run the following. 

```
git clone git@github.com:nathanenglehart/knn-cpp-241.git
cd knn-cpp-241
make
```

The program is meant to be run as the below, where train and test are the paths to the train and test csv files.

```
./knn-cli [train] [test] [options...]
```

For a help menu, please run:

```
./knn-cli -h
```

To run this program in verbose mode, please run:

```
./knn-cli [train] [test] -v 
```
## Notes
This is currently a work in progress. We are currently including the eigen3 linear algebra library folder within this program. Note that example datasets are supplied in the data folder, e.g. iris.csv.

## References
David Barber, [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/171216.pdf), 2016. (pp. 313-317) <br>

Mark Girolami and Simon Rogers, [First Couse in Machine Learning - second edition](http://www.dcs.gla.ac.uk/~srogers/firstcourseml/), 2011.
