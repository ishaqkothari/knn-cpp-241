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


## Prerequisites
Before installing knn-cli, you must have installed gnuplot and boost on your computer. To do so, here are the appropriate package manager commands for various operating systems.

Arch Linux

```
sudo pacman -S boost
sudo pacman -S gnuplot
```

Ubuntu

```
sudo apt-get install boost
sudo apt-get install gnuplot
```

Mac

```
brew install boost
brew install gnuplot
```


## Installation
To install this program to your posix standard system, please run the following.

```
git clone git@github.com:nathanenglehart/knn-cpp-241.git
cd knn-cpp-241
make
sudo cp knn-cli /usr/local/bin/knn-cli
sudo chmod 0755 /usr/local/bin/knn-cli
```

The program can then be run from any location on your system, as in the below.

```
knn-cli [train] [test] [options...]
```

## Uninstall
To uninstall this program from your system, run the following.

```
sudo rm /usr/local/bin/knn-cli
```

## Notes
This is currently a work in progress. We are currently including the eigen3 linear algebra library folder within this program. Note that example datasets from the UCI machine learning repository are supplied in the data folder, e.g. data/iris/iris.csv.

## References
David Barber, [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/171216.pdf), 2016. (pp. 313-317) <br>

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.

Mark Girolami and Simon Rogers, [First Couse in Machine Learning - second edition](http://www.dcs.gla.ac.uk/~srogers/firstcourseml/), 2011.


