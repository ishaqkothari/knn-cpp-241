TARGETS=main
CXX=g++ -std=c++11

all: $(TARGETS)

main: utils.hpp utils.o knn.o main.o
	$(CXX) utils.o knn.o main.o -o main

knn.o: utils.hpp knn.cpp
	$(CXX) -c knn.cpp

utils.o: utils.hpp utils.cpp
	$(CXX) -c utils.cpp

clean:
	rm -rf $(TARGETS) *.o *.gch
