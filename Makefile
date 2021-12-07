TARGETS=knn
CXX=g++ -std=c++11

all: $(TARGETS)

knn: utils.hpp utils.o knn.o
	$(CXX) utils.o knn.o -o knn

knn.o: utils.hpp knn.cpp
	$(CXX) -c knn.cpp

utils.o: utils.hpp utils.cpp
	$(CXX) -c utils.cpp

clean:
	rm -rf $(TARGETS) *.o *.gch
