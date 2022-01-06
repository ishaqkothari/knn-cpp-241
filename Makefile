TARGETS=knn-cli
CXX=g++ -std=c++11
INC=-I./includes

all: $(TARGETS)

knn-cli: utils.o knn.o main.o
	$(CXX) $(INC) utils.o knn.o main.o -o knn-cli

knn.o: utils.o includes/knn.h knn.cpp
	$(CXX) $(INC) -c knn.cpp

utils.o: includes/utils.h utils.cpp
	$(CXX) $(INC) -c utils.cpp

clean:
	rm -rf $(TARGETS) *.o *.gch
