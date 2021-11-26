TARGETS=knn
CXX=g++ -std=c++11

all: $(TARGETS)

knn: knn.cpp
	$(CXX) -o knn knn.cpp
	
clean:
	rm -rf $(TARGETS) *.o
