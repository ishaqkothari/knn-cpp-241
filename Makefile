TARGETS=knn
CC=g++ -std=c++11

all: $(TARGETS)

knn: knn.cpp
	$(CC) -o knn knn.cpp

clean:
	rm -rf $(TARGETS) *.o
