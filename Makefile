TARGETS=knn	datatransform
CXX=g++ -std=c++11

all: $(TARGETS)

knn: knn.cpp
	$(CXX) -o knn knn.cpp

datatransform: datatransform.cpp
		$(CXX) -o datatransform datatransform.cpp

clean:
	rm -rf $(TARGETS) *.o
