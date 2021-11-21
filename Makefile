TARGETS=main
CC=g++ -std=c++11

all: $(TARGETS)

main: main.cpp
	$(CC) -o main main.cpp

clean:
	rm -rf $(TARGETS) *.o
