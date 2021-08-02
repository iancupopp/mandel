all:
	g++ -std=c++17 -O3 -march=native -fopenmp -o mandel main.cpp -lSDL2 -lSDL2main
clean:
	rm -f mandel 

