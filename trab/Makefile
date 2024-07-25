serial:
	g++ -std=c++17 -pg -Wall -Wextra -march=native -fopt-info-vec-missed -D STORE_TO_FILE main.cpp file_utils.cpp tree_printer.cpp utils.cpp kdtree.cpp kdtree_omp.cpp -o tree_serial.x
parallel:
	g++ -std=c++17 -Wall -Wextra -O3 -march=native  -D USE_OMP -fopenmp -D STORE_TO_FILE main.cpp file_utils.cpp tree_printer.cpp utils.cpp kdtree.cpp kdtree_omp.cpp -o tree_omp.x
clean:
	rm *.x
	rm *.gch
