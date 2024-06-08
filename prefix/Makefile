EXECUTABLE := final

CU_FILES := exclusive_scan.cu

CC_FILES := main.cpp

########################################################

CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64

OBJS= main.o exclusive_scan_1.o exclusive_scan_2.o exclusive_scan_3.o check_array.o

default: $(EXECUTABLE)

clean:
	rm -rf *.o $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

main.o: main.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

# exclusive_scan.o: exclusive_scan_1.o exclusive_scan_2.o
# 	$(NVCC) $^ -o $@

exclusive_scan_1.o: exclusive_scan_1.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

exclusive_scan_2.o: exclusive_scan_2.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

exclusive_scan_3.o: exclusive_scan_3.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

check_array.o: check_array.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@