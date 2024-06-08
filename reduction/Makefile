EXECUTABLE := final

CU_FILES := ${wildcard *.cu}

CC_FILES := main.cpp

########################################################

CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64

CU_OBJS= ${CU_FILES:.cu=.o}
CC_OBJS = ${CC_FILES:.cpp=.o}
OBJS = ${CC_OBJS} ${CU_OBJS}

default: $(EXECUTABLE)

clean:
	rm -rf *.o $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

${CC_OBJS}: ${CC_FILES}
	$(CXX) $< $(CXXFLAGS) -c -o $@

# exclusive_scan.o: exclusive_scan_1.o exclusive_scan_2.o
# 	$(NVCC) $^ -o $@

%.o: %.cu
	$(NVCC) $^ $(NVCCFLAGS) -c -o $@
