NVCC = nvcc
CUDA_FLAGS = -g -G -arch=sm_70 -Xcompiler -fopenmp
TARGET = matriz
SOURCES = matriz.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CUDA_FLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean