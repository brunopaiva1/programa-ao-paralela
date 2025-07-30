NVCC = nvcc
CUDA_FLAGS = -g -G -arch=sm_70
TARGET = bitonic_sort
SOURCES = bitonic_sort.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CUDA_FLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean