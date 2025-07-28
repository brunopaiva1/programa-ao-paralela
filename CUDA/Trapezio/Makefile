NVCC = nvcc
CUDA_FLAGS = -g -G
TARGET = trap
SOURCES = trap.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CUDA_FLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean