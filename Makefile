# Compilers
CC     = g++
CUDACC = nvcc

# Flags
FLAGS     = -O3 -Wall -std=c++17
CUDAFLAGS = -O3 -std=c++17 -Wno-deprecated-gpu-targets
INCLUDES  = -I src/

# Targets
OBJS = engine

# Rules. By default show help
help:
	@echo
	@echo "CUDA Vector Search Engine"
	@echo
	@echo "make engine        Build CUDA search engine"
	@echo "make clean         Remove build artifacts"

# Build main CUDA executable
engine: src/main.cpp src/engine.cu
	$(CUDACC) $(CUDAFLAGS) $(INCLUDES) $^ -o $@

# Remove build artifacts
clean:
	rm -rf $(OBJS)