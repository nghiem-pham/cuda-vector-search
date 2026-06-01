# Compilers
CC     = g++
CUDACC = nvcc

# Flags
FLAGS     = -O3 -Wall -std=c++17
CUDAFLAGS = -O3 -std=c++17 -Wno-deprecated-gpu-targets
INCLUDES  = -I src/

# Python binding flags 
PYBIND_INC = $(shell python3 -m pybind11 --includes)
PYTHON_INC = -I$(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_EXT = $(shell python3-config --extension-suffix)

# Targets
OBJS = engine

# Rules. By default show help
help:
	@echo
	@echo "CUDA Vector Search Engine"
	@echo
	@echo "make engine        Build CUDA search engine"
	@echo "make bindings      Build Python bindings"
	@echo "make all		   	  Build both engine and bindings"
	@echo "make clean         Remove build artifacts"

all: 
	$(OBJS) bindings

# Build main CUDA executable
engine: src/main.cpp src/engine.cu
	$(CUDACC) $(CUDAFLAGS) $(INCLUDES) $^ -o $@

bindings:
	$(CUDACC) -O2 -shared --compiler-options -fPIC $(PYBIND_INC) $(PYTHON_INC) $(INCLUDES) -o python/cuda_search$(PYTHON_EXT) python/bindings.cpp src/engine.cu $(CUDAFLAGS)

# Remove build artifacts
clean:
	rm -rf $(OBJS) python/cuda_search*.so