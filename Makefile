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
	@echo "make engine        Build CUDA search engine"\
	@echo "make bindings      Build Python bindings"\
	@echo "make all		   	  Build both engine and bindings"\
	@echo "make clean         Remove build artifacts"

all: 
	$(OBJS) bindings

# Build main CUDA executable
engine: src/main.cpp src/engine.cu
	$(CUDACC) $(CUDAFLAGS) $(INCLUDES) $^ -o $@

bindings:
	$(CUDACC) -O2 -shared \ 
		--compiler-options -fPIC \ 
		$(shell python3 -m pybind11 --includes) \ -I$(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \ 
		$(INCLUDES) \ 
		-o cuda_search$(shell python3-config --extension-suffix) \ 
		python/bindings.cpp src/engine.cu \ 
		$(CUDAFLAGS)

# Remove build artifacts
clean:
	rm -rf $(OBJS) cuda_search*.so