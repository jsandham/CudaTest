CUSPARSE_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/2020/math_libs
CUSPARSE_INCLUDE=$(CUSPARSE_DIR)/include
CUSPARSE_LIB_PATH=$(CUSPARSE_DIR)/lib64
CUSPARSE_LIB=cusparse

CUDA_INCLUDE=/opt/nvidia/hpc_sdk/Linux_x86_64/2020/cuda/include
CUDA_LIB_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2020/cuda/lib64
CUDA_LIB=cudart

LDFLAGS=-L$(CUDA_LIB_PATH) -l$(CUDA_LIB) -L$(CUSPARSE_LIB_PATH) -l$(CUSPARSE_LIB)# -Wl, -R$(CUSPARSE_LIB_PATH)
LD=/opt/nvidia/hpc_sdk/Linux_x86_64/2020/compilers/bin/nvcc
CFLAGS=-I$(CUSPARSE_INCLUDE) -I$(CUDA_INCLUDE) -O3
CPP=/opt/nvidia/hpc_sdk/Linux_x86_64/2020/compilers/bin/nvcc
OBJ=main.o
EXE=main
%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $@ 

clean:
	rm -f $(EXE) $(OBJ) 
