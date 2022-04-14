CUCC ?= nvcc
INCPATHS=
LIBPATHS=
INCPATHS += -I"/opt/cuda-10.0/include" -I"/home/zhaopeng/cuDNN7.6/include" -I"/home/zhaopeng/TensorRT-7.0.0.11/include"
LIBPATHS += -L"/opt/cuda-10.0/lib64" -L"/home/zhaopeng/TensorRT-7.0.0.11/lib" -L"/home/zhaopeng/cuDNN7.6/lib64"
LIBS = -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lcudnn -lcublas -lcudart

CUFLAGS ?= -std=c++11 -w $(INCPATHS) 

BIN_NAME = main 

OBJS =$(patsubst %.cpp, %.o, $(wildcard *.cpp ))
CUOBJS =$(patsubst %.cu, %.o, $(wildcard *.cu))
#test:
#	@echo ${OBJS} ${CUOBJS}

.PHONY = all clean

all: ${BIN_NAME}

SRCS := $(wildcard %.cu)
SRCS += $(wildcard %.cpp)



${BIN_NAME}: ${OBJS} ${CUOBJS} 
	${CUCC} -o ${BIN_NAME} ${OBJS} ${CUOBJS} $(LIBPATHS) $(LIBS) 

%.o: %.cu
	${CUCC} ${CUFLAGS} -c $< 

%.o: %.cpp
	${CUCC} ${CUFLAGS} -c $< 

clean:
	rm *.o ${BIN_NAME}
