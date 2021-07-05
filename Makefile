TARGET=matrix_multiply
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:%.cu=%.o)

$(CU_OBJS): $(CU_SRCS)
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:%.cu=%.o)

$(CU_OBJS): $(CU_SRCS)
	nvcc -c $^ -arch=sm_60

all: $(TARGET)
$(TARGET): $(CU_OBJS)
	nvcc -o  $@ $^ -arch=sm_60

