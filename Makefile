# part of the definitions are from CUDAatScale coursera course
# Define the compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include -Iinclude -I../Common -I../Common/UtilNPP 
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
LIB_DIR = lib

# Define source files and target executable
SRC = $(SRC_DIR)/imageFlipNPP.cpp
TARGET = $(BIN_DIR)/imageFlipNPP

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Define the default rule
all: $(TARGET)

# Rule for running the application
run: $(TARGET)
	./$(TARGET) --input $(DATA_DIR)/grey-sloth.png --output $(DATA_DIR)/grey-sloth_flipped.png

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

# Installation rule (not much to install, but here for completeness)
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Run the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make help   - Display this help message."
