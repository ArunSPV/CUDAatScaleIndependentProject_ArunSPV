# Image Flip NPP

A Image Flip NPP Sample code demonstrates how to use NPP Mirror function to perform a png image flip along vertical axis. Input png image file can be grayscale or coloured. The code uses CUDA NPP libraries to run the code in available GPU.    

## Key concepts:

Usage of FreeImage library, NPP image processing library

## Minimum spec: 
SM 2.0

## Code organisation

### Makefile
   
Makefile contains rules for building and running the png image flip code.

If the repository is cloned and run inside boxFilterNPP VS code workspace of 
CUDAatScale coursera course, nothing needs to be changed in makefile

If the code is built and run in some other machine, 
dependencies for running the makefile present between lines 3 to 7 
needs to be changed accordingly

### src

src folder contains, imageFlipNPP.cpp file where the code for the opening and flipping png
algorithm is present

### bin

bin folder is where binary executable file imageFlipNPP created by Makefile from src/imageFlipNPP.cpp 
file is present. There is a pre built imageFlipNPP binary file present in the folder.

### data

data folder contains both the input png files to executable  bin/imageFlipNPP and output _flipped.png 
file is present. There are already examples of input and output png files present in the folder

### dependencies

Common folder with cuda helper header files and libraries, a copy from CudaAtScale course's lab project.

CUDA library need to be present in below location of the machine where the code is built and run. 

    /usr/local/cuda

If cuda is installed in a different location replace the above location with the different location.

FreeImage library needs to be installed locally in your machine if the code is run outside CUDAatScale 
boxFilterNPP lab vscode workspace 

## How to clone the repository

clone the git repository via terminal

    git clone https://github.com/ArunSPV/CUDAatScaleIndependentProject_ArunSPV.git

there should be folder named CUDAatScaleIndependentProject_ArunSPV

## How to build the code

get inside the cloned CUDAatScaleIndependentProject_ArunSPV directory

    cd CUDAatScaleIndependentProject_ArunSPV
    
run the command

    make

## How to run the code

get inside the cloned CUDAatScaleIndependentProject_ArunSPV directory

    cd CUDAatScaleIndependentProject_ArunSPV

place png file to flip in data folder    

There are 2 ways to run the image flip algorithm on a png image file

### Via executing binary

while inside CUDAatScaleIndependentProject_ArunSPV run the following command in terminal

    ./bin/imageFlipNPP --input data/image-name.png --output data/image-name_flipped.png

replace 'image-name' with the actual png image name that needs to be flipped 

### Via Makefile arguments

in line 28 of the Makefile replace 'grey-sloth' in 2 occasions to the desired png image name
and run the following command in terminal while being inside CUDAatScaleIndependentProject_ArunSPV
directory

    make run
    




