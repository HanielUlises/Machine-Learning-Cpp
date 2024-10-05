cmake_minimum_required(VERSION 3.10)

project(MachineLearningCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(
    include/Classification
    include/Decision_Trees
    include/Dimensionality_Reduction
    include/Model_Validation
    include/Supervised_Learning/Clustering
    include/Supervised_Learning/KNN
    include/Supervised_Learning/Regression
)

file(GLOB_RECURSE SRC_FILES
    src/Classification/*.cpp
    src/Decision_Trees/*.cpp
    src/Model_Validation/*.cpp
    src/Supervised_Learning/Clustering/*.cpp
    src/Supervised_Learning/KNN/*.cpp
    src/Supervised_Learning/Regression/*.cpp
)

add_executable(MachineLearningCpp ${SRC_FILES})