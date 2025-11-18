cmake_minimum_required(VERSION 3.16)
project(LogisticRegressionTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT_DIR "$ENV{ROOTSYS}")
find_package(ROOT REQUIRED CONFIG)

include_directories(
    ${ROOT_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    /usr/include/eigen3
)

add_executable(log_reg src/main.cpp)
target_link_libraries(log_reg ${ROOT_LIBRARIES})
