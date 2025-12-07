cmake_minimum_required(VERSION 3.20)
project(mlpp
    VERSION 0.3.0
    DESCRIPTION "Modern Machine Learning in C++"
    LANGUAGES CXX
)

option(MLPP_BUILD_TESTS      "Build tests"                     ON)
option(MLPP_BUILD_EXAMPLES   "Build examples"                  ON)
option(MLPP_WITH_PYPLOT      "Enable matplotlibcpp (Python+NumPy)" ON)
option(MLPP_WITH_OPENCV      "Enable OpenCV support"           ON)
option(MLPP_USE_EIGEN        "Use Eigen when available"        ON)

add_library(mlpp INTERFACE)
add_library(mlpp::mlpp ALIAS mlpp)

target_include_directories(mlpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(mlpp INTERFACE cxx_std_20)

if(MLPP_USE_EIGEN)
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(Eigen3_FOUND)
        target_link_libraries(mlpp INTERFACE Eigen3::Eigen)
        target_compile_definitions(mlpp INTERFACE MLPP_HAS_EIGEN)
    endif()
endif()

if(MLPP_WITH_PYPLOT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
    target_link_libraries(mlpp INTERFACE Python3::Python Python3::NumPy)
    target_compile_definitions(mlpp INTERFACE MLPP_WITH_PYPLOT)
endif()

if(MLPP_WITH_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc highgui)
    if(OpenCV_FOUND)
        target_link_libraries(mlpp INTERFACE opencv_core opencv_imgproc opencv_highgui)
        target_compile_definitions(mlpp INTERFACE MLPP_HAS_OPENCV)
    endif()
endif()

include(GNUInstallDirs)

install(DIRECTORY include/mlpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS mlpp EXPORT mlppTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT mlppTargets
    FILE mlppTargets.cmake
    NAMESPACE mlpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlpp
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mlppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mlppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mlppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlpp
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mlppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mlppConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlpp
)

if(MLPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MLPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()