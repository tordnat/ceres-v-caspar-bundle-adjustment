cmake_minimum_required(VERSION 3.26)

project(CasparTest VERSION 1.0
        DESCRIPTION "Testing replacing CERES with Caspar"
        LANGUAGES CXX C)

set(CMAKE_CXX_COMPILER c++)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CASPAR_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
set(CASPAR_BUILD_DIR "${CASPAR_LIB_DIR}/build")
add_library(caspar_lib_core STATIC IMPORTED GLOBAL)
set_target_properties(caspar_lib_core PROPERTIES
                      IMPORTED_LOCATION "${CASPAR_BUILD_DIR}/libcaspar_lib_core.a"
                      INTERFACE_INCLUDE_DIRECTORIES "${CASPAR_LIB_DIR}"
                      )
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
target_link_libraries(caspar_lib_core INTERFACE CUDA::cudart)



add_executable(caspar_test main.cc)
set_target_properties(caspar_test PROPERTIES CXX_STANDARD 17)
target_include_directories(caspar_test PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(caspar_test PRIVATE 
caspar_lib_core
    Ceres::ceres
    Eigen3::Eigen
    Sophus::Sophus
)