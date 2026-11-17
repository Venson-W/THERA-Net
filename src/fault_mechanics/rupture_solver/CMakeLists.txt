cmake_minimum_required(VERSION 3.10)
project(thera_rupture LANGUAGES CXX)

# Configure with -Duguca_DIR=<the existing UGUCA build directory>.
# UGUCA remains an unmodified, separately built LGPL dependency.
find_package(uguca CONFIG REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(rupture_driver rupture_driver.cc)
target_compile_features(rupture_driver PRIVATE cxx_std_17)
target_compile_options(rupture_driver PRIVATE -Wall -Wextra -pedantic)
target_include_directories(rupture_driver PRIVATE ${UGUCA_INCLUDE_DIRS})
target_link_libraries(rupture_driver PRIVATE uguca MPI::MPI_CXX)

add_executable(barrier_driver barrier_driver.cc)
target_compile_features(barrier_driver PRIVATE cxx_std_17)
target_compile_options(barrier_driver PRIVATE -Wall -Wextra -pedantic)
target_include_directories(barrier_driver PRIVATE ${UGUCA_INCLUDE_DIRS})
target_link_libraries(barrier_driver PRIVATE uguca MPI::MPI_CXX)
