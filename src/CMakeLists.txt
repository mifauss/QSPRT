add_executable(qsprt main.c qsprt.c)

target_compile_options(qsprt PUBLIC -DNDBUG -fopenmp -Wall -pedantic -O3 -march=native)

find_library(MATH_LIB m)
find_library(GSL_LIB gsl)
find_library(BLAS_LIB openblas)

target_link_libraries(qsprt PUBLIC ${MATH_LIB} ${GSL_LIB} ${BLAS_LIB})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qsprt PUBLIC OpenMP::OpenMP_CXX)
endif()
