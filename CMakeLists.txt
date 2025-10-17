cmake_minimum_required(VERSION 3.20)
project(first_example)

include(FetchContent)


# Use FetchContent
# # Download and configure xtl
FetchContent_Declare(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
    GIT_TAG master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(xtl xsimd xtensor-blas xtensor)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(first_example main.cpp)

target_link_libraries(first_example PUBLIC xtensor xtensor-blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
