cmake_minimum_required(VERSION 3.16)
project(gravomg)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GRAVOMG_BUILD_TEST "Build gravomg test program" OFF)

include(FetchContent)

# Eigen
if (TARGET Eigen3::Eigen)
    if (${GC_EIGEN_LOCATION})
        # Change this to a FATAL_ERROR if the constraint should be observed
        message(WARNING "Target Eigen3::Eigen already exists. ")
    endif ()
else ()

    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
    )

    FetchContent_GetProperties(eigen)
    if (NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif ()

    add_library(eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS eigen)
    target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
endif ()

# Add your project files
SET(SRC_FILES
        src/multigrid.cpp
        src/utility.cpp
        src/sampling.cpp
)

SET(HEADER_FILES
        include/gravomg/multigrid.h
        include/gravomg/utility.h
        include/gravomg/sampling.h
)

add_library(gravomg ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(gravomg PUBLIC Eigen3::Eigen)
target_include_directories(gravomg PUBLIC include)

add_library(gravomg::gravomg ALIAS gravomg)

# Conditionally add the -fPIC flag for shared library on supported platforms
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
    target_compile_options(gravomg PRIVATE $<$<CXX_COMPILER_ID:GNU>:-fPIC>)
endif ()

if (GRAVOMG_BUILD_TEST)
    add_subdirectory(test)
endif ()

