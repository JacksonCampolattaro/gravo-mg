project(gravo-mg-tests)

include(FetchContent)

# Dependencies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if (NOT TARGET gravo-mg)
    find_package(gravo-mg REQUIRED)
endif ()

# Polyscope
FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG v2.1.0)
FetchContent_MakeAvailable(polyscope)

# IGL
FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG v2.5.0)
FetchContent_MakeAvailable(libigl)

# fmt
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

# Nonmanifold-Laplacians
FetchContent_Declare(
        nonmanifold-laplacian
        GIT_REPOSITORY https://github.com/JacksonCampolattaro/nonmanifold-laplacian-cpp.git
        GIT_TAG master
)
FetchContent_MakeAvailable(nonmanifold-laplacian)

# Geometry-central
FetchContent_Declare(
        geometry-central
        GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
        GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(geometry-central)


# Executable
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

add_executable(test
        main.cpp
)

set_property(TARGET test PROPERTY CXX_STANDARD 23)
target_link_libraries(
        test PRIVATE
        gravo-mg::gravo-mg
        fmt::fmt
        igl::core
        polyscope
        geometry-central
        nonmanifold-laplacian
)
