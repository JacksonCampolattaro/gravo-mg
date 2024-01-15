project(gravo-mg-tests)

if (NOT TARGET gravo-mg)
    find_package(gravo-mg REQUIRED)
endif ()

include(FetchContent)

FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG v2.1.0)
FetchContent_MakeAvailable(polyscope)

FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG v2.5.0)
FetchContent_MakeAvailable(libigl)

FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

add_executable(test
        main.cpp
)

set_property(TARGET test PROPERTY CXX_STANDARD 23)
target_link_libraries(
        test PRIVATE
        gravo-mg::gravo-mg
        fmt::fmt
        polyscope
        igl::core
)
