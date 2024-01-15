#include <string>
#include <gravomg/multigrid.h>
#include <gravomg/sampling.h>

#include <polyscope/polyscope.h>

#include <igl/readOBJ.h>
#include <igl/random_points_on_mesh.h>

#include <fmt/core.h>

#include "polyscope/point_cloud.h"

static constexpr int NUM_POINTS = 10'000;
static constexpr int NUM_COARSE_POINTS = 1'000;

int main() {

    // Load the cube
    Eigen::MatrixXd mesh_vertices;
    Eigen::MatrixXi mesh_faces;
    igl::readOBJ("../../test/cube.obj", mesh_vertices, mesh_faces);
    fmt::print("Loaded OBJ file: {}v, {}f\n", mesh_vertices.rows(), mesh_faces.rows());

    // Sample points on the cube
    Eigen::VectorXi _associated_faces;
    Eigen::MatrixXd _barycentric_coordinates;
    Eigen::MatrixXd point_cloud;
    igl::random_points_on_mesh(
        NUM_POINTS, mesh_vertices, mesh_faces,
        _barycentric_coordinates, _associated_faces, point_cloud
    );
    fmt::print("Sampled point cloud: {}x{}\n", point_cloud.rows(), point_cloud.cols());

    // Select coarse point hints
    // todo

    // Show results
    polyscope::init();
    polyscope::registerPointCloud("fine-points", point_cloud);
    polyscope::show();

    return 0;
}
