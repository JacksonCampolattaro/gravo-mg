#include <string>
#include <gravomg/multigrid.h>
#include <gravomg/sampling.h>

#include <polyscope/polyscope.h>

#include <igl/readOBJ.h>
#include <igl/random_points_on_mesh.h>
#include <igl/knn.h>

#include <fmt/core.h>

#include "igl/octree.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

static constexpr int NUM_POINTS = 10'000;
static constexpr int NUM_COARSE_POINTS = 1'000;
static constexpr int K = 32;

int main() {

    // Load the cube
    Eigen::MatrixXd mesh_vertices;
    Eigen::MatrixXi mesh_faces;
    igl::readOBJ("../../test/cube.obj", mesh_vertices, mesh_faces);
    fmt::print("Loaded OBJ file: {}v, {}f\n", mesh_vertices.rows(), mesh_faces.rows());

    // Sample points on the cube
    Eigen::MatrixXd point_cloud; {
        Eigen::VectorXi _associated_faces;
        Eigen::MatrixXd _barycentric_coordinates;
        igl::random_points_on_mesh(
            NUM_POINTS, mesh_vertices, mesh_faces,
            _barycentric_coordinates, _associated_faces, point_cloud
        );
    }
    fmt::print("Sampled point cloud: {}x{}\n", point_cloud.rows(), point_cloud.cols());

    // Find KNN for the point cloud
    GravoMG::EdgeMatrix knn_edges; {
        std::vector<std::vector<int>> O_PI;
        Eigen::MatrixXi O_CH;
        Eigen::MatrixXd O_CN;
        Eigen::VectorXd O_W;
        igl::octree(point_cloud, O_PI, O_CH, O_CN, O_W);
        igl::knn(point_cloud, K, O_PI, O_CH, O_CN, O_W, knn_edges);
    }
    fmt::print("Produced KNN table: {}x{}\n", knn_edges.rows(), knn_edges.cols());

    // Select coarse point hints
    const auto coarse_point_hints = GravoMG::fastDiscSample(point_cloud, knn_edges, 0.1);
    fmt::print("Selected points using fast disc sampling: {}\n", coarse_point_hints.size());
    const auto coarse_points = point_cloud(coarse_point_hints, Eigen::all);

    // Produce prolongation operator
    const auto U = GravoMG::constructProlongation(
        point_cloud, knn_edges,
        coarse_point_hints,
        BARYCENTRIC
    );
    fmt::print("Produced a prolongation operator: {}x{}\n", U.rows(), U.cols());


    // Get relationships from the sparse matrix
    std::vector<Eigen::Triplet<double>> fine_coarse_relationships;
    std::vector<std::array<Eigen::Index, 2>> edges;
    std::vector<double> weights;
    for (int i = 0; i < U.outerSize(); i++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(U, i); it; ++it) {
            fine_coarse_relationships.emplace_back(it.row(), it.col(), it.value());
            edges.push_back({it.row(), coarse_point_hints[it.col()]});
            weights.push_back(it.value());
        }
    }
    fmt::print("Extracted fine-coarse relationships: {} elements\n", fine_coarse_relationships.size());

    // Show results
    polyscope::init();
    polyscope::registerPointCloud("fine-points", point_cloud)
            ->setPointRadius(0.0025);
    polyscope::registerPointCloud("coarse-points", coarse_points)
            ->setPointRadius(0.005);
    polyscope::registerCurveNetwork("fine-coarse", point_cloud, edges)
            ->setRadius(0.0025)
            ->addEdgeScalarQuantity("weights", weights)->setEnabled(true);
    polyscope::show();

    return 0;

}
