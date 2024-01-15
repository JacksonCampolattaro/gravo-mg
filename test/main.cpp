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
#include "polyscope/simple_triangle_mesh.h"

static constexpr int NUM_POINTS = 1000;
static constexpr int NUM_COARSE_POINTS = 100;
static constexpr int K = 32;

int main() {

    // Load the cube
    Eigen::MatrixXd mesh_vertices;
    Eigen::MatrixXi mesh_faces;
    igl::readOBJ("../../test/cube.obj", mesh_vertices, mesh_faces);
    fmt::print("Loaded OBJ file: {}v, {}f\n", mesh_vertices.rows(), mesh_faces.rows());

    // Sample points on the cube
    Eigen::MatrixXd fine_points; {
        Eigen::VectorXi _associated_faces;
        Eigen::MatrixXd _barycentric_coordinates;
        igl::random_points_on_mesh(
            NUM_POINTS, mesh_vertices, mesh_faces,
            _barycentric_coordinates, _associated_faces, fine_points
        );
    }
    fmt::print("Sampled point cloud: {}x{}\n", fine_points.rows(), fine_points.cols());

    // Find KNN for the point cloud
    GravoMG::EdgeMatrix knn_edges; {
        std::vector<std::vector<int>> O_PI;
        Eigen::MatrixXi O_CH;
        Eigen::MatrixXd O_CN;
        Eigen::VectorXd O_W;
        igl::octree(fine_points, O_PI, O_CH, O_CN, O_W);
        igl::knn(fine_points, K, O_PI, O_CH, O_CN, O_W, knn_edges);
    }
    fmt::print("Produced KNN table: {}x{}\n", knn_edges.rows(), knn_edges.cols());

    // Select coarse point hints
    const auto coarse_point_recommendations = GravoMG::fastDiscSample(fine_points, knn_edges, 0.1);
    fmt::print("Selected coarse points using fast disc sampling: {}\n", coarse_point_recommendations.size());
    //const auto coarse_points = point_cloud(coarse_point_hints, Eigen::all);

    // Produce prolongation operator
    const auto [coarse_points, coarse_edges, U] = GravoMG::constructProlongation(
        fine_points, knn_edges,
        coarse_point_recommendations,
        BARYCENTRIC,
        true, false
    );
    fmt::print("Produced a prolongation operator: {}x{}\n", U.rows(), U.cols());

    // Get coarse triangles from the sparse matrix
    std::vector<GravoMG::Triangle> triangles;
    for (int i = 0; i < U.outerSize(); i++) {
        // GravoMG::Triangle triangle;
        // auto vertex_it = triangle.begin();
        for (Eigen::SparseMatrix<double>::InnerIterator it(U, i); it; ++it) {
            fmt::print("({}, {})\n", it.row(), it.col());
            // *vertex_it = it.col();
            // vertex_it++;
        }
        //triangles.emplace_back(triangle);
    }
    fmt::print("Extracted triangles: {} elements\n", triangles.size());


    // Combine coarse and fine points into a single matrix
    auto total_point_count = fine_points.rows() + coarse_points.rows();
    GravoMG::PointMatrix multilevel_point_cloud{total_point_count, fine_points.cols()};
    multilevel_point_cloud.topRows(fine_points.rows()) = fine_points;
    multilevel_point_cloud.bottomRows(coarse_points.rows()) = coarse_points;
    std::vector<Eigen::Triplet<double>> offset_triplets;
    for (int i = 0; i < U.outerSize(); i++)
        for (Eigen::SparseMatrix<double>::InnerIterator it(U, i); it; ++it)
            offset_triplets.emplace_back(it.row(), it.col() + fine_points.rows(), it.value());
    Eigen::SparseMatrix<double> multilevel_U{total_point_count, total_point_count};
    multilevel_U.setFromTriplets(offset_triplets.begin(), offset_triplets.end());
    fmt::print("Combined coarse and fine points into a single cloud, with a combined prolongation operator\n");

    // Get relationships from the combined sparse matrix
    std::vector<std::array<Eigen::Index, 2>> edges;
    std::vector<double> weights;
    for (int i = 0; i < multilevel_U.outerSize(); i++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(multilevel_U, i); it; ++it) {
            edges.push_back({it.row(), it.col()});
            weights.push_back(it.value());
        }
    }
    fmt::print("Extracted fine-coarse relationships: {} elements\n", edges.size());

    // Show results
    polyscope::init();
    polyscope::registerPointCloud("fine-points", fine_points)
            ->setPointRadius(0.0015);
    polyscope::registerPointCloud("coarse-points", coarse_points)
            ->setPointRadius(0.005);
    polyscope::registerCurveNetwork("fine-coarse", multilevel_point_cloud, edges)
            ->setRadius(0.0025)
            ->addEdgeScalarQuantity("weights", weights)->setEnabled(true);
    // todo: visualize triangles
    polyscope::registerSimpleTriangleMesh("triangles", coarse_points, triangles);
    polyscope::show();

    return 0;

}
