#include <string>
#include <ranges>

#include <polyscope/polyscope.h>

#include <igl/readOBJ.h>
#include <igl/random_points_on_mesh.h>
#include <igl/knn.h>

#include <fmt/core.h>

#include "igl/octree.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/simple_triangle_mesh.h"

#include <gravomg/multigrid.h>
#include <gravomg/sampling.h>

static constexpr int NUM_POINTS = 8000;
static constexpr int NUM_COARSE_POINTS = 100; // todo: use binary search for disc sampling radius
static constexpr int K = 32;

using Eigen::Index;

std::vector<std::array<Eigen::Index, 2>> toEdgePairs(
    const std::span<Eigen::Index> indices,
    const std::function<Index(Index)>& transformation = std::identity{}
) {
    std::vector<std::array<Eigen::Index, 2>> pairs;
    for (Eigen::Index i = 0; i < indices.size(); ++i)
        pairs.push_back({i, transformation(indices[i])});
    return pairs;
}

std::vector<std::array<Eigen::Index, 2>> toEdgePairs(
    const GravoMG::EdgeMatrix& edges,
    const std::function<Index(Index)>& transformation = std::identity{}
) {
    std::vector<std::array<Eigen::Index, 2>> pairs;
    for (Eigen::Index i = 0; i < edges.rows(); ++i) {
        for (const Eigen::Index j: edges.row(i)) {
            if (j >= 0) pairs.push_back({transformation(i), transformation(j)});
        }
    }
    return pairs;
}

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
    GravoMG::EdgeMatrix fine_edges; {
        std::vector<std::vector<int>> O_PI;
        Eigen::MatrixXi O_CH;
        Eigen::MatrixXd O_CN;
        Eigen::VectorXd O_W;
        igl::octree(fine_points, O_PI, O_CH, O_CN, O_W);
        igl::knn(fine_points, K, O_PI, O_CH, O_CN, O_W, fine_edges);
    }
    fmt::print("Produced KNN table: {}x{}\n", fine_edges.rows(), fine_edges.cols());

    // Select coarse point hints
    const auto coarse_point_recommendations = GravoMG::fastDiscSample(fine_points, fine_edges, 0.1);
    fmt::print("Selected coarse points using fast disc sampling: {}\n", coarse_point_recommendations.size());
    //const auto coarse_points = point_cloud(coarse_point_hints, Eigen::all);

    // Associate all fine points with their coarse parent
    std::vector<Eigen::Index> fine_to_nearest_coarse(fine_points.rows());
    Eigen::VectorXd fine_to_nearest_coarse_distance(fine_points.rows());
    fine_to_nearest_coarse_distance.setConstant(std::numeric_limits<double>::max());
    GravoMG::constructDijkstraWithCluster(
        fine_points, coarse_point_recommendations,
        fine_edges, fine_to_nearest_coarse_distance, fine_to_nearest_coarse
    );
    const auto fine_coarse_edge_pairs = toEdgePairs(
        fine_to_nearest_coarse,
        [&](auto coarse_index) { return coarse_point_recommendations[coarse_index]; }
    );

    // Produce a coarse edge graph from the fine one
    auto coarse_edges = GravoMG::toPaddedEdgeMatrix(GravoMG::extractCoarseEdges(
        fine_points,
        fine_edges,
        coarse_point_recommendations,
        fine_to_nearest_coarse
    ));
    const auto coarse_edge_pairs = toEdgePairs(coarse_edges);

    // Improve the locations of the coarse points
    auto coarse_points = GravoMG::coarseFromMeanOfFineChildren(
        fine_points,
        fine_edges,
        fine_to_nearest_coarse,
        coarse_point_recommendations.size()
    );
    // todo

    // Produce voronoi triangles for the coarse points
    // todo


#if(false)
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
#endif

    // Show results
    polyscope::init();
    polyscope::registerPointCloud("fine-points", fine_points)
            ->setPointRadius(0.0015);
    polyscope::registerPointCloud("coarse-points", coarse_points)
            ->setPointRadius(0.005);
    polyscope::registerCurveNetwork("fine-coarse", fine_points, fine_coarse_edge_pairs);
    polyscope::registerCurveNetwork("coarse-coarse", coarse_points, coarse_edge_pairs);
    // ->setRadius(0.0025)
    // ->addEdgeScalarQuantity("weights", weights)->setEnabled(true);
    // todo: visualize triangles
    // polyscope::registerSimpleTriangleMesh("triangles", coarse_points, triangles);
    polyscope::show();

    return 0;

}
