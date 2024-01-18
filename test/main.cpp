#include <string>
#include <ranges>

#include <polyscope/polyscope.h>

#include <igl/readOBJ.h>
#include <igl/random_points_on_mesh.h>
#include <igl/knn.h>

#include <fmt/core.h>

#include <igl/octree.h>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>
#include <polyscope/simple_triangle_mesh.h>

#include <nonmanifoldlaplacian/robust_laplacian.h>

#include <gravomg/multigrid.h>
#include <gravomg/sampling.h>

static constexpr int NUM_POINTS = 5000;
static constexpr double REDUCTION_RATIO = 5; // todo: this doesn't seem reasonable
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

std::vector<std::array<Index, 2>> toEdgePairs(
    const GravoMG::NeighborList& edges,
    const std::function<Index(Index)>& transformation = std::identity{}
) {
    std::vector<std::array<Index, 2>> pairs;
    for (Index i = 0; i < edges.size(); ++i) {
        for (const Index j: edges[i]) {
            pairs.push_back({transformation(i), transformation(j)});
        }
    }
    return pairs;
}

std::vector<std::array<Index, 2>> toEdgePairs(
    const GravoMG::NeighborMatrix& edges,
    const std::function<Index(Index)>& transformation = std::identity{}
) {
    std::vector<std::array<Index, 2>> pairs;
    for (Index i = 0; i < edges.rows(); ++i) {
        for (const Index j: edges.row(i)) {
            if (j >= 0) pairs.push_back({transformation(i), transformation(j)});
        }
    }
    return pairs;
}

std::set<GravoMG::Triangle> toFaces(const std::vector<GravoMG::TriangleWithNormal>& triangles_with_normals) {
    std::set<GravoMG::Triangle> triangles;
    for (auto& [triangle, normal]: triangles_with_normals) {
        auto [a, b, c] = triangle;
        // todo: this is silly
        if (!triangles.contains({a, b, c}) &&
            !triangles.contains({b, c, a}) &&
            !triangles.contains({c, a, b})) {
            // Insert a version of the triangle with both normals
            triangles.insert({a, b, c});
            triangles.insert({a, c, b});
        }


    }
    return triangles;
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
    const auto [stiffness, M] = buildPointCloudLaplacian(fine_points, 1e-5, K);
    const auto fine_edges = GravoMG::extractEdges(stiffness);
    const auto fine_edges_homogenous = GravoMG::toHomogenous(fine_edges);
    fmt::print("Produced neighbor table: {}x{}\n", fine_edges_homogenous.rows(), fine_edges_homogenous.cols());

    // Select coarse point hints
    const auto radius = std::cbrt(REDUCTION_RATIO) * GravoMG::averageEdgeLength(fine_points, fine_edges);
    fmt::print("Selected radius for fast disc sampling: {}\n", radius);
    const auto coarse_point_recommendations = GravoMG::fastDiscSample(fine_points, fine_edges_homogenous, radius);
    fmt::print("Selected coarse points using fast disc sampling: {}\n", coarse_point_recommendations.size());
    //const auto coarse_points = point_cloud(coarse_point_hints, Eigen::all);

    // Associate all fine points with their coarse parent
    // std::vector<Eigen::Index> fine_to_nearest_coarse(fine_points.rows());
    // Eigen::VectorXd fine_to_nearest_coarse_distance(fine_points.rows());
    // fine_to_nearest_coarse_distance.setConstant(std::numeric_limits<double>::max());
    // GravoMG::constructDijkstraWithCluster(
    //     fine_points, coarse_point_recommendations,
    //     fine_edges, fine_to_nearest_coarse_distance, fine_to_nearest_coarse
    // );
    auto fine_to_nearest_coarse = GravoMG::assignParents(
        fine_points,
        fine_edges,
        coarse_point_recommendations
    );
    const auto fine_coarse_edge_pairs = toEdgePairs(
        fine_to_nearest_coarse,
        [&](auto coarse_index) { return coarse_point_recommendations[coarse_index]; }
    );
    fmt::print("Associated each fine point with a coarse \"parent\"\n");

    // Produce a coarse edge graph from the fine one
    auto coarse_edges = GravoMG::extractCoarseEdges(
        fine_points,
        fine_edges,
        coarse_point_recommendations,
        fine_to_nearest_coarse
    );
    const auto coarse_edge_pairs = toEdgePairs(coarse_edges);

    // Improve the locations of the coarse points
    auto coarse_points = GravoMG::coarseFromMeanOfFineChildren(
        fine_points,
        fine_edges,
        fine_to_nearest_coarse,
        coarse_point_recommendations.size()
    );
    fmt::print("Moved each coarse point to the mean of its \"children\"\n");

    // Produce voronoi triangles for the coarse points
    auto [triangles_with_normals, point_triangle_associations] = GravoMG::constructVoronoiTriangles(
        coarse_points,
        coarse_edges
    );
    fmt::print("Constructed voronoi triangles from the coarse points\n");

    // Produce a prolongation operator
    auto U = GravoMG::constructProlongation(
        fine_points,
        coarse_points,
        coarse_edges,
        fine_to_nearest_coarse,
        BARYCENTRIC
    );
    fmt::print("Produced a prolongation operator: {}x{}\n", U.rows(), U.cols());

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
    std::vector<std::array<Index, 2>> edges;
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
    polyscope::registerCurveNetwork("coarse-coarse", coarse_points, coarse_edge_pairs);
    polyscope::registerCurveNetwork("fine-coarse", multilevel_point_cloud, edges)
            ->setRadius(0.0025)
            ->addEdgeScalarQuantity("weights", weights)->setEnabled(true);
    // todo: visualize triangles
    polyscope::registerSimpleTriangleMesh("triangles", coarse_points, toFaces(triangles_with_normals));
    polyscope::show();

    return 0;

}
