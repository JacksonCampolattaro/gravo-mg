#include "gravomg/multigrid.h"
#include "gravomg/utility.h"
#include "gravomg/sampling.h"

#include <Eigen/Dense>

#include <cmath>
#include <numeric>
#include <chrono>

#include <utility>
#include <set>

namespace GravoMG {

    using VertexWithDistance = std::pair<double, Index>;

    double inTriangle(const Point &p, std::span<Index, 3> tri,
                      const Normal &triNormal, const PointMatrix &pos,
                      Eigen::RowVector3d &bary, std::map<Index, float> &insideEdge) {
        Eigen::RowVector3d v1, v2, v3;
        v1 = pos.row(tri[0]);
        v2 = pos.row(tri[1]);
        v3 = pos.row(tri[2]);
        Eigen::RowVector3d v1ToP = p - v1;
        Eigen::RowVector3d e12 = v2 - v1;
        Eigen::RowVector3d e13 = v3 - v1;

        double distToTriangle = (p - v1).dot(triNormal);
        Eigen::RowVector3d pProjected = p - distToTriangle * triNormal;

        double doubleArea = (v2 - v1).cross(v3 - v1).dot(triNormal);
        bary(0) = (v3 - v2).cross(pProjected - v2).dot(triNormal) / doubleArea;
        bary(1) = (v1 - v3).cross(pProjected - v3).dot(triNormal) / doubleArea;
        bary(2) = 1. - bary(0) - bary(1);

        if (insideEdge.find(tri[1]) == insideEdge.end()) {
            insideEdge[tri[1]] = ((v1ToP) - ((v1ToP).dot(e12) * (e12))).norm();
        }
        if (insideEdge.find(tri[2]) == insideEdge.end()) {
            insideEdge[tri[2]] = ((v1ToP) - ((v1ToP).dot(e13) * (e13))).norm();
        }
        if (bary(0) < 0. || bary(1) < 0.) {
            insideEdge[tri[1]] = -1.;
        }
        if (bary(0) < 0. || bary(2) < 0.) {
            insideEdge[tri[2]] = -1.;
        }

        if (bary(0) >= 0. && bary(1) >= 0. && bary(2) >= 0.) {
            return abs(distToTriangle);
        }

        return -1.;
    }

    std::vector<double> uniformWeights(const int &n_points) {
        std::vector<double> weights(n_points);
        std::fill(weights.begin(), weights.end(), 1. / n_points);
        return weights;
    }

    std::vector<double> inverseDistanceWeights(const Eigen::MatrixXd &pos, const Eigen::RowVector3d &p,
                                               const std::span<Index> &edges) {
        double sumWeight = 0.;
        std::vector<double> weights(edges.size());
        for (size_t j = 0; j < edges.size(); ++j) {
            weights[j] = 1. / std::max(1e-8, (p - pos.row(edges[j])).norm());
            sumWeight += weights[j];
        }
        for (size_t j = 0; j < weights.size(); ++j) {
            weights[j] = weights[j] / sumWeight;
        }
        return weights;
    }

    std::vector<Index> assignParents(
            const Eigen::MatrixXd &fine_points,
            const EdgeMatrix &fine_edge_matrix,
            const std::vector<Index> &coarse_samples
    ) {
        std::vector<Index> parents(fine_points.rows());
        Eigen::VectorXd parent_distances(fine_points.rows());
        parent_distances.setConstant(std::numeric_limits<double>::max());

        std::priority_queue<VertexWithDistance, std::vector<VertexWithDistance>, std::greater<>> distance_queue;

        // Self-interactions have zero path length
        for (Index coarse = 0; coarse < coarse_samples.size(); ++coarse) {
            parents[coarse_samples[coarse]] = coarse;
            distance_queue.emplace(0.0, coarse_samples[coarse]);
            parent_distances[coarse_samples[coarse]] = 0.0;
        }

        // Treat connections in order of path length
        while (!distance_queue.empty()) {

            // Pop the connection with the shortest path length
            const auto [fine_path_length_to_coarse, fine] = distance_queue.top();
            distance_queue.pop();
            const auto fine_point = fine_points.row(fine);

            // Treat every neighbor of this point to search for a shorter path
            for (Eigen::SparseMatrix<double>::InnerIterator it(fine_edge_matrix, fine); it; ++it) {
                const auto neighbor = it.index();
                const auto neighbor_point = fine_points.row(neighbor);
                const auto neighbor_to_fine_distance = (fine_point - neighbor_point).norm();
                const auto neighbor_path_length_to_course = neighbor_to_fine_distance + fine_path_length_to_coarse;

                // If we find a shorter path for a neighbor...
                if (neighbor_path_length_to_course < parent_distances[neighbor]) {

                    // Update its nearest parent & associated distance
                    parents[neighbor] = parents[fine];
                    parent_distances[neighbor] = neighbor_path_length_to_course;

                    // Add it to the queue
                    distance_queue.emplace(neighbor_path_length_to_course, neighbor);
                }

            }
        }

        return parents;
    }

    double averageEdgeLength(const PointMatrix &positions, const EdgeList &neighbors) {
        const auto &rows = neighbors.rowwise();
        return std::transform_reduce(rows.begin(), rows.end(), 0.0, std::plus<>{}, [&](const auto &row) {
            auto [i, j] = std::make_tuple(row[0], row[1]);
            return (positions.row(j) - positions.row(i)).norm();
        }) / double(neighbors.rows() - positions.rows()); // Self connections are not included in the average
    }

    Eigen::SparseMatrix<double> extractCoarseEdges(
            const PointMatrix &fine_points,
            const EdgeMatrix &fine_edge_matrix,
            const std::vector<Index> &coarse_samples,
            const std::vector<Index> &fine_to_nearest_coarse
    ) {
        Eigen::SparseMatrix<double> coarse_edge_matrix{Index(coarse_samples.size()), Index(coarse_samples.size())};
        for (Index fine = 0; fine < fine_points.rows(); ++fine) {
            const auto parent = fine_to_nearest_coarse[fine];
            for (Eigen::SparseMatrix<double>::InnerIterator it(fine_edge_matrix, fine); it; ++it) {
                const auto neighbor_parent = fine_to_nearest_coarse[it.index()];

                // If the two points belong to different parent points, add a connection between the parents
                if (parent != neighbor_parent) {
                    // The distance through this point is given by the sum of distances to either coarse point
                    const auto distance_through_this_point =
                            fine_edge_matrix.coeff(fine, parent) + it.value();
//                            fine_edge_matrix.coeff(fine, parent) + fine_edge_matrix.coeff(fine, neighbor_parent);
//                            (fine_points.row(parent) - fine_points.row(fine)).norm() +
//                            (fine_points.row(neighbor_parent) - fine_points.row(fine)).norm();
                    // Set the coarse distance to the shortest path through any of the child connections
                    if (coarse_edge_matrix.coeff(parent, neighbor_parent) == 0)
                        // Make sure not to leave a value of zero
                        // this step wouldn't be necessary with an empty value of inf!
                        coarse_edge_matrix.coeffRef(parent, neighbor_parent) = distance_through_this_point;
                    else
                        coarse_edge_matrix.coeffRef(parent, neighbor_parent) = std::min(
                                coarse_edge_matrix.coeff(parent, neighbor_parent),
                                distance_through_this_point
                        );
                }
            }
        }
        return coarse_edge_matrix;
    }

    PointMatrix coarseFromMeanOfFineChildren(
            const PointMatrix &fine_points,
            const EdgeMatrix &fine_edge_matrix,
            const std::vector<Index> &fine_to_nearest_coarse,
            std::size_t num_coarse_points
    ) {

        // Find the children associated with each coarse point
        std::vector<std::set<Index>> associated_children{num_coarse_points};
        for (Index fine = 0; fine < fine_to_nearest_coarse.size(); ++fine)
            associated_children[fine_to_nearest_coarse[fine]].insert(fine);

        // "Lonely" coarse points get children based on their nearest neighbors
        // todo: is this actually helpful?
        for (auto &child_set: associated_children)
            if (child_set.size() == 1) {
                const auto &child = *child_set.begin();
                for (Eigen::SparseMatrix<double>::InnerIterator it(fine_edge_matrix, child); it; ++it) {
                    child_set.insert(it.index());
                }
            }

        // Produce coarse points by averaging the associated children
        PointMatrix coarse_points{num_coarse_points, fine_points.cols()};
        for (Index coarse = 0; coarse < coarse_points.rows(); coarse++) {
            const auto &children = associated_children[coarse];
            auto r = coarse_points.row(coarse);
            coarse_points.row(coarse) = std::transform_reduce(
                    children.begin(), children.end(),
                    Point{0.0, 0.0, 0.0},
                    std::plus{},
                    [&](auto fine) { return fine_points.row(fine); }
            ) / children.size();
        }

        return coarse_points;
    }

    std::pair<std::vector<TriangleWithNormal>, std::vector<std::vector<size_t>>> constructVoronoiTriangles(
            const PointMatrix &points,
            const EdgeMatrix &edge_matrix
    ) {

        std::vector<std::pair<Triangle, Eigen::RowVector3d>> triangles_with_normals{};
        std::vector<std::vector<size_t>> associated_triangles(points.rows());

        for (Index vertex_0 = 0; vertex_0 < points.rows(); ++vertex_0) {
            // Iterate over neighbors of the first vertex to get a "pinwheel" of triangles
            for (Eigen::SparseMatrix<double>::InnerIterator neighbor(edge_matrix, vertex_0); neighbor; ++neighbor) {
                const Index vertex_1 = neighbor.index();

                // We iterate over the vertices in order,
                // so if the neighboring idx is lower then the current idx,
                // it must have been considered before and be part of a triangle.
                if (vertex_1 < vertex_0) continue;

                // The third vertex of each triangle comes from the remaining neighbors of the first one
                for (auto other_neighbor = neighbor + Index(1); other_neighbor; ++other_neighbor) {
                    const Index vertex_2 = other_neighbor.index();

                    // As with vertex 2; if this vertex has been considered before we don't make a triangle
                    if (vertex_2 < vertex_0) continue;

                    // Only create triangles from vertices which are neighbors
                    // (we already know that vertex_2 and vertex_3 are neighbors of vertex_0)
                    // todo: what now?
                    if (edge_matrix.coeff(vertex_1, vertex_2) != 0) {

                        // Produce a normal for the triangle
                        Eigen::RowVector3d edge_01 = points.row(vertex_1) - points.row(vertex_0);
                        Eigen::RowVector3d edge_12 = points.row(vertex_2) - points.row(vertex_0);
                        auto normal = edge_01.cross(edge_12).normalized();

                        // Add the triangle
                        triangles_with_normals.emplace_back(
                                Triangle{vertex_0, vertex_1, vertex_2},
                                normal
                        );

                        // Triangle ID is equivalent to its location in the list
                        auto triangle_id = triangles_with_normals.size() - 1;

                        // Register the triangle with all the vertices it touches
                        associated_triangles[vertex_0].emplace_back(triangle_id);
                        associated_triangles[vertex_1].emplace_back(triangle_id);
                        associated_triangles[vertex_2].emplace_back(triangle_id);
                    }
                }
            }
        }

        return {triangles_with_normals, associated_triangles};
    }

    ProlongationOperator constructProlongation(
            const PointMatrix &fine_points, const PointMatrix &coarse_points,
            const EdgeMatrix &coarse_edge_matrix,
            const std::vector<Index> &fine_to_nearest_coarse,
            Weighting weighting_scheme
    ) {

        auto [triangles_with_normals, point_triangle_associations] = constructVoronoiTriangles(
                coarse_points,
                coarse_edge_matrix
        );

        // List of triplets to build prolongation operator U
        std::vector<Eigen::Triplet<double>> AllTriplet, UNeighAllTriplet;

        // Create local triangulation on each cluster (centralized at sample i)
        int notrisfound = 0;
        int edgesfound = 0;
        int fallbackCount = 0;

        // Iterate over each point
        for (Index fine = 0; fine < fine_points.rows(); ++fine) {
            Index coarse = fine_to_nearest_coarse[fine];
            Eigen::RowVector3d fine_point = fine_points.row(fine);
            Eigen::RowVector3d coarse_point = coarse_points.row(coarse);

            // Exact matches get a weight of 1.0 (if in nested mode)
            // todo: this isn't currently supported

            auto edge_iterator = Eigen::SparseMatrix<double>::InnerIterator(coarse_edge_matrix, coarse);
            if (!edge_iterator) {
                // If the coarse point has no neighbors,
                // set the weight to 1 for the coarse point.
                // todo: this should never happen!
                AllTriplet.emplace_back(fine, coarse, 1.0);

            } else if (!(edge_iterator + Index(1))) {

                // If the coarse point only has one neighbor, no triangle can be created.
                // Thus, the weights are distributed w.r.t. the distance to each coarse point.
                Index neighbor = edge_iterator.index();
                Eigen::RowVector3d neighbor_point = coarse_points.row(neighbor);

                // Get the distance to the two neighboring centroids
                Eigen::RowVector3d coarse_to_neighbor = neighbor_point - coarse_point;
                Eigen::RowVector3d coarse_to_fine = fine_points.row(fine) - coarse_point;
                double coarse_to_neighbor_length = std::max(coarse_to_neighbor.norm(), 1e-8);
                double neighbor_weight = (coarse_to_fine).dot(coarse_to_neighbor.normalized())
                                         / coarse_to_neighbor_length;
                neighbor_weight = std::clamp(neighbor_weight, 0.0, 1.0);
                double coarse_weight = 1 - neighbor_weight;

                std::vector<double> weights;
                switch (weighting_scheme) {
                    case Weighting::BARYCENTRIC:
                        AllTriplet.emplace_back(fine, coarse, coarse_weight);
                        AllTriplet.emplace_back(fine, neighbor, neighbor_weight);
                        break;
                    case Weighting::UNIFORM:
                        weights = uniformWeights(2);
                        AllTriplet.emplace_back(fine, coarse, weights[0]);
                        AllTriplet.emplace_back(fine, neighbor, weights[1]);
                        break;
                    case Weighting::INVDIST:
                        std::vector<Index> endPoints = {coarse, neighbor};
                        weights = inverseDistanceWeights(coarse_points, fine_point, endPoints);
                        AllTriplet.emplace_back(fine, coarse, weights[0]);
                        AllTriplet.emplace_back(fine, neighbor, weights[1]);
                        break;
                }
            } else {
                // This fine point's coarse parent has at least two neighbors, so we can use a triangle

                // std::cout << fine << " -> " << coarse << "\n\t";
                // for (auto neighbor: coarse_neighbors[coarse])
                //     std::cout << neighbor << ", ";
                // std::cout << std::endl;


                // Only keep the triangle with the minimum distance
                double distance_to_chosen_triangle = std::numeric_limits<double>::max();
                Eigen::RowVector3d chosen_triangle_barycenter = {0., 0., 0.};
                Triangle chosen_triangle;
                bool found_triangle = false;

                // Values are positive if inside and negative if not
                // Float value represents distance
                // todo: I don't know if I understand the role of this
                std::map<Index, float> distances_to_edges;

                // Iterate over all triangles
                for (size_t t: point_triangle_associations[coarse]) {
                    auto [triangle, triangle_normal] = triangles_with_normals[t];

                    // Rotate the triangle until the coarse index is in position 0
                    while (triangle[0] != coarse) std::rotate(triangle.begin(), triangle.begin() + 1, triangle.end());

                    // If the triangle contains the point, the distance is positive, else it's negative
                    Eigen::RowVector3d barycenter = {0., 0., 0.};
                    double distance_to_triangle = inTriangle(
                            fine_point, triangle,
                            triangle_normal, coarse_points,
                            barycenter, distances_to_edges
                    );
                    // std::cout << "\t{" << triangle[0] << ", " << triangle[1] << ", " << triangle[2] << "} "
                    //         << "--> " << distance_to_triangle << std::endl;
                    // std::cout << "\t{" << coarse_points.row(triangle[0]) << ", " << coarse_points.row(triangle[1]) << ", " << coarse_points.row(triangle[2]) << "}\n";

                    // If we've discovered a closer triangle, update the selection
                    if (distance_to_triangle >= 0. && distance_to_triangle < distance_to_chosen_triangle) {
                        found_triangle = true;
                        distance_to_chosen_triangle = distance_to_triangle;
                        chosen_triangle = triangle;
                        chosen_triangle_barycenter = barycenter;
                        break;
                    }
                }

                // If we managed to find a triangle, we can apply our weighting scheme
                // (otherwise we'll need to use a fallback, such as the nearest edge or the three nearest points)
                if (found_triangle) {
                    std::vector<double> weights;
                    switch (weighting_scheme) {
                        case Weighting::BARYCENTRIC:
                            AllTriplet.emplace_back(fine, chosen_triangle[0], chosen_triangle_barycenter(0));
                            AllTriplet.emplace_back(fine, chosen_triangle[1], chosen_triangle_barycenter(1));
                            AllTriplet.emplace_back(fine, chosen_triangle[2], chosen_triangle_barycenter(2));
                            break;
                        case Weighting::UNIFORM:
                            weights = uniformWeights(3);
                            AllTriplet.emplace_back(fine, chosen_triangle[0], weights[0]);
                            AllTriplet.emplace_back(fine, chosen_triangle[1], weights[1]);
                            AllTriplet.emplace_back(fine, chosen_triangle[2], weights[2]);
                            break;
                        case Weighting::INVDIST:
                            weights = inverseDistanceWeights(coarse_points, fine_point, chosen_triangle);
                            AllTriplet.emplace_back(fine, chosen_triangle[0], weights[0]);
                            AllTriplet.emplace_back(fine, chosen_triangle[1], weights[1]);
                            AllTriplet.emplace_back(fine, chosen_triangle[2], weights[2]);
                            break;
                    }
                } else {
                    // First fallback: attempt to find the nearest edge, and weigh by distance along that edge

                    // Find the closest edge to this point
                    // One vertex of the edge is the coarse index, so we only need to choose the other vertex
                    bool found_edge = false;
                    double distance_to_chosen_edge = std::numeric_limits<double>::max();
                    Index chosen_edge = 0;
                    for (const auto &[edge, distance]: distances_to_edges) {
                        if (distance >= 0. && distance < distance_to_chosen_edge) {
                            found_edge = true;
                            distance_to_chosen_edge = distance;
                            chosen_edge = edge;
                            break;
                        }
                    }
                    if (found_edge) {
                        ++edgesfound;
                        Eigen::RowVector3d p2 = coarse_points.row(chosen_edge);
                        Eigen::RowVector3d e12 = p2 - coarse_point;
                        double e12Length = std::max(e12.norm(), 1e-8);
                        double w2 = (fine_point - coarse_point).dot(e12.normalized()) / e12Length;
                        w2 = std::min(std::max(w2, 0.), 1.);
                        double w1 = 1. - w2;

                        std::vector<double> weights;
                        switch (weighting_scheme) {
                            case Weighting::BARYCENTRIC:
                                AllTriplet.emplace_back(fine, coarse, w1);
                                AllTriplet.emplace_back(fine, chosen_edge, w2);
                                break;
                            case Weighting::UNIFORM:
                                weights = uniformWeights(2);
                                AllTriplet.emplace_back(fine, coarse, weights[0]);
                                AllTriplet.emplace_back(fine, chosen_edge, weights[1]);
                                break;
                            case Weighting::INVDIST:
                                std::array<Index, 2> edge = {coarse, chosen_edge};
                                weights = inverseDistanceWeights(coarse_points, fine_point, edge);
                                AllTriplet.emplace_back(fine, coarse, weights[0]);
                                AllTriplet.emplace_back(fine, chosen_edge, weights[1]);
                                break;
                        }
                    } else {
                        // Second fallback: use weights based on the nearest three coarse points

                        // The first of the three is going to be the nearest coarse point
                        Triangle nearest_coarse_triangle;
                        nearest_coarse_triangle[0] = coarse;

                        // Find the distances of all the points
                        std::vector<VertexWithDistance> coarse_distances;
                        for (auto it = edge_iterator; it; ++it) {
                            auto neighbor = it.index();

                            // Skip dummy neighbors & self connection
                            if (neighbor == coarse) continue;

                            // Add this coarse distance to the list
                            auto distance_to_fine = (fine_point - coarse_points.row(neighbor)).norm();
                            coarse_distances.emplace_back(distance_to_fine, neighbor);
                        }
                        // Sort the coarse points by distance
                        std::sort(coarse_distances.begin(), coarse_distances.end(), std::less<>());
                        // Take the next two closest points (after the one we've already chosen) to complee our triangle
                        for (int j = 1; j < 3; ++j) {
                            const auto [other_distance, other_coarse] = coarse_distances[j - 1];
                            nearest_coarse_triangle[j] = other_coarse;
                        }

                        // Use inverse distance weights
                        // todo: do the other schemes not make sense here?
                        auto weights = inverseDistanceWeights(coarse_points, fine_point, nearest_coarse_triangle);
                        for (int j = 0; j < nearest_coarse_triangle.size(); j++) {
                            AllTriplet.emplace_back(fine, nearest_coarse_triangle[j], weights[j]);
                        }
                        ++fallbackCount;
                    }
                    ++notrisfound;
                }
            }
        }
        assert((float) fallbackCount / fine_points.rows() < 0.5);
        // std::cout << "Percentage of fallback: " << (double)fallbackCount / (double)fine_points.rows() * 100 <<
        //         std::endl;

        // The matrix U maps between fine points and coarse
        ProlongationOperator U;
        U.resize(fine_points.rows(), coarse_points.rows());
        U.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

        return U;
    }

    PointMatrix projectedPoints(const ProlongationOperator &weights, const PointMatrix &coarse_points) {
        PointMatrix projected_fine_points(weights.rows(), coarse_points.cols());
        projected_fine_points.setConstant(0.0);
        for (Index fine = 0; fine < weights.outerSize(); ++fine) {
            for (ProlongationOperator::InnerIterator it(weights, fine); it; ++it) {
                projected_fine_points.row(fine) += coarse_points.row(it.col()) * it.value();
            }
        }

        return projected_fine_points;
    }

}
