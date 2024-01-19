#include "gravomg/multigrid.h"
#include "gravomg/utility.h"
#include "gravomg/sampling.h"

#include <cmath>
#include <numeric>
#include <chrono>

#include <Eigen/Dense>
#include <utility>

namespace GravoMG {

    double inTriangle(const Eigen::RowVector3d& p, std::span<Index, 3> tri,
                      const Eigen::RowVector3d& triNormal, const Eigen::MatrixXd& pos,
                      Eigen::RowVector3d& bary, std::map<Index, float>& insideEdge) {
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

    std::vector<double> uniformWeights(const int& n_points) {
        std::vector<double> weights(n_points);
        std::fill(weights.begin(), weights.end(), 1. / n_points);
        return weights;
    }

    std::vector<double> inverseDistanceWeights(const Eigen::MatrixXd& pos, const Eigen::RowVector3d& p,
                                               const std::span<Index>& edges) {
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

    void constructDijkstraWithCluster(const Eigen::MatrixXd& points, const std::vector<Index>& source,
                                      const NeighborMatrix& neigh,
                                      Eigen::VectorXd& D,
                                      std::vector<Index>& nearestSourceK) {
        std::priority_queue<VertexPair, std::vector<VertexPair>, std::greater<>> DistanceQueue;
        if (nearestSourceK.empty()) nearestSourceK.resize(points.rows(), source[0]);

        for (size_t i = 0; i < source.size(); ++i) {
            D(source[i]) = 0.0;
            VertexPair vp{source[i], D(source[i])};
            DistanceQueue.push(vp);
            nearestSourceK[source[i]] = i;
        }

        size_t curSource;
        while (!DistanceQueue.empty()) {
            VertexPair vp1 = DistanceQueue.top();
            curSource = nearestSourceK[vp1.vId];
            Eigen::RowVector3d vertex1 = points.row(vp1.vId);
            DistanceQueue.pop();

            for (size_t i = 0; i < neigh.cols(); ++i) {
                Index vNeigh = neigh(vp1.vId, i);

                if (vNeigh >= 0) {
                    double dist, distTemp;
                    Eigen::RowVector3d vertex2 = points.row(vNeigh);
                    dist = (vertex2 - vertex1).norm();
                    distTemp = vp1.distance + dist;
                    if (distTemp < D(vNeigh)) {
                        // Assign a new distance
                        D(vNeigh) = distTemp;
                        VertexPair v2{vNeigh, distTemp};
                        DistanceQueue.push(v2);


                        // Assign the nearest source to a certain point
                        nearestSourceK[vNeigh] = curSource;
                    }
                }
            }
        }
    }

    std::vector<Index> assignParents(
        const Eigen::MatrixXd& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& coarse_samples
    ) {
        std::vector<Index> parents(fine_points.rows());
        Eigen::VectorXd parent_distances(fine_points.rows());
        parent_distances.setConstant(std::numeric_limits<double>::max());

        std::priority_queue<VertexPair, std::vector<VertexPair>, std::greater<>> distance_queue;

        // Self-interactions have zero path length
        for (Index coarse = 0; coarse < coarse_samples.size(); ++coarse) {
            parents[coarse_samples[coarse]] = coarse;
            distance_queue.emplace(coarse_samples[coarse], 0.0);
            parent_distances[coarse_samples[coarse]] = 0.0;
        }

        // Treat connections in order of path length
        while (!distance_queue.empty()) {

            // Pop the connection with the shortest path length
            const VertexPair pair = distance_queue.top();
            distance_queue.pop();
            const auto fine_path_length_to_coarse = pair.distance;
            const auto fine = pair.vId;
            const auto fine_point = fine_points.row(fine);

            // Treat every neighbor of this point to search for a shorter path
            for (const auto neighbor: fine_neighbors[fine]) {
                const auto neighbor_point = fine_points.row(neighbor);
                const auto neighbor_to_fine_distance = (fine_point - neighbor_point).norm();
                const auto neighbor_path_length_to_course = neighbor_to_fine_distance + fine_path_length_to_coarse;

                // If we find a shorter path for a neighbor...
                if (neighbor_path_length_to_course < parent_distances[neighbor]) {

                    // Update its nearest parent & associated distance
                    parents[neighbor] = parents[fine];
                    parent_distances[neighbor] = neighbor_path_length_to_course;

                    // Add it to the queue
                    distance_queue.emplace(neighbor, neighbor_path_length_to_course);
                }

            }
        }

        return parents;
    }

    double averageEdgeLength(const Eigen::MatrixXd& positions, const NeighborList& neighbors) {
        double sumLength = 0;
        int nEdges = 0;
        for (size_t i = 0; i < positions.rows(); ++i) {
            Eigen::Vector3d p1 = positions.row(i);
            for (auto neighbor: neighbors[i]) {
                Eigen::Vector3d p2 = positions.row(neighbor);
                double dist = (p1 - p2).norm();
                if (dist > 0) {
                    sumLength += dist;
                    ++nEdges;
                }
            }
        }
        return sumLength / (double)nEdges;
    }

    NeighborList extractCoarseEdges(
        const PointMatrix& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& coarse_samples,
        const std::vector<Index>& fine_to_nearest_coarse
    ) {

        std::vector<std::set<Index>> coarse_to_coarse_neighbors{coarse_samples.size()};
        for (Index fine = 0; fine < fine_points.rows(); ++fine) {
            for (auto& neighbor: fine_neighbors[fine]) {
                // If the two points belong to different parent points
                if (fine_to_nearest_coarse[fine] != fine_to_nearest_coarse[neighbor]) {
                    // Ensure the other point's parent is in the neighbor list
                    coarse_to_coarse_neighbors[fine_to_nearest_coarse[fine]].insert(fine_to_nearest_coarse[neighbor]);
                }
            }
        }

        return coarse_to_coarse_neighbors;

    }

    PointMatrix coarseFromMeanOfFineChildren(
        const PointMatrix& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& fine_to_nearest_coarse,
        std::size_t num_coarse_points
    ) {

        // Find the children associated with each coarse point
        std::vector<std::set<Index>> associated_children{num_coarse_points};
        for (Index fine = 0; fine < fine_to_nearest_coarse.size(); ++fine)
            associated_children[fine_to_nearest_coarse[fine]].insert(fine);

        // "Lonely" coarse points get children based on thier nearest neighbors
        for (auto& child_set: associated_children)
            if (child_set.size() == 1) {
                const auto& neighbors = fine_neighbors[*child_set.begin()];
                child_set.insert(neighbors.begin(), neighbors.end());
            }

        // Produce coarse points by averaging the associated children
        PointMatrix coarse_points{num_coarse_points, fine_points.cols()};
        for (Index coarse = 0; coarse < coarse_points.rows(); coarse++) {
            const auto& children = associated_children[coarse];
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
        const PointMatrix& points,
        const NeighborList& edges
    ) {

        std::vector<std::pair<Triangle, Eigen::RowVector3d>> triangles_with_normals{};
        std::vector<std::vector<size_t>> associated_triangles(points.rows());

        for (Index vertex_0 = 0; vertex_0 < points.rows(); ++vertex_0) {
            // Iterate over neighbors of the first vertex to get a "pinwheel" of triangles
            const auto& neighbors = edges[vertex_0];
            for (auto neighbor = neighbors.begin(); neighbor != neighbors.end(); ++neighbor) {
                const Index vertex_1 = *neighbor;

                // We iterate over the vertices in order,
                // so if the neighboring idx is lower then the current idx,
                // it must have been considered before and be part of a triangle.
                if (vertex_1 < vertex_0) continue;

                // The third vertex of each triangle comes from the remaining neighbors of the first one
                for (auto other_neighbor = std::next(neighbor); other_neighbor != neighbors.end(); other_neighbor++) {
                    const Index vertex_2 = *other_neighbor;

                    // As with vertex 2; if this vertex has been considered before we don't make a triangle
                    if (vertex_2 < vertex_0) continue;

                    // Only create triangles from vertices which are neighbors
                    // (we already know that vertex_2 and vertex_3 are neighbors of vertex_0)
                    if (edges[vertex_1].contains(vertex_2)) {

                        // Prodce a normal for the triangle
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
        const PointMatrix& fine_points, const PointMatrix& coarse_points,
        const NeighborList& coarse_neighbors,
        const std::vector<Index>& fine_to_nearest_coarse,
        Weighting weighting_scheme
    ) {

        auto [triangles_with_normals, point_triangle_associations] = constructVoronoiTriangles(
            coarse_points,
            coarse_neighbors
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

            if (coarse_neighbors[coarse].empty()) {
                // If the coarse point has no neighbors,
                // set the weight to 1 for the coarse point.
                // todo: this should never happen!
                AllTriplet.emplace_back(fine, coarse, 1.0);

            } else if (coarse_neighbors[coarse].size() == 1) {

                // If the coarse point only has one neighbor, no triangle can be created.
                // Thus, the weights are distributed w.r.t. the distance to each coarse point.
                Index neighbor = *coarse_neighbors[coarse].begin();
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
                    case BARYCENTRIC:
                        AllTriplet.emplace_back(fine, coarse, coarse_weight);
                        AllTriplet.emplace_back(fine, neighbor, neighbor_weight);
                        break;
                    case UNIFORM:
                        weights = uniformWeights(2);
                        AllTriplet.emplace_back(fine, coarse, weights[0]);
                        AllTriplet.emplace_back(fine, neighbor, weights[1]);
                        break;
                    case INVDIST:
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
                        case BARYCENTRIC:
                            AllTriplet.emplace_back(fine, chosen_triangle[0], chosen_triangle_barycenter(0));
                            AllTriplet.emplace_back(fine, chosen_triangle[1], chosen_triangle_barycenter(1));
                            AllTriplet.emplace_back(fine, chosen_triangle[2], chosen_triangle_barycenter(2));
                            break;
                        case UNIFORM:
                            weights = uniformWeights(3);
                            AllTriplet.emplace_back(fine, chosen_triangle[0], weights[0]);
                            AllTriplet.emplace_back(fine, chosen_triangle[1], weights[1]);
                            AllTriplet.emplace_back(fine, chosen_triangle[2], weights[2]);
                            break;
                        case INVDIST:
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
                    for (const auto& [edge, distance]: distances_to_edges) {
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
                            case BARYCENTRIC:
                                AllTriplet.emplace_back(fine, coarse, w1);
                                AllTriplet.emplace_back(fine, chosen_edge, w2);
                                break;
                            case UNIFORM:
                                weights = uniformWeights(2);
                                AllTriplet.emplace_back(fine, coarse, weights[0]);
                                AllTriplet.emplace_back(fine, chosen_edge, weights[1]);
                                break;
                            case INVDIST:
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
                        std::vector<VertexPair> coarse_distances;
                        for (Index neighbor: coarse_neighbors[coarse]) {
                            // Skip dummy neighbors & self connection
                            if (neighbor == coarse) continue;

                            // Add this coarse distance to the list
                            auto distance_to_fine = (fine_point - coarse_points.row(neighbor)).norm();
                            coarse_distances.emplace_back(neighbor, distance_to_fine);
                        }
                        // Sort the coarse points by distance
                        std::sort(coarse_distances.begin(), coarse_distances.end(), std::less<>());
                        // Take the next two closest points (after the one we've already chosen) to complee our triangle
                        for (int j = 1; j < 3; ++j) {
                            nearest_coarse_triangle[j] = coarse_distances[j - 1].vId;
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
        assert((float)fallbackCount / fine_points.rows() < 0.5);
        // std::cout << "Percentage of fallback: " << (double)fallbackCount / (double)fine_points.rows() * 100 <<
        //         std::endl;

        // The matrix U maps between fine points and coarse
        ProlongationOperator U;
        U.resize(fine_points.rows(), coarse_points.rows());
        U.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

        return U;
    }

    PointMatrix projectedPoints(const ProlongationOperator& weights, const PointMatrix& coarse_points) {
        PointMatrix projected_fine_points(weights.rows(), coarse_points.cols());
        projected_fine_points.setConstant(0.0);
        for (Index fine = 0; fine < weights.outerSize(); ++fine) {
            for (ProlongationOperator::InnerIterator it(weights, fine); it; ++it) {
                projected_fine_points.row(fine) += coarse_points.row(it.col()) * it.value();
                //auto row = coarse_points.row(it.col()) * it.value();
                //std::cout << it.row() << " " << it.col()
                //        << " --> " << it.value()
                //     value   << std::endl;
            }
        }

        return projected_fine_points;
    }

}
