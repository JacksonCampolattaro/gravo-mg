
#include "gravomg/multigrid.h"
#include "gravomg/utility.h"
#include "gravomg/sampling.h"

#include <cmath>
#include <numeric>
#include <chrono>

#include <Eigen/Dense>
#include <utility>

namespace GravoMG {

    double inTriangle(const Eigen::RowVector3d&p, std::span<Index, 3> tri,
                      const Eigen::RowVector3d&triNormal, const Eigen::MatrixXd&pos,
                      Eigen::RowVector3d&bary, std::map<Index, float>&insideEdge) {
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

    std::vector<double> uniformWeights(const int&n_points) {
        std::vector<double> weights(n_points);
        std::fill(weights.begin(), weights.end(), 1. / n_points);
        return weights;
    }

    std::vector<double> inverseDistanceWeights(const Eigen::MatrixXd&pos, const Eigen::RowVector3d&p,
                                               const std::span<Index>&edges) {
        double sumWeight = 0.;
        std::vector<double> weights(edges.size());
        for (size_t j = 0; j < edges.size(); ++j) {
            weights[j] = 1. / max(1e-8, (p - pos.row(edges[j])).norm());
            sumWeight += weights[j];
        }
        for (size_t j = 0; j < weights.size(); ++j) {
            weights[j] = weights[j] / sumWeight;
        }
        return weights;
    }

    void constructDijkstraWithCluster(const Eigen::MatrixXd&points, const std::vector<Index>&source,
                                      const EdgeMatrix&neigh, Eigen::VectorXd&D,
                                      vector<Index>&nearestSourceK) {
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

    double averageEdgeLength(const Eigen::MatrixXd&pos, const EdgeMatrix&neigh) {
        double sumLength = 0;
        int nEdges = 0;
        for (size_t i = 0; i < pos.rows(); ++i) {
            Eigen::Vector3d p1 = pos.row(i);
            for (size_t j = 0; j < neigh.cols(); ++j) {
                if (neigh(i, j) < 0) continue;
                Eigen::Vector3d p2 = pos.row(neigh(i, j));
                double dist = (p1 - p2).norm();
                if (dist > 0) {
                    sumLength += dist;
                    ++nEdges;
                }
            }
        }
        return sumLength / (double)nEdges;
    }


    Eigen::SparseMatrix<double> constructProlongation(
        const Eigen::MatrixXd&fine_points,
        const EdgeMatrix&fine_edges,
        const std::vector<Index>&coarse_samples,
        Weighting weighting_scheme,
        bool verbose, bool nested
    ) {

        // Nearest source for every given point
        std::vector<Index> fine_to_nearest_coarse(fine_points.rows());

        // Distance of the nearest source for every point (initialized to max)
        Eigen::VectorXd fine_to_nearest_coarse_distance(fine_points.rows());
        fine_to_nearest_coarse_distance.setConstant(std::numeric_limits<double>::max());

        // Compute distance from fine points to coarse points and get the closest coarse point
        constructDijkstraWithCluster(
            fine_points, coarse_samples,
            fine_edges, fine_to_nearest_coarse_distance, fine_to_nearest_coarse
        );

        // Create neighborhood for the next level
        std::vector<std::set<Index>> coarse_to_coarse_neighbors{coarse_samples.size()};
        for (Index fine = 0; fine < fine_points.rows(); ++fine) {
            for (auto&neighbor: fine_edges.row(fine)) {
                if (neighbor < 0) break;
                // If the two points belong to different parent points
                if (fine_to_nearest_coarse[fine] != fine_to_nearest_coarse[neighbor]) {
                    // Ensure the other point's parent is in the neighbor list
                    coarse_to_coarse_neighbors[fine_to_nearest_coarse[fine]].insert(fine_to_nearest_coarse[neighbor]);
                }
            }
        }

        // Convert the set-based neighbor list to a standard homogenuous table
        auto max_num_neighbors = std::transform_reduce(
                                     coarse_to_coarse_neighbors.begin(), coarse_to_coarse_neighbors.end(),
                                     std::size_t{0}, [](const auto&a, const auto&b) { return std::max(a, b); },
                                     [](const auto&set) { return set.size(); }
                                 ) + 1; // todo: I think this is necessary to leave room for self-connections
        EdgeMatrix coarse_edges{coarse_samples.size(), max_num_neighbors};
        coarse_edges.setConstant(-1); // Unused slots are set to -1
        for (Index coarse = 0; coarse < coarse_edges.rows(); ++coarse) {
            // Add self-connection
            coarse_edges(coarse, 0) = coarse;

            // Add all connections from the set
            Index j{1};
            for (auto neighbor: coarse_to_coarse_neighbors[coarse]) {
                if (neighbor == coarse) continue;
                coarse_edges(coarse, j++) = neighbor;
            }
        }

        if (verbose) std::cout << "Setting up the point locations for the next level\n";

        // Setting up the DoF for the next level
        // tempPoints are the centers of the voronoi cells, each row for each voronoi cells
        Eigen::MatrixXd coarse_points(coarse_samples.size(), fine_points.cols());
        coarse_points.setZero();
        if (nested) {
            // If the user chooses, the coarse points can be the same as the sampled points
            coarse_points = fine_points(coarse_samples, Eigen::all);
        } else {
            // Otherwise, coarse points will be produced by averaging the related fine points

            // Accumulate point coordinates
            std::vector<int> cluster_sizes(coarse_samples.size());
            for (Index fine = 0; fine < fine_points.rows(); ++fine) {
                // Each fine point gets included in the nearest coarse point
                auto coarse = fine_to_nearest_coarse[fine];
                coarse_points.row(coarse) += fine_points.row(fine);
                ++cluster_sizes[coarse];
            }
            for (Index coarse = 0; coarse < coarse_samples.size(); ++coarse) {
                if (cluster_sizes[coarse] == 1) {
                    // If this coarse point is only associated with one fine point
                    // fall back to including all of its neighbors
                    // todo: what is the motivation for this?
                    coarse_points.row(coarse) = fine_points.row(coarse_samples[coarse]); // todo: shouldn't be necessary
                    for (Index neighbor: coarse_to_coarse_neighbors[coarse]) {
                        coarse_points.row(coarse) += fine_points.row(coarse_samples[neighbor]);
                    }
                    coarse_points.row(coarse) /= ((double)coarse_to_coarse_neighbors[coarse].size() + 1.0);
                } else {
                    // Divide by the number of points included in teh average
                    coarse_points.row(coarse) /= cluster_sizes[coarse];
                }
            }
        }

        // Create triangles for this level based on Voronoi cells
        std::vector<Triangle> triangles;
        triangles.reserve(coarse_samples.size() * max_num_neighbors);
        std::vector<std::vector<size_t>> connected_triangles{coarse_samples.size()};
        std::vector<Eigen::RowVector3d> triangle_normals{coarse_samples.size() * max_num_neighbors};
        size_t current_triangle = 0;
        for (Index coarse = 0; coarse < coarse_samples.size(); ++coarse) {
            // Iterate over neighbors of the coarse point to get a "pinwheel" of triangles
            const auto&neighbors = coarse_to_coarse_neighbors[coarse];
            for (auto neighbor = neighbors.begin(); neighbor != neighbors.end(); ++neighbor) {
                Index vertex_2 = *neighbor;
                // We iterate over the coarse indices in order,
                // so if the neighboring idx is lower then the current coarseIdx,
                // it must have been considered before and be part of a triangle.
                if (vertex_2 < coarse) continue;

                // The third vertex of each triangle comes from the remaining neighbors of the coarse point
                for (auto other_neighbor = std::next(neighbor); other_neighbor != neighbors.end(); other_neighbor++) {
                    Index vertex_3 = *other_neighbor;

                    // As with vertex 2; if this vertex has been considered before we don't make a triangle
                    if (vertex_3 < coarse) continue;

                    // Only create triangles from vertices which are neighbors
                    // (we already know that vertex_2 and vertex_3 are neighbors of the coarse point)
                    if (coarse_to_coarse_neighbors[vertex_2].find(vertex_3)
                        == coarse_to_coarse_neighbors[vertex_2].end())
                        continue;

                    // Add the triangle
                    triangles.push_back({coarse, vertex_2, vertex_3});

                    // Prodce a normal for the triangle
                    Eigen::RowVector3d e12 = coarse_points.row(vertex_2) - coarse_points.row(coarse);
                    Eigen::RowVector3d e13 = coarse_points.row(vertex_3) - coarse_points.row(coarse);
                    triangle_normals.push_back(e12.cross(e13).normalized());

                    // Register the triangle with all the vertices it touches
                    connected_triangles[coarse].emplace_back(current_triangle);
                    connected_triangles[vertex_2].emplace_back(current_triangle);
                    connected_triangles[vertex_3].emplace_back(current_triangle);

                    ++current_triangle;
                }
            }
        }
        triangles.shrink_to_fit();
        triangle_normals.shrink_to_fit();

        // List of triplets to build prolongation operator U
        std::vector<Eigen::Triplet<double>> AllTriplet, UNeighAllTriplet;

        // Create local triangulation on each cluster (centralized at sample i)
        int notrisfound = 0;
        int edgesfound = 0;
        int fallbackCount = 0;

        // Iterate over each point
        for (Index fine = 0; fine < fine_points.rows(); ++fine) {
            Eigen::RowVector3d fine_point = fine_points.row(fine);
            Index coarse = fine_to_nearest_coarse[fine];
            Eigen::RowVector3d coarse_point = coarse_points.row(coarse);

            // Exact matches get a weight of 1.0 (if in nested mode)
            if (nested && coarse_samples[fine_to_nearest_coarse[fine]] == fine) {
                AllTriplet.emplace_back(fine, coarse, 1.);
                continue;
            }

            // If the coarse point has no neighbors,
            // set the weight to 1 for the coarse point.
            if (coarse_to_coarse_neighbors[coarse].empty()) {
                // todo: this should never happen!
                AllTriplet.emplace_back(fine, coarse, 1.);
            } else if (coarse_to_coarse_neighbors[coarse].size() == 1) {
                // If the coarse point only has one neighbor, no triangle can be created.
                // Thus, the weights are distributed w.r.t. the distance to each coarse point.
                Index neighbor = *coarse_to_coarse_neighbors[coarse].begin();
                Eigen::RowVector3d neighbor_point = coarse_points.row(neighbor);

                // get the distance to the two neighboring centroids
                Eigen::RowVector3d coarse_to_neighbor = neighbor_point - coarse_point;
                Eigen::RowVector3d coarse_to_fine = fine_points.row(fine) - coarse_point;
                double coarse_to_neighbor_length = max(coarse_to_neighbor.norm(), 1e-8);
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
                for (size_t t: connected_triangles[coarse]) {
                    auto triangle = triangles[t];
                    auto triangle_normal = triangle_normals[t];

                    // Rotate the triangle until the coarse index is in position 0
                    while (triangle[0] != coarse) std::rotate(triangle.begin(), triangle.begin() + 1, triangle.end());

                    Eigen::RowVector3d barycenter = {0., 0., 0.};
                    // If the triangle contains the point, the distance is positive, else it's negative
                    double distance_to_triangle = inTriangle(
                        fine_point, triangle,
                        triangle_normal, coarse_points,
                        barycenter, distances_to_edges
                    );

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
                    for (const auto&[edge, distance]: distances_to_edges) {
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
                        double e12Length = max(e12.norm(), 1e-8);
                        double w2 = (fine_point - coarse_point).dot(e12.normalized()) / e12Length;
                        w2 = min(max(w2, 0.), 1.);
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
                        for (Index neighbor: coarse_edges.row(coarse)) {
                            // Skip dummy neighbors & self connection
                            if (neighbor < 0 || neighbor == coarse) continue;

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
        if (verbose)
            cout << "Percentage of fallback: " << (double)fallbackCount / (double)fine_points.rows() * 100 <<
                    endl;

        // The matrix U maps between fine points and coarse
        Eigen::SparseMatrix<double> U;
        U.resize(fine_points.rows(), coarse_points.rows());
        U.setFromTriplets(AllTriplet.begin(), AllTriplet.end());

        // todo Return the following:
        // coarse points (voronoi centers)
        // edges in the coarse point cloud
        // the matrix U
        return U;
    }

}
