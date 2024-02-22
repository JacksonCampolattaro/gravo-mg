#include "gravomg/sampling.h"

#include <iostream>

namespace GravoMG {
    std::vector<size_t> maximumDeltaIndependentSet(
        const Eigen::MatrixXd& pos,
        const Eigen::MatrixXi& edges,
        const double& radius
    ) {
        std::vector<bool> visited(edges.rows());
        std::vector<size_t> selection;
        for (int i = 0; i < edges.rows(); ++i) {
            if (!visited[i]) {
                selection.push_back(i);
                for (int j = 0; j < edges.cols(); ++j) {
                    int neighIdx = edges(i, j);
                    if (neighIdx < 0) break;
                    double dist = (pos.row(i) - pos.row(neighIdx)).norm();
                    if (dist < radius) {
                        visited[neighIdx] = true;
                    }
                }
            }
        }
        return selection;
    }

    std::vector<Index> maximumDeltaIndependentSetWithDistances(
        const Eigen::MatrixXd& pos,
        const NeighborMatrix& edges,
        const double& radius,
        Eigen::VectorXd& D,
        std::vector<Index>& nearestSourceK
    ) {
        std::vector<bool> visited(edges.rows());
        std::vector<Index> selection;
        int sampleIdx = 0;
        for (int i = 0; i < edges.rows(); ++i) {
            if (!visited[i]) {
                selection.push_back(i);
                nearestSourceK[i] = sampleIdx;
                for (int j = 0; j < edges.cols(); ++j) {
                    int neighIdx = edges(i, j);
                    if (neighIdx < 0) break;
                    double dist = (pos.row(i) - pos.row(neighIdx)).norm();
                    if (dist < radius) {
                        visited[neighIdx] = true;
                        if (dist < D(neighIdx)) {
                            D(neighIdx) = dist;
                            nearestSourceK[neighIdx] = sampleIdx;
                        }
                    }
                }
                ++sampleIdx;
            }
        }
        return selection;
    }

    std::vector<Index> fastDiscSample(
        const Eigen::MatrixXd& pos, const NeighborMatrix& edges, const double& radius
    ) {
        // todo: in the future it might make sense to return these!
        Eigen::VectorXd D(edges.size());
        D.setConstant(std::numeric_limits<double>::max());

        std::vector<bool> visited(edges.rows());
        std::vector<Index> selection{};
        int sampleIdx = 0;
        for (int i = 0; i < edges.rows(); ++i) {
            if (!visited[i]) {
                selection.push_back(i);
                for (int j = 0; j < edges.cols(); ++j) {
                    int neighIdx = edges(i, j);
                    if (neighIdx < 0) break;
                    double dist = (pos.row(i) - pos.row(neighIdx)).norm();
                    if (dist < radius) {
                        visited[neighIdx] = true;
                        if (dist < D(neighIdx)) {
                            D(neighIdx) = dist;
                        }
                        for (int j2 = 0; j2 < edges.cols(); ++j2) {
                            int neighIdx2 = edges(neighIdx, j2);
                            if (neighIdx2 < 0) break;
                            double dist2 = dist + (pos.row(neighIdx) - pos.row(neighIdx2)).norm();
                            if (dist2 < radius) {
                                visited[neighIdx2] = true;
                                if (dist2 < D(neighIdx2)) {
                                    D(neighIdx2) = dist2;
                                }
                            }
                        }
                    }
                }
                ++sampleIdx;
            }
        }
        for (const auto &i : selection) std::cout << i << " ";
        std::cout << std::endl;
        return selection;
    }

    std::vector<Index> fastDiscSampleCOO(
        const Eigen::MatrixXd& pos,
        const Eigen::SparseMatrix<double>& edge_matrix,
        const double& radius
        // todo: this would be a useful feature!
        //std::size_t min_selection = 0, std::size_t max_selection = std::numeric_limits<std::size_t>::max()
    ) {

        Eigen::VectorXd distances(pos.rows());
        distances.setConstant(std::numeric_limits<double>::max());
        std::vector<Index> nearest_source(pos.rows());

        std::vector<bool> visited(pos.rows());
        std::vector<Index> selection{};

        for (Index i = 0; i < edge_matrix.outerSize(); ++i) {

            // Skip indices which have already been visited
            if (visited[i]) continue;

            // Select this point
            selection.push_back(i);

            // Mark neighbors as visited
            for (Eigen::SparseMatrix<double>::InnerIterator it(edge_matrix, i); it; ++it) {

                // If a neighbor is too close, reject it
                const auto neighbor_distance = (pos.row(it.index()) - pos.row(i)).norm();
                if (neighbor_distance < radius) {

                    // Mark this point rejected
                    visited[it.index()] = true;

                    // Mark neighbors of neighbors as visited, too
                    // todo: could this be improved by also checking neighbors of neighbors of neighbors, etc. ?
                    // Also, why do we use edge+edge distance instead of direct?
                    for (Eigen::SparseMatrix<double>::InnerIterator it2(edge_matrix, it.index()); it2; ++it2) {
                        if (neighbor_distance + (pos.row(it2.index()) - pos.row(it.index())).norm() < radius)
                            visited[it2.index()] = true;
                    }

                }
            }
        }

        return selection;
    }

}
