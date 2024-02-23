#include "gravomg/sampling.h"

#include <iostream>

namespace GravoMG {

    std::vector<Index> fastDiscSample(
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
