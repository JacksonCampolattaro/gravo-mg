#include "gravomg/utility.h"

#include <fstream>


namespace GravoMG {

    void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio) {
        Eigen::Vector3d minV;
        Eigen::Vector3d maxV;
        Eigen::Vector3d length;
        Eigen::MatrixXd MatSubs(V.rows(), V.cols());
        Eigen::MatrixXd MatAdd(V.rows(), V.cols());
        double maxVal;
        double scalingRatio = scaleRatio; //dia-meter, not radius


        /* Get the min and max coefficients */
        for (int i = 0; i < V.cols(); i++) {
            minV(i) = V.col(i).minCoeff();
            maxV(i) = V.col(i).maxCoeff();
            length(i) = maxV(i) - minV(i);
            MatSubs.col(i).setConstant(minV(i));
        }

        maxVal = length.maxCoeff();

        /* Translate to the Origin */
        V = V - MatSubs;

        /* Scale w.r.t the longest axis */
        V = V * (scalingRatio / maxVal);

        for (int i = 0; i < V.cols(); i++) {
            maxV(i) = V.col(i).maxCoeff();
            MatAdd.col(i).setConstant(0.5 * maxV(i));
        }

        /* Translate s.t. the center is in the Origin */
        V = V - MatAdd;

        for (int i = 0; i < V.cols(); i++) {
            minV(i) = V.col(i).minCoeff();
            maxV(i) = V.col(i).maxCoeff();
            length(i) = maxV(i) - minV(i);
        }
        maxVal = length.maxCoeff();
    }

    NeighborMatrix toHomogenous(const NeighborList& edges) {
        // Convert the set-based neighbor list to a standard homogenuous table

        // Prepare a matrix with room for the largest edge set
        const auto max_num_neighbors = std::transform_reduce(
                                           edges.begin(), edges.end(),
                                           std::size_t{0},
                                           [](const auto& a, const auto& b) { return std::max(a, b); },
                                           [](const auto& set) { return set.size(); }
                                       ) + 1;
        NeighborMatrix edge_matrix{edges.size(), max_num_neighbors};

        // Unused slots are set to -1
        edge_matrix.setConstant(-1);

        // Set edges row by row
        for (Index i = 0; i < edge_matrix.rows(); ++i) {
            // Add self-connection in the first position
            edge_matrix(i, 0) = i;

            // Add all connections from the set
            Index j{1};
            for (const auto neighbor: edges[i]) {
                if (neighbor == i) continue;
                edge_matrix(i, j++) = neighbor;
            }
        }

        return edge_matrix;
    }


    NeighborList extractEdges(const Eigen::SparseMatrix<double>& matrix) {
        NeighborList neighbors(matrix.rows());
        for (Index i = 0; i < matrix.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it) {
                neighbors[it.row()].insert(it.col());
            }
        }
        return neighbors;
    }

}
