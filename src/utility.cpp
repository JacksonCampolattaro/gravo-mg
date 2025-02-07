#include "gravomg/utility.h"

#include <fstream>


namespace GravoMG {

    void scaleMesh(Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double scaleRatio) {
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

    EdgeMatrix toEdgeDistanceMatrix(const EdgeMatrix &matrix, const PointMatrix &points) {
        auto distance_matrix = matrix;
        for (Index i = 0; i < matrix.outerSize(); ++i)
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it)
                distance_matrix.coeffRef(it.row(), it.col()) = (points.row(it.row()) - points.row(it.col())).norm();
        return distance_matrix;
    }

    std::pair<EdgeList, Eigen::VectorXd> extractEdges(const EdgeMatrix &matrix) {
        EdgeList neighbors{matrix.nonZeros(), 2};
        Eigen::VectorXd values{matrix.nonZeros()};
        // todo: this could probably be reduced to a bulk memory copy
        Index n = 0;
        for (Index i = 0; i < matrix.outerSize(); ++i)
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it) {
                values[n] = it.value();
                neighbors.row(n) = Eigen::RowVector2i{it.col(), it.row()};
                n++;
            }

        return {neighbors, values};
    }

}
