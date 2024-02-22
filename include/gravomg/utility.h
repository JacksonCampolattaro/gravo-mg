#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <set>
#include <vector>
#include <random>

#include <Eigen/Eigen>

namespace GravoMG {

    using Eigen::Index;
    using Point = Eigen::RowVector3d;
    using Normal = Eigen::RowVector3d;
    using PointMatrix = Eigen::MatrixXd;
    using NeighborMatrix = Eigen::MatrixXi;
    using NeighborList = Eigen::MatrixXi;
    using EdgeMatrix = Eigen::SparseMatrix<double>;
    using Triangle = std::array<Index, 3>;
    using TriangleWithNormal = std::pair<Triangle, Normal>;
    using ProlongationOperator = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio = 1.0);

    NeighborList extractEdges(const EdgeMatrix& matrix);

    struct SparseInnerEnd {
        template<typename Derived>
        bool operator==(const typename Eigen::SparseCompressedBase<Derived>::InnerIterator& iter) const {
            return !iter;
        }
    };

}

#endif // !UTILITY_H
