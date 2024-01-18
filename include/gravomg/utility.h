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
    using NeighborList = std::vector<std::set<Index>>;
    using Triangle = std::array<Index, 3>;
    using TriangleWithNormal = std::pair<Triangle, Normal>;

    void scaleMesh(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleRatio = 1.0);

    NeighborMatrix toHomogenous(const NeighborList& edges);

    NeighborList extractEdges(const Eigen::SparseMatrix<double>& matrix);

}

#endif // !UTILITY_H
