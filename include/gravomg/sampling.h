#ifndef GRAVOMG_SAMPLING_H
#define GRAVOMG_SAMPLING_H

#include <Eigen/Eigen>

#include <vector>

#include "multigrid.h"

namespace GravoMG {

    using Eigen::Index;

    std::vector<size_t> maximumDeltaIndependentSet(
            const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius
    );

    std::vector<Index> maximumDeltaIndependentSetWithDistances(
        const Eigen::MatrixXd &pos, const EdgeMatrix&edges,
        const double &radius,
        Eigen::VectorXd &D, std::vector<Index>&nearestSourceK
    );

    std::vector<Index> fastDiscSample(const Eigen::MatrixXd&pos, const EdgeMatrix& edges, const double&radius);
}

#endif //GRAVOMG_SAMPLING_H
