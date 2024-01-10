#ifndef GRAVOMG_SAMPLING_H
#define GRAVOMG_SAMPLING_H

#include <Eigen/Eigen>

#include <vector>

namespace GravoMG {


    std::vector<size_t> maximumDeltaIndependentSet(
            const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius
    );

    std::vector<size_t> maximumDeltaIndependentSetWithDistances(
            const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges,
            const double &radius,
            Eigen::VectorXd &D, std::vector<size_t> &nearestSourceK
    );

    std::vector<size_t> fastDiscSample(const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius);
}

#endif //GRAVOMG_SAMPLING_H
