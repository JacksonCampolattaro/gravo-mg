#ifndef GRAVOMG_SAMPLING_H
#define GRAVOMG_SAMPLING_H

#include <Eigen/Eigen>

#include <vector>

namespace GravoMG {


    std::vector<int>
    maximumDeltaIndependentSet(const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius);

    std::vector<int>
    maximumDeltaIndependentSet(const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius,
                               Eigen::VectorXd &D, std::vector<size_t> &nearestSourceK);

    std::vector<int> fastDiskSample(const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius,
                                    Eigen::VectorXd &D, std::vector<size_t> &nearestSourceK);
}

#endif //GRAVOMG_SAMPLING_H
