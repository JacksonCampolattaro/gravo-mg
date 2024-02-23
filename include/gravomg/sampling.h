#ifndef GRAVOMG_SAMPLING_H
#define GRAVOMG_SAMPLING_H

#include <Eigen/Eigen>

#include <vector>

#include "utility.h"

namespace GravoMG {

    using Eigen::Index;

    std::vector<Index> fastDiscSample(
        const Eigen::MatrixXd& pos,
        const Eigen::SparseMatrix<double>& edge_matrix,
        const double& radius
    );

}

#endif //GRAVOMG_SAMPLING_H
