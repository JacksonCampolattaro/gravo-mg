
#include "gravomg/sampling.h"

namespace GravoMG {

    std::vector<int> maximumDeltaIndependentSet(
            const Eigen::MatrixXd &pos,
            const Eigen::MatrixXi &edges,
            const double &radius
    ) {
        std::vector<bool> visited(edges.rows());
        std::vector<int> selection;
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

    std::vector<int> maximumDeltaIndependentSetWithDistances(
            const Eigen::MatrixXd &pos,
            const Eigen::MatrixXi &edges,
            const double &radius,
            Eigen::VectorXd &D,
            std::vector<size_t> &nearestSourceK
    ) {
        std::vector<bool> visited(edges.rows());
        std::vector<int> selection;
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

    std::vector<int> fastDiskSample(
            const Eigen::MatrixXd &pos, const Eigen::MatrixXi &edges, const double &radius,
            Eigen::VectorXd &D, std::vector<size_t> &nearestSourceK
    ) {
        std::vector<bool> visited(edges.rows());
        std::vector<int> selection;
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
                        for (int j2 = 0; j2 < edges.cols(); ++j2) {
                            int neighIdx2 = edges(neighIdx, j2);
                            if (neighIdx2 < 0) break;
                            double dist2 = dist + (pos.row(neighIdx) - pos.row(neighIdx2)).norm();
                            if (dist2 < radius) {
                                visited[neighIdx2] = true;
                                if (dist2 < D(neighIdx2)) {
                                    D(neighIdx2) = dist2;
                                    nearestSourceK[neighIdx2] = sampleIdx;
                                }
                            }
                        }
                    }
                }
                ++sampleIdx;
            }
        }
        return selection;
    }

}