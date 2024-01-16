#ifndef MULTIGRIDSOLVER_H
#define MULTIGRIDSOLVER_H

#include <queue>
#include <vector>
#include <span>

#include "utility.h"

/* Data structure for Priority Queue */
struct VertexPair {

    Eigen::Index vId;
    double distance;

    bool operator>(const VertexPair& ref) const { return distance > ref.distance; }

    bool operator<(const VertexPair& ref) const { return distance < ref.distance; }
};

/* Enum to set solver type */
enum Sampling {
    FASTDISK = 0,
    RANDOM = 1,
    MIS = 2
};

enum Weighting {
    BARYCENTRIC = 0,
    UNIFORM = 1,
    INVDIST = 2
};

namespace GravoMG {

    using Eigen::Index;
    using PointMatrix = Eigen::MatrixXd;
    using EdgeMatrix = Eigen::MatrixXi;
    using Triangle = std::array<Index, 3>;

    double inTriangle(
        const Eigen::RowVector3d& p, std::span<Index, 3> tri,
        const Eigen::RowVector3d& triNormal, const Eigen::MatrixXd& pos,
        Eigen::RowVector3d& bary, std::map<Index, float>& insideEdge
    );

    std::vector<double> uniformWeights(const int& n_points);

    std::vector<double> inverseDistanceWeights(
        const Eigen::MatrixXd& pos, const Eigen::RowVector3d& source, const std::span<Index>& edges
    );

    void constructDijkstraWithCluster(
        const Eigen::MatrixXd& points, const std::vector<Index>& source,
        const EdgeMatrix& neigh, Eigen::VectorXd& D,
        std::vector<Index>& nearestSourceK
    );

    double averageEdgeLength(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& neigh);

    std::vector<std::set<Index>> extractCoarseEdges(
        const Eigen::MatrixXd& fine_points,
        const EdgeMatrix& fine_edges,
        const std::vector<Index>& coarse_samples,
        const std::vector<Index>& fine_to_nearest_coarse
    );

    EdgeMatrix toPaddedEdgeMatrix(const std::vector<std::set<Index>>& edges);

    PointMatrix coarseFromMeanOfFineChildren(
        const PointMatrix& fine_points,
        const EdgeMatrix& fine_edges,
        const std::vector<Index>& fine_to_nearest_coarse,
        std::size_t num_coarse_points
    );

    std::vector<Triangle> constructVoronoiTriangles(
        const Eigen::MatrixXd& points,
        const EdgeMatrix& edges
    );

    std::tuple<PointMatrix, EdgeMatrix, Eigen::SparseMatrix<double>> constructProlongation(
        const Eigen::MatrixXd& fine_points,
        const EdgeMatrix& fine_edges,
        const std::vector<Index>& coarse_samples,
        Weighting weighting_scheme,
        bool verbose = false, bool nested = true
    );

}

#endif // !MULTIGRIDSOLVER_H
