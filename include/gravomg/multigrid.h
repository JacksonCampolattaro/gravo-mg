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
        const NeighborMatrix& neigh,
        Eigen::VectorXd& D,
        std::vector<Index>& nearestSourceK
    );

    std::vector<Index> assignParents(
        const Eigen::MatrixXd& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& coarse_samples
    );

    double averageEdgeLength(const Eigen::MatrixXd& positions, const NeighborList& neighbors);

    NeighborList extractCoarseEdges(
        const PointMatrix& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& coarse_samples,
        const std::vector<Index>& fine_to_nearest_coarse
    );

    PointMatrix coarseFromMeanOfFineChildren(
        const PointMatrix& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& fine_to_nearest_coarse,
        std::size_t num_coarse_points
    );

    std::pair<std::vector<TriangleWithNormal>, std::vector<std::vector<size_t>>> constructVoronoiTriangles(
        const PointMatrix& points,
        const NeighborList& edges
    );

    Eigen::SparseMatrix<double> constructProlongation(
        const PointMatrix& fine_points,
        const PointMatrix& coarse_points,
        const NeighborList& coarse_edges,
        const std::vector<Index>& fine_to_nearest_coarse,
        Weighting weighting_scheme
    );

    std::tuple<PointMatrix, NeighborMatrix, Eigen::SparseMatrix<double>> constructProlongation(
        const PointMatrix& fine_points,
        const NeighborList& fine_neighbors,
        const std::vector<Index>& coarse_samples,
        Weighting weighting_scheme,
        bool verbose = false, bool nested = true
    );

}

#endif // !MULTIGRIDSOLVER_H
