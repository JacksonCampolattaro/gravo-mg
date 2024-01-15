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

    bool operator>(const VertexPair&ref) const { return distance > ref.distance; }

    bool operator<(const VertexPair&ref) const { return distance < ref.distance; }
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
    using EdgeMatrix = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic>;
    using Triangle = std::array<Index, 3>;

    static double inTriangle(
        const Eigen::RowVector3d&p, std::span<Index, 3> tri,
        const Eigen::RowVector3d&triNormal, const Eigen::MatrixXd&pos,
        Eigen::RowVector3d&bary, std::map<Index, float>&insideEdge
    );

    static std::vector<double> uniformWeights(const int&n_points);

    static std::vector<double> inverseDistanceWeights(
        const Eigen::MatrixXd&pos, const Eigen::RowVector3d&source, const std::span<Index>&edges
    );

    static void constructDijkstraWithCluster(
        const Eigen::MatrixXd&points, const std::vector<Index>&source,
        const EdgeMatrix&neigh, Eigen::VectorXd&D,
        std::vector<Index>&nearestSourceK
    );

    double averageEdgeLength(const Eigen::MatrixXd&pos, const Eigen::MatrixXi&neigh);

    Eigen::SparseMatrix<double> constructProlongation(
        Eigen::MatrixXd points,
        double ratio = 8.0, bool nested = false, int lowBound = 1000,
        Sampling samplingStrategy = FASTDISK, Weighting weightingScheme = BARYCENTRIC,
        bool verbose = false
    );

}

#endif // !MULTIGRIDSOLVER_H
