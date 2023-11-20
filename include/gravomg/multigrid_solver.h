#ifndef MULTIGRIDSOLVER_H
#define MULTIGRIDSOLVER_H

#include <queue>
#include <vector>

#include "utility.h"

/* Data structure for Priority Queue */
struct VertexPair {

    int vId;
    double distance;

    bool operator>(const VertexPair &ref) const { return distance > ref.distance; }

    bool operator<(const VertexPair &ref) const { return distance < ref.distance; }
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


    static double computeAverageEdgeLength(const Eigen::MatrixXd &pos, const Eigen::MatrixXi &neigh);

    static double inTriangle(
            const Eigen::RowVector3d &p, const std::vector<int> &tri, const Eigen::RowVector3d &triNormal,
            const Eigen::MatrixXd &pos, Eigen::RowVector3d &bary, std::map<int, float> &insideEdge
    );

    static std::vector<double> uniformWeights(const int &n_points);

    static std::vector<double> inverseDistanceWeights(
            const Eigen::MatrixXd &pos, const Eigen::RowVector3d &source, const std::vector<int> &edges
    );

    static void constructDijkstraWithCluster(
            const Eigen::MatrixXd &points, const std::vector<int> &source,
            const Eigen::MatrixXi &neigh, int k, Eigen::VectorXd &D,
            std::vector<size_t> &nearestSourceK
    );

    class MultigridSolver {
    public:

        void constructProlongation(
                Eigen::MatrixXd points,
                double ratio = 8.0, bool nested = false, int lowBound = 1000,
                Sampling samplingStrategy = FASTDISK, Weighting weightingScheme = BARYCENTRIC,
                bool verbose = false
        );

        /* Hierarchy data*/
        std::vector<Eigen::SparseMatrix<double>> U;                                //!< Prolongation operators
        std::vector<std::vector<int>> samples;

    };


}

#endif // !MULTIGRIDSOLVER_H
