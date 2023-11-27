#define _USE_MATH_DEFINES

#include "gravomg/multigrid.h"
#include "gravomg/utility.h"
#include "gravomg/sampling.h"

#include <cmath>
#include <numeric>
#include <chrono>

#include <Eigen/Eigenvalues>
#include <utility>

namespace GravoMG {

    Eigen::SparseMatrix<double> constructProlongation(
            Eigen::MatrixXd points,
            double ratio, bool nested, int lowBound,
            Sampling samplingStrategy, Weighting weightingScheme,
            bool verbose
    ) {
        // todo
        return {};
    }


    std::vector<Eigen::SparseMatrix<double>> constructProlongations(
            Eigen::MatrixXd points,
            double ratio, bool nested, int lowBound,
            Sampling samplingStrategy, Weighting weightingScheme,
            bool verbose
    ) {
        // Prolongation operators
        std::vector<Eigen::SparseMatrix<double>> U;

        // Degrees of freedom per level
        std::vector<size_t> DoF;

        // Sampled points (per level)
        std::vector<std::vector<int>> samples;

        // Nearest source for every given point (per level)
        std::vector<std::vector<size_t>> nearestSource;

        // Create prolongation operator for level k+1 to k
        // Points in current level
        Eigen::MatrixXd levelPoints = std::move(points);
        // Neighborhood data structure
        Eigen::MatrixXi neighLevelK;

        // Compute initial radius
        double densityRatio = std::sqrt(ratio);
        int nLevel1 = int(double(levelPoints.rows()) / ratio);

        std::random_device rd;
        std::default_random_engine generator(rd());

        // For each level
        int k = 0;
        DoF.clear();
        DoF.shrink_to_fit();
        DoF.push_back(levelPoints.rows());
        while (levelPoints.rows() > lowBound && k < 10) {
            double radius = std::cbrt(ratio) * computeAverageEdgeLength(levelPoints, neighLevelK);

            // Data structure for neighbors inside level k
            std::vector<std::set<int>> neighborsLists;

            // List of triplets to build prolongation operator U
            std::vector<Eigen::Triplet<double>> AllTriplet, UNeighAllTriplet;

            // The nearest coarse point for each fine point and the distances computed
            Eigen::VectorXd D(levelPoints.rows());
            D.setConstant(std::numeric_limits<double>::max());
            nearestSource.emplace_back(levelPoints.rows());

            // -- Sample a subset for level k + 1
            if (verbose) printf("__Constructing Prolongation Operator for level = %d using closest triangle. \n", k);

            // Sample points that will be part of the coarser level with Poisson disk sampling
            if (verbose) std::cout << "Obtaining subset from the finer level\n";


            switch (samplingStrategy) {
                case FASTDISK:
                    samples.push_back(fastDiskSample(levelPoints, neighLevelK, radius, D, nearestSource[k]));
                    DoF.push_back(samples[k].size());
                    break;
                case RANDOM:
                    DoF.push_back(DoF[k] / ratio);
                    samples.push_back(std::vector<int>(DoF[k]));
                    std::iota(samples[k].begin(), samples[k].end(), 0);
                    std::shuffle(samples[k].begin(), samples[k].end(), generator);
                    samples[k].resize(DoF[k + 1]);
                    break;
                case MIS:
                    samples.push_back(maximumDeltaIndependentSetWithDistances(
                            levelPoints, neighLevelK,
                            radius, D, nearestSource[k]
                    ));
                    DoF.push_back(samples[k].size());
                    break;
            }

            if (samples[k].size() < lowBound) {
                nearestSource.pop_back();
                break;
            }

            if (verbose) cout << "Actual number found: " << samples[k].size() << endl;
            DoF[k + 1] = samples[k].size();

            // Compute distance from fine points to coarse points and get the closest coarse point
            // using distances from MIS if computed before
            constructDijkstraWithCluster(levelPoints, samples[k], neighLevelK, k, D,
                                         nearestSource[k]); // Stores result in nearestSource[k]

            // Create neighborhood for the next level
            neighborsLists.resize(DoF[k + 1]);
            for (int fineIdx = 0; fineIdx < DoF[k]; ++fineIdx) {
                for (int j = 0; j < neighLevelK.cols(); ++j) {
                    int neighIdx = neighLevelK(fineIdx, j);
                    if (neighIdx < 0) break;
                    if (nearestSource[k][fineIdx] != nearestSource[k][neighIdx]) {
                        neighborsLists[nearestSource[k][fineIdx]].insert(nearestSource[k][neighIdx]);
                    }
                }
            }

            // Store in homogeneous data structure
            std::size_t maxNeighNum = 0;
            for (const auto &neighbors: neighborsLists) {
                if (neighbors.size() > maxNeighNum) {
                    maxNeighNum = neighbors.size();
                }
            }
            neighLevelK.resize(DoF[k + 1], maxNeighNum);
            neighLevelK.setConstant(-1);
            for (int i = 0; i < neighborsLists.size(); ++i) {
                neighLevelK(i, 0) = i;
                int iCounter = 1;
                for (int node: neighborsLists[i]) {
                    if (node == i) continue;
                    if (iCounter >= maxNeighNum) break;
                    neighLevelK(i, iCounter) = node;
                    iCounter++;
                }
            }

            if (verbose) std::cout << "Setting up the point locations for the next level\n";

            // Setting up the DoF for the next level
            // tempPoints are the centers of the voronoi cells, each row for each voronoi cells
            Eigen::MatrixXd tempPoints(DoF[k + 1], levelPoints.cols());
            tempPoints.setZero();
            if (nested) {
                for (int coarseIdx = 0; coarseIdx < DoF[k + 1]; ++coarseIdx) {
                    tempPoints.row(coarseIdx) = levelPoints.row(samples[k][coarseIdx]);
                }
            } else {
                std::vector<int> clusterSizes(DoF[k + 1]);
                for (int fineIdx = 0; fineIdx < DoF[k]; ++fineIdx) {
                    int coarseIdx = nearestSource[k][fineIdx];
                    tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) + levelPoints.row(fineIdx);
                    ++clusterSizes[coarseIdx];
                }
                for (int coarseIdx = 0; coarseIdx < DoF[k + 1]; ++coarseIdx) {
                    if (clusterSizes[coarseIdx] == 1) {
                        tempPoints.row(coarseIdx) = levelPoints.row(samples[k][coarseIdx]);
                        for (int neighIdx: neighborsLists[coarseIdx]) {
                            tempPoints.row(coarseIdx) =
                                    tempPoints.row(coarseIdx) + levelPoints.row(samples[k][neighIdx]);
                        }
                        tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) / (neighborsLists[coarseIdx].size() + 1.);
                    } else {
                        tempPoints.row(coarseIdx) = tempPoints.row(coarseIdx) / clusterSizes[coarseIdx];
                    }
                }
            }


            // Create triangles for this level based on Voronoi cells
            std::vector<std::vector<int>> tris;
            tris.reserve(DoF[k + 1] * maxNeighNum);
            std::vector<std::vector<int>> connectedTris(DoF[k + 1]);
            std::vector<Eigen::RowVector3d> triNormals;
            triNormals.reserve(DoF[k + 1] * maxNeighNum);
            int triIdx = 0;
            for (int coarseIdx = 0; coarseIdx < DoF[k + 1]; ++coarseIdx) {
                // Iterate over delaunay triangles
                int v2Idx, v3Idx;
                for (auto it = neighborsLists[coarseIdx].begin(); it != neighborsLists[coarseIdx].end(); it++) {
                    v2Idx = *it;
                    // We iterate over the coarse indices in order,
                    // so if the neighboring idx is lower then the current coarseIdx,
                    // it must have been considered before and be part of a triangle.
                    if (v2Idx < coarseIdx) continue;
                    for (auto it2 = std::next(it); it2 != neighborsLists[coarseIdx].end(); it2++) {
                        v3Idx = *it2;
                        if (v3Idx < coarseIdx) continue;
                        if (neighborsLists[v2Idx].find(v3Idx) != neighborsLists[v2Idx].end()) {
                            tris.push_back({coarseIdx, v2Idx, v3Idx});
                            Eigen::RowVector3d e12 = tempPoints.row(v2Idx) - tempPoints.row(coarseIdx);
                            Eigen::RowVector3d e13 = tempPoints.row(v3Idx) - tempPoints.row(coarseIdx);
                            triNormals.push_back(e12.cross(e13).normalized());
                            connectedTris[coarseIdx].push_back(triIdx);
                            connectedTris[v2Idx].push_back(triIdx);
                            connectedTris[v3Idx].push_back(triIdx);
                            ++triIdx;
                        }
                    }
                }
            }
            tris.shrink_to_fit();
            triNormals.shrink_to_fit();


            // Create local triangulation on each cluster (centralized at sample i)
            int notrisfound = 0;
            int edgesfound = 0;
            int fallbackCount = 0;

            for (int fineIdx = 0; fineIdx < DoF[k]; ++fineIdx) {
                Eigen::RowVector3d finePoint = levelPoints.row(fineIdx);
                int coarseIdx = nearestSource[k][fineIdx];
                Eigen::RowVector3d coarsePoint = tempPoints.row(coarseIdx);
                std::vector<double> weights;

                if (nested && samples[k][nearestSource[k][fineIdx]] == fineIdx) {
                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, 1.));
                    continue;
                }

                if (neighborsLists[coarseIdx].empty()) {
                    // If the coarse point has no neighbors,
                    // set the weight to 1 for the coarse point.
                    // Note: this should not happen.
                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, 1.));
                } else if (neighborsLists[coarseIdx].size() == 1) {
                    // If the coarse point only has one neighbor, no triangle can be created.
                    // Thus, the weights are distributed w.r.t. the distance to each coarse point.
                    int neighIdx = *neighborsLists[coarseIdx].begin();
                    Eigen::RowVector3d neighPoint = tempPoints.row(neighIdx);

                    // get the distance to the two neighboring centroids
                    Eigen::RowVector3d e12 = neighPoint - coarsePoint;
                    double e12Length = max(e12.norm(), 1e-8);
                    double w2 = (levelPoints.row(fineIdx) - coarsePoint).dot(e12.normalized()) / e12Length;
                    w2 = min(max(w2, 0.), 1.);
                    double w1 = 1 - w2;

                    switch (weightingScheme) {
                        case BARYCENTRIC:
                            AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, w1));
                            AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, w2));
                            break;
                        case UNIFORM:
                            weights = uniformWeights(2);
                            AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
                            AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, weights[1]));
                            break;
                        case INVDIST:
                            std::vector<int> endPoints = {coarseIdx, neighIdx};
                            weights = inverseDistanceWeights(tempPoints, finePoint, endPoints);
                            AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
                            AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, weights[1]));
                            break;
                    }
                } else {
                    // Only keep triangle with minimum distance
                    double minDistToTriangle = std::numeric_limits<double>::max();
                    Eigen::RowVector3d minBary = {0., 0., 0.};
                    std::vector<int> minTri;
                    bool triFound = false;

                    // Values are positive if inside and negative if not
                    // Float value represents distance
                    std::map<int, float> insideEdge;

                    // Iterate over all triangles
                    for (int triIdx: connectedTris[coarseIdx]) {
                        std::vector<int> tri = tris[triIdx];
                        // Make sure that the coarseIdx is in position 0, while keeping orientation
                        while (tri[0] != coarseIdx) std::rotate(tri.begin(), tri.begin() + 1, tri.end());

                        Eigen::RowVector3d bary = {0., 0., 0.};
                        // If the triangle contains the point, the distance is positive, else it's negative
                        double distToTriangle = inTriangle(finePoint, tri, triNormals[triIdx], tempPoints, bary,
                                                           insideEdge);
                        if (distToTriangle >= 0. && distToTriangle < minDistToTriangle) {
                            triFound = true;
                            minDistToTriangle = distToTriangle;
                            minTri = tri;
                            minBary = bary;
                            break;
                        }
                    }

                    if (triFound) {
                        switch (weightingScheme) {
                            case BARYCENTRIC:
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[0], minBary(0)));
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[1], minBary(1)));
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[2], minBary(2)));
                                break;
                            case UNIFORM:
                                weights = uniformWeights(3);
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[0], weights[0]));
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[1], weights[1]));
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[2], weights[2]));
                                break;
                            case INVDIST:
                                weights = inverseDistanceWeights(tempPoints, finePoint, minTri);
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[0], weights[0]));
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[1], weights[1]));
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minTri[2], weights[2]));
                                break;
                        }
                    } else {
                        bool edgeFound = false;
                        double minEdge = std::numeric_limits<double>::max();
                        int minEdgeIdx = 0;
                        for (const auto &element: insideEdge) {
                            const auto &key = element.first;
                            const auto &value = element.second;
                            if (value >= 0. && value < minEdge) {
                                edgeFound = true;
                                minEdge = value;
                                minEdgeIdx = key;
                                break;
                            }
                        }
                        if (edgeFound) {
                            ++edgesfound;
                            Eigen::RowVector3d p2 = tempPoints.row(minEdgeIdx);
                            Eigen::RowVector3d e12 = p2 - coarsePoint;
                            double e12Length = max(e12.norm(), 1e-8);
                            double w2 = (finePoint - coarsePoint).dot(e12.normalized()) / e12Length;
                            w2 = min(max(w2, 0.), 1.);
                            double w1 = 1. - w2;

                            switch (weightingScheme) {
                                case BARYCENTRIC:
                                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, w1));
                                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minEdgeIdx, w2));
                                    break;
                                case UNIFORM:
                                    weights = uniformWeights(2);
                                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
                                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minEdgeIdx, weights[1]));
                                    break;
                                case INVDIST:
                                    std::vector<int> endPointsEdge = {coarseIdx, minEdgeIdx};
                                    weights = inverseDistanceWeights(tempPoints, finePoint, endPointsEdge);
                                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, weights[0]));
                                    AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, minEdgeIdx, weights[1]));
                                    break;
                            }
                        } else {
                            // Use closest three
                            std::vector<int> prolongFrom(3);
                            prolongFrom[0] = coarseIdx;

                            std::vector<VertexPair> pointsDistances;
                            for (int j = 0; j < neighLevelK.cols(); ++j) {
                                int neighIdx = neighLevelK(coarseIdx, j);
                                if (neighIdx < 0 || neighIdx == coarseIdx) continue;
                                VertexPair vp = {neighIdx, (finePoint - tempPoints.row(neighIdx)).norm()};
                                pointsDistances.push_back(vp);
                            }
                            std::sort(pointsDistances.begin(), pointsDistances.end(), std::less<VertexPair>());
                            for (int j = 1; j < 3; ++j) {
                                prolongFrom[j] = pointsDistances[j - 1].vId;
                            }
                            std::vector<double> weights = inverseDistanceWeights(tempPoints, finePoint, prolongFrom);
                            for (int j = 0; j < prolongFrom.size(); j++) {
                                AllTriplet.push_back(Eigen::Triplet<double>(fineIdx, prolongFrom[j], weights[j]));
                            }
                            ++fallbackCount;
                        }
                        ++notrisfound;
                    }
                }
            }
            if (verbose) cout << "Percentage of fallback: " << (double) fallbackCount / (double) DoF[k] * 100 << endl;

            levelPoints = tempPoints;

            Eigen::SparseMatrix<double> ULevel;
            ULevel.resize(DoF[k], DoF[k + 1]);
            ULevel.setFromTriplets(AllTriplet.begin(), AllTriplet.end());
            U.push_back(ULevel);
            AllTriplet.clear();
            AllTriplet.shrink_to_fit();
            ++k;
        }

        return U;
    }

    double inTriangle(const Eigen::RowVector3d &p, const std::vector<int> &tri,
                      const Eigen::RowVector3d &triNormal, const Eigen::MatrixXd &pos,
                      Eigen::RowVector3d &bary, std::map<int, float> &insideEdge) {
        Eigen::RowVector3d v1, v2, v3;
        v1 = pos.row(tri[0]);
        v2 = pos.row(tri[1]);
        v3 = pos.row(tri[2]);
        Eigen::RowVector3d v1ToP = p - v1;
        Eigen::RowVector3d e12 = v2 - v1;
        Eigen::RowVector3d e13 = v3 - v1;

        double distToTriangle = (p - v1).dot(triNormal);
        Eigen::RowVector3d pProjected = p - distToTriangle * triNormal;

        double doubleArea = (v2 - v1).cross(v3 - v1).dot(triNormal);
        bary(0) = (v3 - v2).cross(pProjected - v2).dot(triNormal) / doubleArea;
        bary(1) = (v1 - v3).cross(pProjected - v3).dot(triNormal) / doubleArea;
        bary(2) = 1. - bary(0) - bary(1);

        if (insideEdge.find(tri[1]) == insideEdge.end()) {
            insideEdge[tri[1]] = ((v1ToP) - ((v1ToP).dot(e12) * (e12))).norm();
        }
        if (insideEdge.find(tri[2]) == insideEdge.end()) {
            insideEdge[tri[2]] = ((v1ToP) - ((v1ToP).dot(e13) * (e13))).norm();
        }
        if (bary(0) < 0. || bary(1) < 0.) {
            insideEdge[tri[1]] = -1.;
        }
        if (bary(0) < 0. || bary(2) < 0.) {
            insideEdge[tri[2]] = -1.;
        }

        if (bary(0) >= 0. && bary(1) >= 0. && bary(2) >= 0.) {
            return abs(distToTriangle);
        }

        return -1.;
    }

    std::vector<double> uniformWeights(const int &n_points) {
        std::vector<double> weights(n_points);
        std::fill(weights.begin(), weights.end(), 1. / n_points);
        return weights;
    }

    std::vector<double> inverseDistanceWeights(const Eigen::MatrixXd &pos, const Eigen::RowVector3d &p,
                                               const std::vector<int> &edges) {
        double sumWeight = 0.;
        std::vector<double> weights(edges.size());
        for (int j = 0; j < edges.size(); ++j) {
            weights[j] = 1. / max(1e-8, (p - pos.row(edges[j])).norm());
            sumWeight += weights[j];
        }
        for (int j = 0; j < weights.size(); ++j) {
            weights[j] = weights[j] / sumWeight;
        }
        return weights;
    }

    double computeAverageEdgeLength(const Eigen::MatrixXd &pos, const Eigen::MatrixXi &neigh) {
        double sumLength = 0;
        int nEdges = 0;
        for (int i = 0; i < pos.rows(); ++i) {
            Eigen::Vector3d p1 = pos.row(i);
            for (int j = 0; j < neigh.cols(); ++j) {
                if (neigh(i, j) < 0) continue;
                Eigen::Vector3d p2 = pos.row(neigh(i, j));
                double dist = (p1 - p2).norm();
                if (dist > 0) {
                    sumLength += dist;
                    ++nEdges;
                }
            }
        }
        return sumLength / (double) nEdges;
    }

    void constructDijkstraWithCluster(const Eigen::MatrixXd &points, const std::vector<int> &source,
                                      const Eigen::MatrixXi &neigh, int k, Eigen::VectorXd &D,
                                      std::vector<size_t> &nearestSourceK) {
        std::priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistanceQueue;
        if (nearestSourceK.empty()) nearestSourceK.resize(points.rows(), source[0]);

        for (int i = 0; i < source.size(); ++i) {
            D(source[i]) = 0.0;
            VertexPair vp{source[i], D(source[i])};
            DistanceQueue.push(vp);
            nearestSourceK[source[i]] = i;
        }

        int curSource;
        while (!DistanceQueue.empty()) {
            VertexPair vp1 = DistanceQueue.top();
            curSource = nearestSourceK[vp1.vId];
            Eigen::RowVector3d vertex1 = points.row(vp1.vId);
            DistanceQueue.pop();

            //for (int vNeigh : neigh.row(vp1.vId)) {
            for (int i = 0; i < neigh.cols(); ++i) {
                int vNeigh = neigh(vp1.vId, i);

                if (vNeigh >= 0) {
                    double dist, distTemp;
                    Eigen::RowVector3d vertex2 = points.row(vNeigh);
                    dist = (vertex2 - vertex1).norm();
                    distTemp = vp1.distance + dist;
                    if (distTemp < D(vNeigh)) {
                        // Assign a new distance
                        D(vNeigh) = distTemp;
                        VertexPair v2{vNeigh, distTemp};
                        DistanceQueue.push(v2);


                        // Assign the nearest source to a certain point
                        nearestSourceK[vNeigh] = curSource;
                    }
                }
            }
        }
    }


}

