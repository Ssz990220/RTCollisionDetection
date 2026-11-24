// computeEdges: unique edges, loop edges and the watertightness check, exactly as the old
// RTCD::meshModel::loadOBJ (RTCD/Meshes/mesh.h: addTriangleEdges, the BFS over triangle adjacency and the final
// loop check). The control flow below mirrors the old code statement by statement; only the containers differ
// (they are used for membership tests only, so their iteration order never matters), and the old code's
// diagnostics / exceptions become `watertight = false`.

#include <rtcd/geometry.h>

#include <array>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rtcd {

    namespace {

        enum class EdgeType : int { Backward = -1, Undef = 0, Forward = 1, Bidirect = 2 };

        // Old hashIndexPair: Cantor pairing in 32-bit int arithmetic. It is used as the identity of an edge in the
        // old knownEdges set, so a collision would drop an edge; to stay identical for large indices the int
        // overflow of the old code is reproduced with explicit two's complement wrap-around (MSVC behaviour).
        // For x + y <= 46340 (every mesh in models/) no wrap-around occurs and the pairing is injective.
        int hashIndexPair(int x, int y) noexcept {
            const std::uint32_t s    = static_cast<std::uint32_t>(x) + static_cast<std::uint32_t>(y);
            const auto product       = static_cast<std::int32_t>(s * (s + 1u));
            const std::int32_t half  = product / 2;
            return static_cast<std::int32_t>(static_cast<std::uint32_t>(half) + static_cast<std::uint32_t>(y));
        }

        // Directed edge (a -> b) as a 64-bit key (old int2 in unordered_set<int2>; only membership is used).
        std::uint64_t directedKey(int a, int b) noexcept {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
        }
        // Undirected edge (min, max) (old edge_to_triangles key).
        std::uint64_t undirectedKey(int a, int b) noexcept {
            return a < b ? directedKey(a, b) : directedKey(b, a);
        }

        bool doesLoop(int forwardCount, int backwardCount) {
            return forwardCount == 3 || backwardCount == 3;
        }
        bool doesAgree(int forwardCount, int backwardCount) {
            return forwardCount == 2 || backwardCount == 2;
        }

        // Old findRuleBreaker. Its only call is for three assigned edges that do not form a loop; there one of the
        // returns below is always reached (forward < backward implies a FORWARD edge, forward > backward a
        // BACKWARD one, equal counts a non-bidirectional edge). The old function fell off its end (undefined
        // behaviour) otherwise; -1 is returned here and the caller then adds no edge.
        int findRuleBreaker(const std::array<EdgeType, 3>& edges, int forwardCount, int backwardCount) {
            if (forwardCount == backwardCount) {
                for (int i = 0; i < 3; i++) {
                    if (edges[i] != EdgeType::Bidirect) {
                        return i;
                    }
                }
            }
            if (forwardCount < backwardCount) {
                for (int i = 0; i < 3; i++) {
                    if (edges[i] == EdgeType::Forward) {
                        return i;
                    }
                }
            } else {
                for (int i = 0; i < 3; i++) {
                    if (edges[i] == EdgeType::Backward) {
                        return i;
                    }
                }
            }
            return -1;
        }

    } // namespace

    void computeEdges(TriangleMesh& mesh) {
        mesh.uniqueEdges.clear();
        mesh.loopEdges.clear();
        mesh.watertight = false;

        const int numTriangles = static_cast<int>(mesh.triangles.size());
        if (numTriangles == 0) {
            return; // an empty mesh has no edges and is reported as not watertight
        }
        auto tri             = [&](int t) {
            const auto& v = mesh.triangles[static_cast<std::size_t>(t)];
            return std::array<int, 3>{static_cast<int>(v[0]), static_cast<int>(v[1]), static_cast<int>(v[2])};
        };

        // ---- unique edges (old addTriangleEdges / addEdge) and edge -> triangles adjacency ----
        std::unordered_set<int> knownEdges;
        knownEdges.reserve(mesh.triangles.size() * 2);
        auto addEdge = [&](int idx0, int idx1) {
            if (knownEdges.contains(hashIndexPair(idx1, idx0)) || knownEdges.contains(hashIndexPair(idx0, idx1))) {
                return;
            }
            knownEdges.insert(hashIndexPair(idx1, idx0));
            mesh.uniqueEdges.push_back(Edge{static_cast<std::uint32_t>(idx0), static_cast<std::uint32_t>(idx1)});
        };

        // Old: unordered_map<int2, std::set<int>>; triangles are inserted in increasing index order, so a sorted
        // vector without duplicates is the same set, iterated in the same (ascending) order.
        std::unordered_map<std::uint64_t, std::vector<int>> edgeToTriangles;
        edgeToTriangles.reserve(mesh.triangles.size() * 2);
        auto addAdjacency = [&](int a, int b, int t) {
            std::vector<int>& list = edgeToTriangles[undirectedKey(a, b)];
            if (list.empty() || list.back() != t) {
                list.push_back(t);
            }
        };

        for (int t = 0; t < numTriangles; ++t) {
            const auto v = tri(t);
            addAdjacency(v[0], v[1], t);
            addAdjacency(v[1], v[2], t);
            addAdjacency(v[2], v[0], t);
            addEdge(v[0], v[1]);
            addEdge(v[1], v[2]);
            addEdge(v[2], v[0]);
        }

        // ---- consistently oriented edges: BFS over triangle adjacency (old loadOBJ) ----
        std::unordered_set<std::uint64_t> orderedEdges;
        orderedEdges.reserve(mesh.triangles.size() * 2);
        auto has    = [&](int a, int b) { return orderedEdges.contains(directedKey(a, b)); };
        auto insert = [&](int a, int b) { orderedEdges.insert(directedKey(a, b)); };

        std::vector<char> processed(static_cast<std::size_t>(numTriangles), 0);
        int firstUnprocessed = 0; // old: linear scan from 0 for the first unprocessed triangle (processed only grows)

        while (true) {
            while (firstUnprocessed < numTriangles && processed[static_cast<std::size_t>(firstUnprocessed)]) {
                ++firstUnprocessed;
            }
            if (firstUnprocessed == numTriangles) {
                break;
            }
            std::queue<int> openTriangles;
            openTriangles.push(firstUnprocessed);

            while (!openTriangles.empty()) {
                const int current = openTriangles.front();
                openTriangles.pop();
                if (processed[static_cast<std::size_t>(current)]) {
                    continue;
                }

                const auto t                         = tri(current);
                const std::array<int, 4> forwardLoop = {t[0], t[1], t[2], t[0]};
                std::array<EdgeType, 3> edges{};
                std::vector<int> undefinedIds;
                int forwardCount  = 0;
                int backwardCount = 0;
                int nOrderedEdges = 0;

                for (int i = 0; i < 3; ++i) {
                    const bool isForward  = has(forwardLoop[i], forwardLoop[i + 1]);
                    const bool isBackward = has(forwardLoop[i + 1], forwardLoop[i]);
                    if (isForward && isBackward) {
                        edges[i] = EdgeType::Bidirect;
                        forwardCount++;
                        backwardCount++;
                        nOrderedEdges++;
                    } else if (isForward) {
                        edges[i] = EdgeType::Forward;
                        forwardCount++;
                        nOrderedEdges++;
                    } else if (isBackward) {
                        edges[i] = EdgeType::Backward;
                        backwardCount++;
                        nOrderedEdges++;
                    } else {
                        edges[i] = EdgeType::Undef;
                        undefinedIds.push_back(i);
                    }
                }

                switch (nOrderedEdges) {
                case 3:
                    if (!doesLoop(forwardCount, backwardCount)) {
                        const int ruleBreaker = findRuleBreaker(edges, forwardCount, backwardCount);
                        if (ruleBreaker >= 0) {
                            if (edges[ruleBreaker] == EdgeType::Forward) {
                                insert(forwardLoop[ruleBreaker + 1], forwardLoop[ruleBreaker]);
                            } else {
                                insert(forwardLoop[ruleBreaker], forwardLoop[ruleBreaker + 1]);
                            }
                        }
                    }
                    break;

                case 2:
                    {
                        const int edgeId = undefinedIds[0];
                        // The old code left loopDirc uninitialized when no assigned non-bidirectional edge exists
                        // (two bidirectional edges); Undef takes the "not FORWARD" branch below.
                        EdgeType loopDirc = EdgeType::Undef;
                        if (!doesAgree(forwardCount, backwardCount)) { // one forward and one backward
                            for (int i = 0; i < 3; ++i) {
                                if (edges[i] != EdgeType::Undef && edges[i] != EdgeType::Bidirect) {
                                    const bool isEdgeForward = edges[i] == EdgeType::Forward;
                                    if (isEdgeForward) {
                                        insert(forwardLoop[i + 1], forwardLoop[i]);
                                    } else {
                                        insert(forwardLoop[i], forwardLoop[i + 1]);
                                    }
                                    loopDirc = isEdgeForward ? EdgeType::Backward : EdgeType::Forward;
                                    break;
                                }
                            }
                        } else { // both forward, both backward, or one bidirectional
                            for (int i = 0; i < 3; ++i) {
                                if (edges[i] != EdgeType::Bidirect && edges[i] != EdgeType::Undef) {
                                    loopDirc = edges[i];
                                    break;
                                }
                            }
                        }
                        if (loopDirc == EdgeType::Forward) {
                            insert(forwardLoop[edgeId], forwardLoop[edgeId + 1]);
                        } else {
                            insert(forwardLoop[edgeId + 1], forwardLoop[edgeId]);
                        }
                    }
                    break;

                case 1:
                    {
                        int assignedEdgeId = 0;
                        for (int i = 0; i < 3; ++i) {
                            if (edges[i] != EdgeType::Undef) {
                                assignedEdgeId = i;
                                break;
                            }
                        }
                        const EdgeType loopDirc = edges[assignedEdgeId];
                        for (const int i : undefinedIds) {
                            if (loopDirc == EdgeType::Forward) {
                                insert(forwardLoop[i], forwardLoop[i + 1]);
                            } else {
                                insert(forwardLoop[i + 1], forwardLoop[i]);
                            }
                        }
                    }
                    break;

                case 0:
                    for (int i = 0; i < 3; ++i) {
                        insert(forwardLoop[i], forwardLoop[i + 1]);
                    }
                    break;

                default:
                    break;
                }

                processed[static_cast<std::size_t>(current)] = 1;

                for (int i = 0; i < 3; ++i) {
                    const auto it = edgeToTriangles.find(undirectedKey(forwardLoop[i], forwardLoop[i + 1]));
                    if (it == edgeToTriangles.end()) {
                        continue;
                    }
                    for (const int other : it->second) {
                        if (other != current && !processed[static_cast<std::size_t>(other)]) {
                            openTriangles.push(other);
                        }
                    }
                }
            }
        }

        // ---- watertightness: every triangle must be a loop of ordered edges (old: throw otherwise) ----
        for (int i = 0; i < numTriangles; ++i) {
            const auto t                         = tri(i);
            const std::array<int, 4> forwardLoop = {t[0], t[1], t[2], t[0]};
            int forwardCount                     = 0;
            int backwardCount                    = 0;
            for (int j = 0; j < 3; ++j) {
                const bool isForward  = has(forwardLoop[j], forwardLoop[j + 1]);
                const bool isBackward = has(forwardLoop[j + 1], forwardLoop[j]);
                if (isForward && isBackward) {
                    forwardCount++;
                    backwardCount++;
                } else if (isForward) {
                    forwardCount++;
                } else if (isBackward) {
                    backwardCount++;
                } else {
                    return; // not watertight: an edge without orientation
                }
            }
            if (!doesLoop(forwardCount, backwardCount)) {
                return; // not watertight
            }
        }

        // ---- loop edges in triangle order (old: each ordered edge emitted once, at its first triangle) ----
        for (int i = 0; i < numTriangles; ++i) {
            const auto t                         = tri(i);
            const std::array<int, 4> forwardLoop = {t[0], t[1], t[2], t[0]};
            for (int j = 0; j < 3; ++j) {
                const int a = forwardLoop[j];
                const int b = forwardLoop[j + 1];
                if (orderedEdges.erase(directedKey(a, b)) != 0) {
                    mesh.loopEdges.push_back(Edge{static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)});
                }
                if (orderedEdges.erase(directedKey(b, a)) != 0) {
                    mesh.loopEdges.push_back(Edge{static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(a)});
                }
            }
        }
        mesh.watertight = true;
    }

} // namespace rtcd
