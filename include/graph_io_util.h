#ifndef GRAPH_IO_UTIL_H_
#define GRAPH_IO_UTIL_H_

#include <cerrno>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include "utils/thread_pool.h"
#include "graph.h"

namespace GraphGASLite {

namespace GraphIOUtil {

// Get the next line until a non-commented, non-empty line.
inline static std::istream& nextEffectiveLine(std::istream& input, string& line) {
    do {
        std::getline(input, line);
    } while (input && (line.empty() || line.at(0) == '#'));
    return input;
}

/**
 * Read graph topology from edge list file (and partition file).
 *
 * @param tileCount             Number of graph tiles.
 * @param edgeListFileName      graph topology file in edge list format.
 * @param partitionFileName     Partition file name can be empty string, which means
 *                              the graph is not partitioned (only one tile).
 * @param defaultWeight         The default edge weight value, used when no weight
 *                              is given in the edge list file.
 * @param tileMergeFactor       The factor for tile merge. The actual tile index of
 *                              a vertex will be the index in the partition file
 *                              divided by this factor.
 * @param vertexArgs            Used by vertex constructor.
 *
 * @return                      graph tiles.
 */
template<typename GraphTileType, typename... Args>
std::vector< Ptr<GraphTileType> > graphTilesFromEdgeList(const size_t tileCount, const size_t tileIndex,
        const string& edgeListFileName, const string& partitionFileName,
        const typename GraphTileType::EdgeType::WeightType& defaultWeight,
        const bool undirected, const size_t tileMergeFactor, const bool finalize, std::unordered_map< VertexIdx, TileIdx, std::hash<VertexIdx::Type> >& tidMap,
        Args&&... vertexArgs) {

    try{
        std::vector< Ptr<GraphTileType> > tiles(tileCount);
        for (size_t tid = 0; tid < tileCount; tid++) {
            tiles[tid].reset(new GraphTileType(tid));
        }

        bool partitioned = (tileCount != 1);

        // Read vertices and their partitioned tile number, build the map.
        // The map ought to be public
        tidMap.clear();
        if (partitioned) {
            if (partitionFileName.empty()) {
                throw FileException(partitionFileName);
            }
            std::ifstream infile(partitionFileName, std::ifstream::in);
            if (!infile.is_open()) {
                throw FileException(partitionFileName);
            }
            string line;
            while (nextEffectiveLine(infile, line)) {
                std::istringstream iss(line);
                uint64_t vid = 0;
                uint32_t tid = 0;
                // Line format: <vid> <tid>
                if (!(iss >> vid >> tid)) {
                    throw FileException(partitionFileName);
                }
                // Merge tiles.
                tid /= tileMergeFactor;
                if (tid >= tileCount) {
                    throw RangeException(std::to_string(tid));
                }

                if (tidMap.emplace(vid, tid).second == false) {
                    throw KeyInUseException(std::to_string(vid));
                }
                // Add the vertex.
                tiles[tid]->vertexNew(vid, std::forward<Args>(vertexArgs)...);
            }
        }

        auto vertexTileIdx = [&tidMap, partitioned](const VertexIdx& vid) {
            // decltype(tidMap)::const_iterator it;
            std::remove_reference<decltype(tidMap)>::type::const_iterator it;
            if (partitioned && (it = tidMap.find(vid)) != tidMap.end()) {
                return it->second;
            } else {
                return TileIdx(0);
            }
        };

        // Read edge list file, build the graph tiles.
        if (edgeListFileName.empty()) {
            throw FileException(edgeListFileName);
        }
        std::ifstream infile(edgeListFileName, std::ifstream::in);
        if (!infile.is_open()) {
            throw FileException(edgeListFileName);
        }
        string line;

        // Store edge info while reading file, then use multiple load threads to build tiles.
        struct EdgeInfo {
            VertexIdx srcId;
            VertexIdx dstId;
            typename GraphTileType::EdgeType::WeightType weight;
            TileIdx srcTid;
            TileIdx dstTid;
        };
        // Graph tiles for thread i will be loaded by load thread i % loadThreadCount.
        constexpr uint32_t loadThreadCount = 8;
        std::array<std::vector<EdgeInfo>, loadThreadCount> edgeInfoArray;

        while (nextEffectiveLine(infile, line)) {
            // Line format: <srcId> <dstId> [weight]

            // A faster way to convert string to numbers than using operator>>.
            const char* pbegin = line.c_str();
            char* pend = nullptr;
            uint64_t srcId = strtoull(pbegin, &pend, 10);
            if (pend == pbegin || errno == ERANGE) {
                // No conversion or out of range.
                throw FileException(edgeListFileName);
            }
            pbegin = pend;
            uint64_t dstId = strtoull(pbegin, &pend, 10);
            if (pend == pbegin || errno == ERANGE) {
                // No conversion or out of range.
                throw FileException(edgeListFileName);
            }
            pbegin = pend;

            typename GraphTileType::EdgeType::WeightType weight = defaultWeight;
            line = line.substr(pbegin - line.c_str());
            if (line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
                std::istringstream iss(line);
                if (!(iss >> weight)) {
                    throw FileException(edgeListFileName);
                }
            }

            // Get corresponding tile and add vertex if hasn't been done.
            const auto srcTid = vertexTileIdx(srcId);
            const auto dstTid = vertexTileIdx(dstId);
            if (!partitioned && !tiles[srcTid]->vertex(srcId)) {
                tiles[srcTid]->vertexNew(srcId, std::forward<Args>(vertexArgs)...);
            }
            if (!partitioned && !tiles[dstTid]->vertex(dstId)) {
                tiles[dstTid]->vertexNew(dstId, std::forward<Args>(vertexArgs)...);
            }

            // Store edge info.
            edgeInfoArray[srcTid % loadThreadCount].push_back(EdgeInfo{srcId, dstId, weight, srcTid, dstTid});
            if (undirected) {
                edgeInfoArray[dstTid % loadThreadCount].push_back(EdgeInfo{dstId, srcId, weight, dstTid, srcTid});
            }
        }

        ThreadPool loadPool(loadThreadCount);
        auto loadFunc = [&edgeInfoArray, &tiles, &tileIndex](uint32_t idx) {
            for (const auto& e : edgeInfoArray[idx]) {
                // Add edge.
                if (e.srcTid == tileIndex) {
                    tiles[e.srcTid]->edgeNew(e.srcId, e.dstId, e.dstTid, e.weight);
                    if (e.srcTid != e.dstTid) tiles[e.srcTid]->vertex(e.srcId)->setIsBorderVertex(true);
                } else if (e.dstTid == tileIndex) {
                    tiles[e.dstTid]->vertex(e.dstId)->inDegInc();
                }
            }
        };
        for (uint32_t idx = 0; idx < loadThreadCount; idx++) {
            loadPool.add_task(std::bind(loadFunc, idx));
        }
        loadPool.wait_all();

        if (finalize) {
            // Finalize each tile.
            for (auto& t : tiles) {
                // Propagate mirror vertex degree to master tile.
                for (auto mvIter = t->mirrorVertexIter(); mvIter != t->mirrorVertexIterEnd(); ++mvIter) {
                    auto& mv = mvIter->second;
                    auto vid = mv->vid();
                    auto masterTileId = mv->masterTileId();
                    tiles[masterTileId]->vertex(vid)->inDegInc(mv->accDeg());
                    mv->accDegDel();
                }
                t->finalizedIs(true);
            }
        } else {
            // Only sort edges.
            for (auto& t : tiles) {
                t->edgeSortedIs(true);
            }
        }

        return tiles;

    } catch (...) {
        throw FileException("Invalid format in graph topology input files.");
    }
}

} // namespace GraphIOUtil

} // namespace GraphGASLite

#endif // GRAPH_UTIL_H_