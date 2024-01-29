#ifndef ENGINE_H_
#define ENGINE_H_

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "utils/thread_pool.h"
#include "algo_kernel.h"
#include "comm_sync.h"
#include "graph.h"

#include <thread>
#include <vector>
#include <memory>

#include "Common/Defines.h"
#include "Network/IOService.h"

#include "Network/Endpoint.h"
#include "Network/Channel.h"

// #include "Common/ByteStream.h"

namespace GraphGASLite {

template<typename GraphTileType>
class Engine {
public:
    typedef BaseAlgoKernel<GraphTileType> AlgoKernelType;

    typedef std::vector< Ptr<GraphTileType> > GraphTileList;

    typedef std::vector< Ptr<const AlgoKernelType> > AlgoKernelList;
    typedef typename AlgoKernelList::iterator AlgoKernelIter;
    typedef typename AlgoKernelList::const_iterator AlgoKernelConstIter;

public:
    Ptr<GraphTileType> graphTile(const TileIdx& tid) const {
        if (tid >= graphs_.size()) return nullptr;
        return graphs_[tid];
    }
    size_t graphTileCount() const {
        return graphs_.size();
    }
    size_t graphTileIndex() const {
        return tileIndex_;
    }
    /**
     * Append a single graph tile \c graphTile.
     *
     * The appended tile must have incremental tile index.
     */
    void graphTileNew(const Ptr<GraphTileType>& graphTile) {
        if (graphTile == nullptr) {
            throw NullPointerException("graphTile");
        }
        // Graph tile index must be incremental.
        if (graphTile->tid() != graphs_.size()) {
            throw InvalidArgumentException("graphTile");
        }
        graphs_.push_back(graphTile);
    }
    /**
     * Copy-assign all graph tiles.
     *
     * Each tile in \c graphs must have incremental tile index.
     */
    void graphTileIs(const GraphTileList& graphs) {
        for (uint32_t idx = 0; idx < graphs.size(); idx++) {
            if (graphs[idx] == nullptr) {
                throw NullPointerException("graphs[" + std::to_string(idx) + "]");
            }
            // Test graph tile list is incremental.
            if (graphs[idx]->tid() != idx) {
                throw InvalidArgumentException("graphs[" + std::to_string(idx) + "]");
            }
        }
        graphs_ = graphs;
    }
    /**
     * Move-assign all graph tiles.
     *
     * Each tile in \c graphs must have incremental tile index.
     */
    void graphTileIs(GraphTileList&& graphs) {
        for (uint32_t idx = 0; idx < graphs.size(); idx++) {
            if (graphs[idx] == nullptr) {
                throw NullPointerException("graphs[" + std::to_string(idx) + "]");
            }
            // Test graph tile list is incremental.
            if (graphs[idx]->tid() != idx) {
                throw InvalidArgumentException("graphs[" + std::to_string(idx) + "]");
            }
        }
        graphs_.swap(graphs);
    }
    /**
     * Set current tile index
     * 
     */
    void tileIndexIs(size_t tileIndex) {
        tileIndex_ = tileIndex;
    }

    size_t algoKernelCount() const {
        return kernels_.size();
    }
    AlgoKernelIter algoKernelIter() {
        return kernels_.begin();
    }
    AlgoKernelIter algoKernelIterEnd() {
        return kernels_.end();
    }
    AlgoKernelConstIter algoKernelConstIter() const {
        return kernels_.cbegin();
    }
    AlgoKernelConstIter algoKernelConstIterEnd() const {
        return kernels_.cend();
    }
    /**
     * Append an algorithm kernel.
     */
    void algoKernelNew(const Ptr<AlgoKernelType>& kernel) {
        if (kernel == nullptr) {
            throw NullPointerException("kernel");
        }
        kernels_.push_back(kernel);
    }
    /**
     * Delete an algorithm kernel.
     */
    AlgoKernelIter algoKernelDel(const AlgoKernelIter& iter) {
        return kernels_.erase(iter);
    }

    /**
     * Run all algorithm kernels in sequence on the graph tiles.
     *
     * The engine is defined as a functor class.
     */
    void operator()() {
        // Number of worker threads.
        // Currently use one thread for each tile.
        auto threadCount = graphTileCount();
        auto tileIndex = graphTileIndex();

        // Utility for communication and synchronization.
        typedef CommSync<VertexIdx, typename GraphTileType::UpdateType> CommSyncType;
        CommSyncType cs(threadCount,
                typename CommSyncType::KeyValue(-1uL, typename GraphTileType::UpdateType()));

        auto nodeCount = graphTileCount();
        TaskComm& clientTaskComm = TaskComm::getClientInstance();
        bool isCluster = clientTaskComm.getIsCluster();
        uint32_t basePort = 1712;
        std::string ip;
        if (!isCluster) {
            ip = "127.0.0.1";
        } else {
            ip = "10.0.0.";
            ip += std::to_string(tileIndex + 1);
        }

        std::vector<std::vector<osuCrypto::Channel*>> channels(nodeCount);
        for (uint64_t i = 0; i < nodeCount; ++i)
            channels[i].resize(nodeCount);

        std::list<osuCrypto::Session> endpoints;
        osuCrypto::IOService ioService(0);

        printf("Set up channels\n");

        // Establish channels
        for (uint32_t j = 0; j < nodeCount; ++j) {
            if (j != tileIndex) {
                uint32_t port = 0;
                bool host = tileIndex > j;
                std::string name("endpoint:");
                if (host) {
                    name += std::to_string(tileIndex) + "->" + std::to_string(j);
                    port = basePort + (uint32_t)tileIndex;
                    printf("Not host\n");
                }
                else
                {
                    name += std::to_string(j) + "->" + std::to_string(tileIndex);
                    port = basePort + (uint32_t)j;
                }
                printf("Endpoints name %s\n", name.c_str());
                if (!isCluster || host) {
                    endpoints.emplace_back(ioService, ip, port, host?osuCrypto::SessionMode::Server:osuCrypto::SessionMode::Client, name);
                } else {
                    std::string remoteIp = "10.0.0.";
                    remoteIp += std::to_string(j + 1);
                    endpoints.emplace_back(ioService, remoteIp, port, host?osuCrypto::SessionMode::Server:osuCrypto::SessionMode::Client, name);             
                }
                channels[tileIndex][j] = new osuCrypto::Channel(endpoints.back().addChannel("chl", "chl"));
            }
        }

        printf("Finish setup channels\n");

        cs.setChannels(&channels);

        cs.threadIdIs(tileIndex);

        for (auto& k : kernels_) {
            (*k)(graphs_[tileIndex], cs);
        }

        for (auto chls : channels)
            for (auto chl : chls)
                if (chl)
                    chl->close();

        for (auto& ep : endpoints)
            ep.stop();

        ioService.stop();
    }

private:
    GraphTileList graphs_;
    AlgoKernelList kernels_;
    size_t tileIndex_;

};

} // namespace GraphGASLite

#endif // ENGINE_H_
