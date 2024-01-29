#ifndef ALGO_KERNEL_H_
#define ALGO_KERNEL_H_

#include <chrono>
#include <limits>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <thread>
#include "comm_sync.h"
#include "graph.h"
#include "task.h"

#include "TaskqHandler.h"

namespace GraphGASLite {

enum class AlgoKernelTag {
    EdgeCentric,
    VertexCentric,
};

static inline string algoKernelTagName(const AlgoKernelTag& tag) {
    switch(tag) {
        case AlgoKernelTag::EdgeCentric: return "edge-centric";
        case AlgoKernelTag::VertexCentric: return "vertex-centric";
        default: return "invalid";
    }
}

class IterCountRepType;
typedef CountType<uint64_t, IterCountRepType> IterCount;

static constexpr auto INF_ITER_COUNT = std::numeric_limits<typename IterCount::Type>::max();

template<typename GraphTileType>
class BaseAlgoKernel {
protected:
    typedef CommSync<VertexIdx, typename GraphTileType::UpdateType> CommSyncType;

public:
    /**
     * Algorithm kernel name.
     */
    string name() const { return name_; }

    /**
     * Algorithm kernel tag.
     */
    virtual AlgoKernelTag tag() const  = 0;

    /**
     * If print progress.
     */
    bool verbose() const { return verbose_; }
    void verboseIs(const bool verbose) {
        verbose_ = verbose;
    }

    /**
     * Maximum gather-scatter iterations to run.
     */
    IterCount maxIters() const { return maxIters_; }
    void maxItersIs(const IterCount& maxIters) {
        maxIters_ = maxIters;
    }

    /**
     * The number of partitions in each tile.
     *
     * In order to increase locality during gather phase, etc..
     */
    uint32_t numParts() const { return numParts_; }
    void numPartsIs(const uint32_t numParts) {
        numParts_ = numParts;
    }

    /**
     * Map vertex index to tile index.
     * 
     */
    std::unordered_map< VertexIdx, TileIdx, std::hash<VertexIdx::Type> >& tidMap() {return tidMap_;}
    void tidMapIs(std::unordered_map< VertexIdx, TileIdx, std::hash<VertexIdx::Type> >& tidMap) {
        tidMap_ = tidMap;
    }

    uint64_t getVertexTid(const VertexIdx& dstId) const {
        return (uint64_t)(tidMap_.find(dstId)->second);
    }

    /**
     * Current tid for this party.
     * 
     */
    size_t curTid() {return curTid_;}
    void curTidIs(size_t curTid) {
        curTid_ = curTid;
    }

    /*
     * Task queue handler wrapper
     */
    void taskQueueHandler(std::queue<Task>* taskq) const {
        task_queue_handler(taskq);
    }

    /**
     * Call the algorithm kernel. Run iterations.
     *
     * The algorithm kernel is defined as a functor class.
     *
     * @param graph     Graph tile on which this kernel works.
     * @param cs        Utility for comm & sync.
     */
    virtual void operator()(Ptr<GraphTileType>& graph, CommSyncType& cs) const {
        // If need to print progress, i.e., verbose kernel and primary (index 0) tile.
        auto printProgress = verbose() && (graph->tid() == 0);

        // Start barrier, ensure all preparation is done in all threads.
        // cs.barrier(graph->tid());
        std::vector<std::thread> algo_kernel_server_threads;
        runAlgoKernelServer(algo_kernel_server_threads);
        onAlgoKernelStart(graph);

        IterCount iter(0);
        bool allConverged = false;
        while (!allConverged && iter < maxIters()) {
            bool converged = onIteration(graph, cs, iter);
            onIterationEnd(graph, iter);
            // if (printProgress) info("->%lu", iter.cnt());
            // Check if all tiles have converged.
            // allConverged = cs.barrierAND(graph->tid(), converged);
            iter++;
        }
        // if (printProgress) info("Completed in %lu iterations", iter.cnt());

        onAlgoKernelEnd(graph);
        closeAlgoKernelServer(algo_kernel_server_threads);
        
        std::cout<<graph->tid()<<" "<<"Finish all iterations and decrypt vertex data"<<std::endl;
    }

protected:
    /**
     * Operations on end of each iteration.
     */
    virtual void onIterationEnd(Ptr<GraphTileType>&, const IterCount&) const { }

    /**
     * Operations on start of the algorithm kernel.
     */
    virtual void onAlgoKernelStart(Ptr<GraphTileType>&) const { }

    /**
     * Operations on end of the algorithm kernel.
     */
    virtual void onAlgoKernelEnd(Ptr<GraphTileType>&) const { }

    /**
     * Run kernel servers for coordinating remote algo kernel execution.
     */
    virtual void runAlgoKernelServer(std::vector<std::thread>& threads) const = 0;

    /**
     * Close kernel servers for coordinating remote algo kernel execution.
     */
    virtual void closeAlgoKernelServer(std::vector<std::thread>& threads) const = 0;

protected:
    /**
     * Iteration.
     *
     * @param graph     Graph tile on which this kernel works.
     * @param cs        Utility for comm & sync.
     * @param iter      Current iteration count.
     *
     * @return          If converged in this tile.
     */
    virtual bool onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, const IterCount& iter) const = 0;

protected:
    string name_;
    bool verbose_;
    IterCount maxIters_;
    uint32_t numParts_;
    std::unordered_map< VertexIdx, TileIdx, std::hash<VertexIdx::Type> > tidMap_;
    size_t curTid_;

protected:
    BaseAlgoKernel(const string& name)
        : name_(name), verbose_(false), maxIters_(INF_ITER_COUNT), numParts_(1)
    {
        // Nothing else to do.
    }

    BaseAlgoKernel(const BaseAlgoKernel&) = delete;
    BaseAlgoKernel& operator=(const BaseAlgoKernel&) = delete;
    BaseAlgoKernel(BaseAlgoKernel&&) = delete;
    BaseAlgoKernel& operator=(BaseAlgoKernel&&) = delete;
    bool operator==(const BaseAlgoKernel&) const = delete;

};

// template<typename GraphTileType>
// class VertexCentricAlgoKernel : public BaseAlgoKernel<GraphTileType> {
// public:
//     typedef typename GraphTileType::VertexType VertexType;
//     typedef typename GraphTileType::EdgeType::WeightType EdgeWeightType;
//     typedef typename GraphTileType::UpdateType UpdateType;
//     typedef typename GraphTileType::MirrorVertexType MirrorVertexType;

// public:
//     AlgoKernelTag tag() const final {
//         return AlgoKernelTag::VertexCentric;
//     }

// protected:
//     /**
//      * Vertex-centric gather function.
//      *
//      * @param iter      Current iteration count.
//      * @param src       Source vertex.
//      * @param weight    Weight of the edge.
//      *
//      * @return          Output update data.
//      */
//     virtual UpdateType
//     gather(const IterCount& iter, Ptr<VertexType>& src, EdgeWeightType& weight) const = 0;

//     /**
//      * Vertex-centric apply function.
//      *
//      * @param iter      Current iteration count.
//      * @param v         Vertex.
//      * @param accUpdate Accumulated update data.
//      */
//     virtual void
//     apply(const IterCount& iter, Ptr<VertexType>& v, const UpdateType& accUpdate) const = 0;

//     /**
//      * Vertex-centric scatter function.
//      *
//      * @param iter      Current iteration count.
//      * @param src       Source vertex.
//      *
//      * @return          Whether to activate the destination vertex for gathering
//      *                  in the next iteration.
//      */
//     virtual bool
//     scatter(const IterCount& iter, Ptr<VertexType>& src) const = 0;

// protected:
//     using typename BaseAlgoKernel<GraphTileType>::CommSyncType;
//     bool onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, const IterCount& iter) const final;

// protected:
//     VertexCentricAlgoKernel(const string& name)
//         : BaseAlgoKernel<GraphTileType>(name)
//     {
//         // Nothing else to do.
//     }
// };

} // namespace GraphGASLite

#endif // ALGO_KERNEL_H_
