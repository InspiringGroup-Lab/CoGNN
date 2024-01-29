#ifndef EDGE_CENTRIC_ALGO_KERNEL_H_
#define EDGE_CENTRIC_ALGO_KERNEL_H_

#include "algo_kernel.h"
#include <thread>

namespace GraphGASLite {

template<typename GraphTileType>
class EdgeCentricAlgoKernel : public BaseAlgoKernel<GraphTileType> {
public:
    typedef typename GraphTileType::VertexType VertexType;
    typedef typename GraphTileType::EdgeType::WeightType EdgeWeightType;
    typedef typename GraphTileType::UpdateType UpdateType;
    typedef typename GraphTileType::MirrorVertexType MirrorVertexType;

public:
    AlgoKernelTag tag() const final {
        return AlgoKernelTag::EdgeCentric;
    }

protected:
    /**
     * Edge-centric scatter function.
     *
     * @param iter      Current iteration count.
     * @param src       Source vertex.
     * @param weight    Weight of the edge.
     *
     * @return          A pair consisting of the output update data, and a bool
     *                  denoting whether the update is valid.
     */
    virtual std::pair<UpdateType, bool>
    scatter(const IterCount& iter, Ptr<VertexType>& src, EdgeWeightType& weight) const = 0;


    /**
     * Edge-centric scatter task, get result to an UpdateType object.
     *
     * @param task      Task object
     *
     * @return          The UpdateType object.
     */
    virtual UpdateType
    getScatterTaskResult(const Task& task) const = 0;

    /**
     * Edge-centric gather function.
     *
     * @param iter      Current iteration count.
     * @param dst       Destination vertex.
     * @param update    Input update data.
     *
     * @return          Whether this vertex is converged.
     */
    virtual bool
    gather(const IterCount& iter, Ptr<VertexType>& dst, const UpdateType& update) const = 0;

    /**
     * Edge-centric gather task generation function.
     *
     * @param iter      Current iteration count.
     * @param dst       Destination vertex.
     * @param update    Input update data.
     *
     * @return          The Task object.
     */
    virtual struct Task
    genGatherTask(const IterCount& iter, Ptr<VertexType>& dst, const UpdateType& update) const = 0;

    /**
     * Edge-centric gather task, write result to destination point.
     *
     * @param task      Task object
     * @param dst       Destination vertex pointer
     *
     * @return          The UpdateType object.
     */
    virtual void
    writeGatherTaskResult(const Task& task, Ptr<VertexType>& dst) const = 0;

protected:
    using typename BaseAlgoKernel<GraphTileType>::CommSyncType;
    bool onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, const IterCount& iter) const;
    void runAlgoKernelServer(std::vector<std::thread>& threads) const;
    void closeAlgoKernelServer(std::vector<std::thread>& threads) const;

protected:
    EdgeCentricAlgoKernel(const string& name)
        : BaseAlgoKernel<GraphTileType>(name)
    {
        // Nothing else to do.
    }
};

template<typename GraphTileType>
bool EdgeCentricAlgoKernel<GraphTileType>::
onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, const IterCount& iter) const {
    return false;
}

template<typename GraphTileType>
void EdgeCentricAlgoKernel<GraphTileType>::runAlgoKernelServer(std::vector<std::thread>& threads) const {
    issue_server_recv_threads(threads);
}

template<typename GraphTileType>
void EdgeCentricAlgoKernel<GraphTileType>::closeAlgoKernelServer(std::vector<std::thread>& threads) const {
    // TaskComm& clientTaskComm = TaskComm::getClientInstance();
    // clientTaskComm.sendFinish();
    for (auto& thrd : threads)
        thrd.join();
}

}

#endif