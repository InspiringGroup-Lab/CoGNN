#ifndef SS_EDGE_CENTRIC_ALGO_KERNEL_H_
#define SS_EDGE_CENTRIC_ALGO_KERNEL_H_

#include "vertex_centric_algo_kernel.h"
#include "ObliviousMapper.h"
#include "SCIHarness.h"

#include <thread>
#include <chrono>
#include <algorithm>

namespace GraphGASLite {

template<typename GraphTileType>
class SSEdgeCentricAlgoKernel : public EdgeCentricAlgoKernel<GraphTileType> {
public:
    typedef typename GraphTileType::VertexType VertexType;
    typedef typename GraphTileType::EdgeType::WeightType EdgeWeightType;
    typedef typename GraphTileType::UpdateType UpdateType;
    typedef typename GraphTileType::MirrorVertexType MirrorVertexType;

    typedef std::map<std::string, ShareTensorVec> TensorVecMap;

    struct GraphSummary {
        std::vector<Ptr<VertexType>> localVertexVec;
        std::vector<bool> isLocalVertexBorder;
        std::vector<std::vector<Ptr<MirrorVertexType>>> mirrorVertexVecs;
        std::vector<std::vector<EdgeWeightType>> localEdgeWeightVecs;
        ShareVecVec localVertexSvv;
        std::vector<ShareVecVec> remoteVertexSvvs;
        ShareVecVec localVertexSvvBackup;
        std::vector<ShareVecVec> remoteVertexSvvsBackup;

        std::vector<std::vector<uint64_t>> updateSrcOutDeg;
        std::vector<std::vector<uint64_t>> updateDstInDeg;
        std::vector<std::vector<uint64_t>> remoteUpdateDstInDeg;
        std::vector<uint64_t> localVertexInDeg;

        std::vector<std::vector<bool>> isUpdateSrcVertexDummy;
        std::vector<std::vector<bool>> isGatherDstVertexDummy;
        std::vector<std::vector<uint64_t>> updateSrcVertexPos;
        std::vector<std::vector<uint64_t>> updateDstVertexPos;
        std::vector<uint64_t> localVertexPos;
        std::vector<std::vector<uint64_t>> mirrorVertexPos;
        std::vector<std::vector<uint64_t>> remoteMirrorVertexPos;
        std::vector<ShareVecVec> remoteUpdateSvvs;
        std::vector<ShareVecVec> localUpdateSvvs;

        std::vector<TensorVecMap> localVertexInterDataTVs;
        std::vector<TensorVecMap> remoteVertexInterDataTVs;
        std::vector<ShareTensor> localWeight;
        std::vector<ShareTensor> remoteWeight;
        std::vector<DoubleTensor> plainWeight;

        double learningRate;
        uint64_t globalNumSamples;
    };
    typedef struct GraphSummary GraphSummary;

protected:
    using typename BaseAlgoKernel<GraphTileType>::CommSyncType;
    
    void getTwoPartyVertexDataVectorShare(GraphSummary& gs, ShareVecVec& vertexSvv0, ShareVecVec& vertexSvv1) const;
    void mergeTwoPartyVertexDataVectorShare(GraphSummary& gs, ShareVecVec& vertexSvv0, ShareVecVec& vertexSvv1) const;
    bool onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, GraphSummary& gs, const IterCount& iter) const;
    bool onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, const IterCount& iter) const {}
    void onPreprocessClient(Ptr<GraphTileType>& graph, CommSyncType& cs, GraphSummary& gs, bool doOMPreprocess = true) const;
    void onPreprocessServer(std::vector<std::thread>& threads, bool doOMPreprocess = true) const;
    void runAlgoKernelServer(std::vector<std::thread>& threads, Ptr<GraphTileType>& graph, CommSyncType& cs, GraphSummary& gs) const;
    void runAlgoKernelServer(std::vector<std::thread>& threads) const {}
    void closeAlgoKernelServer(std::vector<std::thread>& threads) const;
    void operator()(Ptr<GraphTileType>& graph, CommSyncType& cs) const override; // Override
    using EdgeCentricAlgoKernel<GraphTileType>::scatter;
    using EdgeCentricAlgoKernel<GraphTileType>::getScatterTaskResult;
    using EdgeCentricAlgoKernel<GraphTileType>::gather;
    using EdgeCentricAlgoKernel<GraphTileType>::genGatherTask;
    using EdgeCentricAlgoKernel<GraphTileType>::writeGatherTaskResult;
    virtual uint32_t getPlainNumPerOperand() const = 0;
    virtual uint32_t getPlainNumPerOperand(uint64_t layer) const = 0;
    virtual uint32_t getForwardLayerNum() const = 0;
    virtual uint32_t getBackwardLayerNum() const = 0;
    virtual std::vector<uint32_t> getDimensionVec() const = 0;
    virtual struct Task genScatterTask(Ptr<VertexType> src, Ptr<VertexType> dst, const ShareVec& vertexData, EdgeWeightType weight, uint64_t srcId, uint64_t srcTid, uint64_t dstId, uint64_t dstTid, bool isDummy=false) const = 0;
    virtual void PreScatterComp(
        GraphSummary& gs,
        const ShareVecVec& vertexSvv, 
        std::vector<uint64_t>& vertexOutDeg, 
        ShareVecVec& scaledVertexSvv,
        uint64_t iter,
        uint64_t coTid, 
        int party
    ) const = 0;
    virtual void ScatterComp(
        ShareVecVec& updateSrcSvv, 
        std::vector<uint64_t>& updateSrcOutDeg, 
        std::vector<uint64_t>& updateDstInDeg, 
        ShareVecVec& duplicatedUpdateSvv,
        uint64_t coTid, 
        int party
    ) const = 0;
    virtual void UpdatePreMergeComp(
        ShareVecVec& duplicatedUpdateSvv,
        std::vector<uint64_t>& updateDstVertexPos, 
        uint64_t coTid, 
        int party
    ) const = 0;
    virtual void writeGatherTaskResult(const Task& task, ShareVec& dst) const = 0;
    // virtual void writeGatherTaskResult(const std::vector<Task>& task, std::vector<ShareVec>& dst) const = 0;
    virtual struct Task genGatherTask(const ShareVec& vertexData, const ShareVec& update, uint64_t dstId, uint64_t dstTid, bool isDummy=false) const = 0;
    virtual void GatherComp(
        ShareVecVec& vertexSvv, 
        ShareVecVec& updateSvv, 
        std::vector<bool>& isGatherDstVertexDummy,
        std::vector<uint64_t>& localVertexInDeg, 
        uint64_t iter,
        uint64_t updateSrcTid,
        uint64_t coTid, 
        int party
    ) const = 0;
    virtual std::vector<Task> genApplyTaskVec(GraphSummary& gs, uint64_t iter, const std::vector<ShareVec>& vertexDataVec, uint64_t dstTid, bool isClient, bool isDummy=false) const = 0;
    virtual void writeApplyTaskVecResult(GraphSummary& gs, uint64_t iter, const std::vector<Task>& taskVec, std::vector<ShareVec>& dstVec, bool isClient) const = 0;
    virtual void ApplyComp(
        GraphSummary& gs, 
        uint64_t iter, 
        const std::vector<ShareVec>& vertexDataVec, 
        const std::vector<uint64_t>& localVertexInDeg, 
        std::vector<ShareVec>& dstVec,
        uint64_t tileIndex,
        uint64_t dstTid, 
        bool isClient
    ) const = 0;
    void fromScatterTaskvResultToPreMergingTaskv(std::vector<Task>& taskv) const;
    virtual void onAlgoKernelStart(Ptr<GraphTileType>& graph, GraphSummary& gs) const = 0;

protected:
    SSEdgeCentricAlgoKernel(const string& name)
        : EdgeCentricAlgoKernel<GraphTileType>(name)
    {
        // Nothing else to do.
    }
};

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::
getTwoPartyVertexDataVectorShare(GraphSummary& gs, ShareVecVec& vertexSvv0, ShareVecVec& vertexSvv1) const {
    for (uint64_t i=0; i<gs.localVertexVec.size(); ++i) {
        const auto& v = gs.localVertexVec[i];
        auto& data = v->data();
        ShareVec cur_sv0;
        ShareVec cur_sv1;
        data.intoShareVec(cur_sv0, cur_sv1);
        vertexSvv0.push_back(cur_sv0);
        vertexSvv1.push_back(cur_sv1);
        // if (cur_sv0[0] + cur_sv1[0] == 0) printf(">>>>>> %d\n", i);
    }    
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::
mergeTwoPartyVertexDataVectorShare(GraphSummary& gs, ShareVecVec& vertexSvv0, ShareVecVec& vertexSvv1) const {
    for (int i=0; i<gs.localVertexVec.size(); ++i) {
        auto& data = gs.localVertexVec[i]->data();
        data.fromShareVec(vertexSvv0[i], vertexSvv1[i]);
    }    
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::
operator()(Ptr<GraphTileType>& graph, CommSyncType& cs) const {
    TaskComm& clientTaskComm = TaskComm::getClientInstance();
    size_t tileNum = clientTaskComm.getTileNum();
    size_t tileIndex = clientTaskComm.getTileIndex();
    std::cout<<tileIndex<<" "<<"Initialize graph algo kernel"<<std::endl;

    GraphSummary gs;
    
    this->onAlgoKernelStart(graph, gs);

    std::cout<<tileIndex<<" "<<"Begin graph preprocessing"<<std::endl;
    
    // Preprocessing
    std::vector<std::thread> preprocessServerThreads;

    bool doPreprocess = !clientTaskComm.getNoPreprocess();

    this->onPreprocessServer(preprocessServerThreads, doPreprocess);

    auto t_preprocess = std::chrono::high_resolution_clock::now();

    this->onPreprocessClient(graph, cs, gs, doPreprocess);
    
    print_duration(t_preprocess, "preprocess");

    // this->onPreprocessServer(preprocessServerThreads, false);
    // this->onPreprocessClient(graph, cs, gs, false);
    
    for (auto& thrd : preprocessServerThreads)
        thrd.join();

    std::cout<<tileIndex<<" "<<"Begin vertex data sharing"<<std::endl;
    // Share Vertex data
    ShareVecVec& localVertexSvv = gs.localVertexSvv;
    ShareVecVec remoteLocalVertexSvv;
    printf("Here1\n");
    this->getTwoPartyVertexDataVectorShare(gs, localVertexSvv, remoteLocalVertexSvv);
    printf("local vertex svv size %d\n", localVertexSvv.size());
    printf("remote vertex svv size %d\n", remoteLocalVertexSvv.size());

    for (int i=0; i<tileNum; ++i) {
        if (i != tileIndex)
            clientTaskComm.sendShareVecVec(remoteLocalVertexSvv, i);
    }

    printf("Here2\n");

    TaskComm& serverTaskComm = TaskComm::getServerInstance();
    std::vector<ShareVecVec>& remoteVertexSvvs = gs.remoteVertexSvvs;    
    remoteVertexSvvs.resize(tileNum);
    for (int i=0; i<tileNum; ++i) {
        if (i != tileIndex)
            serverTaskComm.recvShareVecVec(remoteVertexSvvs[i], i);
        std::cout<<tileIndex<<" Preprocess "<<remoteVertexSvvs[i].size()<<" "<<i<<std::endl;
    }

    // Backup sample feature
    gs.localVertexSvvBackup = gs.localVertexSvv;
    gs.remoteVertexSvvsBackup = gs.remoteVertexSvvs;

    printf("Here3\n");

    clientTaskComm.sendShareTensorVec(gs.remoteWeight, (tileIndex + 1) % tileNum);
    serverTaskComm.recvShareTensorVec(gs.remoteWeight, (tileIndex + tileNum - 1) % tileNum);

    std::cout<<tileIndex<<" "<<"Begin algo kernel iteration"<<std::endl;

    std::vector<std::thread> algo_kernel_server_threads;
    this->runAlgoKernelServer(algo_kernel_server_threads, graph, cs, gs);
    
    IterCount iter(0);
    bool allConverged = false;
    while (!allConverged && iter < this->maxIters()) {
        auto t_iteration = std::chrono::high_resolution_clock::now();
        bool converged = this->onIteration(graph, cs, gs, iter);
        print_duration(t_iteration, "iteration");

        this->onIterationEnd(graph, iter);
        iter++;
    }

    printf(">>H1\n");

    this->closeAlgoKernelServer(algo_kernel_server_threads);

    // std::cout<<graph->tid()<<" "<<"Finish all iterations and begin merging vertex data"<<std::endl;

    // // Merge vertex data
    // uint32_t dstTid = (tileNum + tileIndex - 1) % tileNum;
    // uint32_t srcTid = (tileIndex + 1) % tileNum;
    // // printf("Here0\n");
    // serverTaskComm.sendShareVecVec(gs.remoteVertexSvvs[dstTid], dstTid);
    // remoteLocalVertexSvv.clear();
    // // printf("Here1\n");
    // clientTaskComm.recvShareVecVec(remoteLocalVertexSvv, srcTid);
    // // printf("Here2\n");
    // this->mergeTwoPartyVertexDataVectorShare(gs, gs.localVertexSvv, remoteLocalVertexSvv);
    // // printf("Here3\n");

    printf(">>H2\n");
    
    serverTaskComm.sendFinish();
    printf(">>H3\n");
    clientTaskComm.recvFinish();
    printf(">>H4\n");

    // this->onAlgoKernelEnd(graph);
    std::cout<<graph->tid()<<" "<<"Finish algo kernel"<<std::endl;
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::
onPreprocessClient(Ptr<GraphTileType>& graph, CommSyncType& cs, GraphSummary& gs, bool doOMPreprocess) const { 
    std::vector<Ptr<VertexType>>& localVertexVec = gs.localVertexVec;
    std::vector<std::vector<Ptr<MirrorVertexType>>>& mirrorVertexVecs = gs.mirrorVertexVecs;
    const auto tid = graph->tid();
    TaskComm& clientTaskComm = TaskComm::getClientInstance();
    size_t tileNum = clientTaskComm.getTileNum();
    size_t tileIndex = clientTaskComm.getTileIndex();
    uint64_t maxIters = this->maxIters().cnt();

    if (mirrorVertexVecs.size() == 0) mirrorVertexVecs.resize(tileNum);

    std::cout<<tid<<" "<<"Begin graph preprocessing"<<std::endl;

    // Record src vertices in each vertex / mirror vertex
    for (auto edgeIter = graph->edgeIter(); edgeIter != graph->edgeIterEnd(); ++edgeIter) {
        const auto srcId = edgeIter->srcId();
        const auto dstId = edgeIter->dstId();
        // std::cout<<(uint64_t)srcId<<" "<<(uint64_t)dstId<<std::endl;
        const auto& weight = edgeIter->weight();
        
        if (graph->hasVertex(dstId)) {
            // Local destination.
            auto v = graph->vertex(dstId);
            v->pushToSrcVertexv((uint64_t)srcId);
            v->pushToIncomingEdgev(weight);
            v->pushToIsSrcDummyv(false);
        } else {
            // Remote destination, use mirror vertex.
            auto mv = graph->mirrorVertex(dstId);
            mv->pushToSrcVertexv((uint64_t)srcId);
            mv->pushToIncomingEdgev(weight);
            mv->pushToIsSrcDummyv(false);
        }
    }

    // Construct local vertex Pos vec and update variable src vecs in Scatter
    std::vector<uint64_t>& localVertexPos = gs.localVertexPos;
    std::vector<std::vector<uint64_t>>& mirrorVertexPos = gs.mirrorVertexPos;
    std::vector<bool>& isLocalVertexBorder = gs.isLocalVertexBorder;

    std::vector<std::vector<uint64_t>>& updateSrcOutDeg = gs.updateSrcOutDeg;
    std::vector<std::vector<uint64_t>>& updateDstInDeg = gs.updateDstInDeg;
    std::vector<uint64_t>& localVertexInDeg = gs.localVertexInDeg;
    std::vector<std::vector<uint64_t>>& remoteUpdateDstInDeg = gs.remoteUpdateDstInDeg;

    std::vector<std::vector<uint64_t>>& updateSrcVertexPos = gs.updateSrcVertexPos;
    std::vector<std::vector<uint64_t>>& updateDstVertexPos = gs.updateDstVertexPos; // Duplicated
    std::vector<std::vector<bool>>& isUpdateSrcVertexDummy = gs.isUpdateSrcVertexDummy;
    std::vector<std::vector<bool>>& isGatherDstVertexDummy = gs.isGatherDstVertexDummy;
    std::vector<std::vector<EdgeWeightType>>& localEdgeWeightVecs = gs.localEdgeWeightVecs;
    std::vector<ShareVecVec>& remoteUpdateSvvs = gs.remoteUpdateSvvs;
    std::vector<ShareVecVec>& localUpdateSvvs = gs.localUpdateSvvs;

    mirrorVertexPos.resize(tileNum);

    updateSrcOutDeg.resize(tileNum);
    updateDstInDeg.resize(tileNum);
    remoteUpdateDstInDeg.resize(tileNum);

    updateSrcVertexPos.resize(tileNum);
    updateDstVertexPos.resize(tileNum);
    isUpdateSrcVertexDummy.resize(tileNum);
    isGatherDstVertexDummy.resize(tileNum);
    localEdgeWeightVecs.resize(tileNum);
    remoteUpdateSvvs.resize(tileNum);
    localUpdateSvvs.resize(tileNum);

    const uint32_t forwardLayerNum = getForwardLayerNum();
    gs.localVertexInterDataTVs.resize(forwardLayerNum);
    gs.remoteVertexInterDataTVs.resize(forwardLayerNum);
    gs.localWeight.resize(forwardLayerNum);
    gs.remoteWeight.resize(forwardLayerNum);

    std::vector<uint64_t> maxSrcVSize;
    maxSrcVSize.resize(tileNum, 0);
    std::vector<std::vector<uint64_t>> idVecs(tileNum);

    if (!clientTaskComm.getIsNoDummyEdge()) {
        printf("Has dummy edges.\n");
        for (auto vIter = graph->vertexIter(); vIter != graph->vertexIterEnd(); ++vIter) {
            auto v = vIter->second;
            uint64_t vid = (uint64_t)v->vid();
            idVecs[tileIndex].push_back(vid);
            std::vector<uint64_t>& cur_srcvv = v->getSrcVertexv();
            std::vector<EdgeWeightType>& cur_incomingEdgev = v->getIncomingEdgev();
            std::vector<bool>& cur_isSrcDummyv = v->getIsSrcDummyv();

            // Add dummy src vertices
            uint64_t dummy_num = get_next_power_of_2(cur_srcvv.size()) - cur_srcvv.size();
            for (int i=0; i<dummy_num; ++i) {
                cur_srcvv.push_back(vid);
                cur_isSrcDummyv.push_back(true);
                cur_incomingEdgev.push_back(-1);
            }
            if (cur_srcvv.size() > maxSrcVSize[tileIndex]) maxSrcVSize[tileIndex] = cur_srcvv.size();
        }
        for (auto mvIter = graph->mirrorVertexIter(); mvIter != graph->mirrorVertexIterEnd(); ++mvIter) {
            auto mv = mvIter->second;
            uint64_t mvid = (uint64_t)mv->vid();
            uint64_t mvTid = this->getVertexTid(mvid);
            idVecs[mvTid].push_back(mvid);
            std::vector<uint64_t>& cur_srcvv = mv->getSrcVertexv();
            std::vector<EdgeWeightType>& cur_incomingEdgev = mv->getIncomingEdgev();
            std::vector<bool>& cur_isSrcDummyv = mv->getIsSrcDummyv();
            if (cur_srcvv.size() == 0) {
                printf("Unexpected mirror vertex with empty src vertex vec!\n");
                exit(-1);
            }     

            // Add dummy src vertex
            uint64_t dummy_num = get_next_power_of_2(cur_srcvv.size()) - cur_srcvv.size();
            for (int i=0; i<dummy_num; ++i) {
                cur_srcvv.push_back(cur_srcvv[0]);
                cur_isSrcDummyv.push_back(true);
                cur_incomingEdgev.push_back(-1);
            }
            if (cur_srcvv.size() > maxSrcVSize[mvTid]) maxSrcVSize[mvTid] = cur_srcvv.size();
        }
    } else {
        printf("No dummy edges.\n");
        for (auto vIter = graph->vertexIter(); vIter != graph->vertexIterEnd(); ++vIter) {
            auto v = vIter->second;
            uint64_t vid = (uint64_t)v->vid();
            idVecs[tileIndex].push_back(vid);
            std::vector<uint64_t>& cur_srcvv = v->getSrcVertexv();
            std::vector<EdgeWeightType>& cur_incomingEdgev = v->getIncomingEdgev();
            std::vector<bool>& cur_isSrcDummyv = v->getIsSrcDummyv();

            // printf("local vertex vid %d, inDeg %d, outDeg %d\n", vid, v->inDeg().cnt(), v->outDeg().cnt());

            // Add a dummy src for zero incoming degree vertex
            if (cur_srcvv.size() == 0) {
                cur_srcvv.push_back(vid);
                cur_isSrcDummyv.push_back(true);
                cur_incomingEdgev.push_back(-1);
                v->inDegInc();
                v->outDegInc();
            }
            if (cur_srcvv.size() > maxSrcVSize[tileIndex]) maxSrcVSize[tileIndex] = cur_srcvv.size();
        }   
        for (auto mvIter = graph->mirrorVertexIter(); mvIter != graph->mirrorVertexIterEnd(); ++mvIter) {
            auto mv = mvIter->second;
            uint64_t mvid = (uint64_t)mv->vid();
            uint64_t mvTid = this->getVertexTid(mvid);
            idVecs[mvTid].push_back(mvid);
            std::vector<uint64_t>& cur_srcvv = mv->getSrcVertexv();
            std::vector<EdgeWeightType>& cur_incomingEdgev = mv->getIncomingEdgev();
            std::vector<bool>& cur_isSrcDummyv = mv->getIsSrcDummyv();
            if (cur_srcvv.size() == 0) {
                printf("Unexpected mirror vertex with empty src vertex vec!\n");
                exit(-1);
            }     

            if (cur_srcvv.size() > maxSrcVSize[mvTid]) maxSrcVSize[mvTid] = cur_srcvv.size();
        }     
    }

    // // Counting sort on vertex incoming degree
    // std::vector<std::vector<std::vector<uint64_t>>> countingVec;
    // countingVec.resize(tileNum);
    // for (int i=0; i<tileNum; ++i) {
    //     countingVec[i].resize(maxSrcVSize[i]);
    // }
    // for (auto vIter = graph->vertexIter(); vIter != graph->vertexIterEnd(); ++vIter) {
    //     auto v = vIter->second;
    //     uint64_t vid = (uint64_t)vIter->second->vid();
    //     std::vector<uint64_t>& cur_srcvv = v->getSrcVertexv();
    //     if (cur_srcvv.size() > 0) {
    //         countingVec[tileIndex][cur_srcvv.size()-1].push_back(vid);
    //     }
    // }
    // for (auto mvIter = graph->mirrorVertexIter(); mvIter != graph->mirrorVertexIterEnd(); ++mvIter) {
    //     auto mv = mvIter->second;
    //     uint64_t mvid = (uint64_t)mvIter->second->vid();
    //     uint64_t mvTid = this->getVertexTid(mvid);
    //     std::vector<uint64_t>& cur_srcvv = mv->getSrcVertexv();
    //     if (cur_srcvv.size() > 0) {
    //         countingVec[mvTid][cur_srcvv.size()-1].push_back(mvid);
    //     }
    // }

    for (int i=0; i<tileNum; ++i) {
        std::sort(idVecs[i].begin(), idVecs[i].end()); // Non-descending order by id.
    }

    // Update src vertex pos vec and isDummy vec construction based on counting sort result
    for (int i=0; i<tileNum; ++i) {
        for (int j=0; j<idVecs[i].size(); ++j) {
            // Get vertex or mirror vertex
            const auto dstId = idVecs[i][j];
            if (tileIndex == i) {
                auto v = graph->vertex(dstId);
                v->setReorderedIndex(localVertexPos.size());
                localVertexPos.push_back(dstId);
                isLocalVertexBorder.push_back(v->isBorderVertex());
                localVertexInDeg.push_back(v->inDeg().cnt());
                localVertexVec.push_back(v);
                std::vector<uint64_t>& cur_srcvv = v->getSrcVertexv();
                std::vector<bool>& cur_isSrcDummyv = v->getIsSrcDummyv();
                std::vector<EdgeWeightType>& cur_incomingEdgev = v->getIncomingEdgev();
                updateSrcVertexPos[i].insert(updateSrcVertexPos[i].end(), cur_srcvv.begin(), cur_srcvv.end());
                updateDstVertexPos[i].insert(updateDstVertexPos[i].end(), cur_srcvv.size(), dstId);
                for (auto& x : cur_srcvv) updateSrcOutDeg[i].push_back(graph->vertex(x)->outDeg().cnt());
                for (auto& x : cur_srcvv) updateDstInDeg[i].push_back(v->inDeg().cnt());
                isUpdateSrcVertexDummy[i].insert(isUpdateSrcVertexDummy[i].end(), cur_isSrcDummyv.begin(), cur_isSrcDummyv.end());
                localEdgeWeightVecs[i].insert(localEdgeWeightVecs[i].end(), cur_incomingEdgev.begin(), cur_incomingEdgev.end());
                isGatherDstVertexDummy[i].push_back(cur_isSrcDummyv[0]);
            } else {
                auto mv = graph->mirrorVertex(dstId);
                uint64_t mvTid = this->getVertexTid(dstId);
                mirrorVertexPos[mvTid].push_back(dstId);
                mirrorVertexVecs[i].push_back(mv);
                std::vector<uint64_t>& cur_srcvv = mv->getSrcVertexv();
                std::vector<bool>& cur_isSrcDummyv = mv->getIsSrcDummyv();
                std::vector<EdgeWeightType>& cur_incomingEdgev = mv->getIncomingEdgev();
                updateSrcVertexPos[i].insert(updateSrcVertexPos[i].end(), cur_srcvv.begin(), cur_srcvv.end());
                updateDstVertexPos[i].insert(updateDstVertexPos[i].end(), cur_srcvv.size(), dstId);
                for (auto& x : cur_srcvv) updateSrcOutDeg[i].push_back(graph->vertex(x)->outDeg().cnt());
                for (auto& x : cur_srcvv) updateDstInDeg[i].push_back(0);
                isUpdateSrcVertexDummy[i].insert(isUpdateSrcVertexDummy[i].end(), cur_isSrcDummyv.begin(), cur_isSrcDummyv.end());
                localEdgeWeightVecs[i].insert(localEdgeWeightVecs[i].end(), cur_incomingEdgev.begin(), cur_incomingEdgev.end());
            }
        }
    }

    // Send mirrorVertexPos to the target party (use cs to avoid channel conflicts in TaskComm)
    for (int i=0; i<tileNum; ++i) {
        if (i != tileIndex) {
            PosVec tmpPosVec;
            // tmpPosVec.pos.swap(mirrorVertexPos[i]);
            tmpPosVec.pos.swap(updateDstVertexPos[i]);
            cs.sendPosVec(tmpPosVec, tileIndex, i);
            // tmpPosVec.pos.swap(mirrorVertexPos[i]);
            tmpPosVec.pos.swap(updateDstVertexPos[i]);
        }
    }

    // Receive remote mirror vertex pos from the target party
    std::vector<std::vector<uint64_t>>& remoteMirrorVertexPos = gs.remoteMirrorVertexPos;
    remoteMirrorVertexPos.resize(tileNum);
    for (int i=0; i<tileNum; ++i) {
        if (i != tileIndex) {
            PosVec tmpPosVec;
            cs.recvPosVec(tmpPosVec, i, tileIndex);
            tmpPosVec.pos.swap(remoteMirrorVertexPos[i]);
            isGatherDstVertexDummy[i].resize(localVertexPos.size(), true);
            remoteUpdateDstInDeg[i].resize(remoteMirrorVertexPos[i].size(), 0);
            for (int j=0; j<remoteMirrorVertexPos[i].size(); ++j) {
                auto v = graph->vertex(remoteMirrorVertexPos[i][j]);
                remoteUpdateDstInDeg[i][j] = v->inDeg().cnt();
                isGatherDstVertexDummy[i][v->getReorderedIndex()] = false;
            }
        }
    }

    if (!doOMPreprocess) return;

    // uint32_t plainNumPerOperand = getPlainNumPerOperand();
    std::vector<uint32_t> dimensions = getDimensionVec();

    auto t_preprocess_OM = std::chrono::high_resolution_clock::now();
    // Scatter & Pre-merge for taskvs
    std::cout<<tid<<" "<<"Begin preprocessing oblivious mapper"<<std::endl;
    std::vector<std::thread> threads;
    for (int i=0; i<tileNum; ++i) {
        if (i != tileIndex) {
            threads.emplace_back([this, tileIndex, tileNum, i, maxIters, &localVertexPos, &updateSrcVertexPos, &updateDstVertexPos, &mirrorVertexPos, &remoteMirrorVertexPos, dimensions]() {
                uint32_t iter = 0;
                uint64_t batchSize = 0;
                for (iter=0; iter<maxIters; iter+=batchSize) {
                    std::cout<<tileIndex<<" "<<"Preprocessing oblivious mapper, iter "<<iter<<std::endl;
                    uint32_t preprocessId = 0;

                    // Oblivious Mapper
                    // srcPos: local vertex
                    // dstPos: update src vertex for local vertex
                    std::cout<<tileIndex<<" "<<"OM "<<0<<std::endl;
                    if (i == (tileIndex + 1) % tileNum) {
                        client_gcn_batch_oblivious_mapper_preprocess(localVertexPos, updateSrcVertexPos[tileIndex], dimensions, iter, preprocessId, i);
                        preprocessId += 1;
                    }

                    // Oblivious Mapper
                    // srcPos: local vertex
                    // dstPos: update src vertex for mirror vertex
                    std::cout<<tileIndex<<" "<<"OM "<<1<<" "<<i<<std::endl;
                    // for (auto x : updateSrcVertexPos[i]) {
                    //     std::cout<<">>"<<x<<std::endl;
                    // }
                    batchSize = client_gcn_batch_oblivious_mapper_preprocess(localVertexPos, updateSrcVertexPos[i], dimensions, iter, preprocessId, i);
                    preprocessId += 1;

                    // Oblivious Mapper
                    // srcPos: duplicated update dst vertex for local vertex
                    // dstPos: deduplicated update dst vertex for local vertex (same as local vertex)
                    std::cout<<tileIndex<<" "<<"OM "<<2<<" "<<i<<std::endl;
                    if (i == (tileIndex + 1) % tileNum) {
                        client_gcn_batch_oblivious_mapper_preprocess(updateDstVertexPos[tileIndex], localVertexPos, dimensions, iter, preprocessId, i);
                        preprocessId += 1;
                    }

                    // // Oblivious Mapper
                    // // srcPos: duplicated update dst vertex for mirror vertex
                    // // dstPos: deduplicated update dst vertex for mirror vertex (same as mirror vertex)
                    // std::cout<<tileIndex<<" "<<"OM "<<3<<" "<<i<<std::endl;
                    // client_batch_oblivious_mapper_preprocess(updateDstVertexPos[i], mirrorVertexPos[i], plainNumPerOperand, iter, preprocessId, i);
                    // preprocessId += 1;

                    // // Oblivious Mapper
                    // // srcPos: remote mirror vertex
                    // // dstPos: local vertex
                    // std::cout<<tileIndex<<" "<<"OM "<<4<<" "<<i<<std::endl;
                    // client_batch_oblivious_mapper_preprocess(remoteMirrorVertexPos[i], localVertexPos, plainNumPerOperand, iter, preprocessId, i, true);
                    // preprocessId += 1;

                    // Oblivious Mapper
                    // srcPos: remote (duplicated) mirror vertex
                    // dstPos: local vertex
                    std::cout<<tileIndex<<" "<<"OM "<<3<<" "<<i<<std::endl;
                    client_gcn_batch_oblivious_mapper_preprocess(remoteMirrorVertexPos[i], localVertexPos, dimensions, iter, preprocessId, i, true);
                    preprocessId += 1;

                    std::cout<<tileIndex<<" "<<"OM "<<"end of iteration"<<" "<<i<<std::endl;
                }
            });
        }
    }

    for (auto& thrd : threads)
        thrd.join();
    
    print_duration(t_preprocess_OM, "preprocess_OM");
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::
onPreprocessServer(std::vector<std::thread>& threads, bool doOMPreprocess) const {
    if (!doOMPreprocess) return;
	TaskComm& serverTaskComm = TaskComm::getServerInstance();
	size_t tileNum = serverTaskComm.getTileNum();
	size_t tileIndex = serverTaskComm.getTileIndex();
    uint64_t maxIters = this->maxIters().cnt();
    std::vector<uint32_t> dimensions = getDimensionVec();

	for (int i = 0; i < tileNum; i++) {
		if (i != tileIndex) {
			threads.emplace_back([this, i, &serverTaskComm, dimensions, tileIndex, tileNum, maxIters]() {
                uint32_t iter = 0;
                uint64_t batchSize = 0;
                for (iter=0; iter<maxIters; iter+=batchSize) {
                    uint32_t preprocessId = 0;

                    // Oblivious Mapper
                    // srcPos: local vertex
                    // dstPos: update src vertex for local vertex
                    if ((i + 1) % tileNum == tileIndex) {
                        server_gcn_batch_oblivious_mapper_preprocess(dimensions, iter, preprocessId, i);
                        preprocessId += 1;
                    }

                    // Oblivious Mapper
                    // srcPos: local vertex
                    // dstPos: update src vertex for mirror vertex
                    batchSize = server_gcn_batch_oblivious_mapper_preprocess(dimensions, iter, preprocessId, i);
                    preprocessId += 1;

                    // Oblivious Mapper
                    // srcPos: duplicated update dst vertex for local vertex
                    // dstPos: deduplicated update dst vertex for local vertex (same as local vertex)
                    if ((i + 1) % tileNum == tileIndex) {
                        server_gcn_batch_oblivious_mapper_preprocess(dimensions, iter, preprocessId, i);
                        preprocessId += 1;
                    }

                    // // Oblivious Mapper
                    // // srcPos: duplicated update dst vertex for mirror vertex
                    // // dstPos: deduplicated update dst vertex for mirror vertex (same as mirror vertex)
                    // server_batch_oblivious_mapper_preprocess(iter, preprocessId, i);
                    // preprocessId += 1;

                    // // Oblivious Mapper
                    // // srcPos: remote mirror vertex
                    // // dstPos: local vertex
                    // server_batch_oblivious_mapper_preprocess(iter, preprocessId, i);
                    // preprocessId += 1;

                    // Oblivious Mapper
                    // srcPos: remote mirror vertex
                    // dstPos: local vertex
                    server_gcn_batch_oblivious_mapper_preprocess(dimensions, iter, preprocessId, i);
                    preprocessId += 1;

                    // std::cout<<"Server final preprocessId "<<preprocessId<<" "<<i<<" "<<(tileIndex - 1) % tileNum<<std::endl;
                }            
			});
		}
	}
}

template<typename GraphTileType>
bool SSEdgeCentricAlgoKernel<GraphTileType>::
onIteration(Ptr<GraphTileType>& graph, CommSyncType& cs, GraphSummary& gs, const IterCount& iter) const {
    printf("tid-> %lld, iteration-> %lld\n", graph->tid(), iter.cnt());    
    const auto tid = graph->tid();
    TaskComm& clientTaskComm = TaskComm::getClientInstance();
    TaskComm& serverTaskComm = TaskComm::getServerInstance();
    size_t tileNum = clientTaskComm.getTileNum();
    size_t tileIndex = clientTaskComm.getTileIndex();

    uint32_t plainNumPerOperand = getPlainNumPerOperand(iter.cnt());

    const uint32_t forwardLayerNum = getForwardLayerNum();
    const uint32_t backwardLayerNum = getBackwardLayerNum();
    const uint32_t epochLayerNum = forwardLayerNum + backwardLayerNum;
    if (iter % epochLayerNum == 0) gs.localVertexSvv = gs.localVertexSvvBackup; // Go back to the first layer

    std::cout<<tid<<" "<<"Begin Scatter task generation"<<std::endl;
    
    std::vector<ShareVecVec> updateSrcs(tileNum);
    std::vector<std::thread> threads;
    bar_t barrier(tileNum - 1);
    for (int i=0; i<tileNum; ++i) {
        if (i != tileIndex) {
            threads.emplace_back([this, i, tileIndex, tileNum, forwardLayerNum, backwardLayerNum, epochLayerNum, plainNumPerOperand, &gs, &updateSrcs, &clientTaskComm, &serverTaskComm, &iter, &barrier, &graph](){

                uint32_t preprocessId = 0;

                // At the first layer of backward pass, we only do apply.
                if (iter.cnt() % epochLayerNum != 0 && (iter.cnt() % epochLayerNum) % forwardLayerNum == 0) {
                    printf("At the first layer of backward pass (iter %lu), we only do apply.\n", iter.cnt());
                    if (i == (tileIndex + 1) % tileNum) {
                        // set_up_mpc_channel(true, i);

                        // Apply
                        ShareVecVec curResult;
                        ApplyComp(
                            gs, 
                            iter.cnt(), 
                            gs.localVertexSvv, 
                            gs.localVertexInDeg,
                            curResult, 
                            tileIndex,
                            i, 
                            true
                        );
                        gs.localVertexSvv.swap(curResult);

                        // close_mpc_channel(true, i);   
                    }

                    return;                                     
                }

                if (i == (tileIndex + 1) % tileNum) {
                    auto t_PreScatterComp = std::chrono::high_resolution_clock::now();
                    PreScatterComp(
                        gs,
                        gs.localVertexSvv, 
                        gs.localVertexInDeg, 
                        gs.localVertexSvv,
                        iter.cnt(),
                        i, 
                        sci::ALICE
                    );
                    print_duration(t_PreScatterComp, "PreScatterComp Client");
                }

                auto t_Scatter_preparation = std::chrono::high_resolution_clock::now();
                std::cout<<tileIndex<<" "<<"Begin update src mapping with "<<i<<std::endl;

                if (i == (tileIndex + 1) % tileNum) {
                    client_oblivious_mapper_online(gs.localVertexPos, gs.updateSrcVertexPos[tileIndex],
                                        gs.localVertexSvv, updateSrcs[tileIndex], plainNumPerOperand, 
                                        iter.cnt(), preprocessId, i);
                    preprocessId += 1;
                }

                // printf("H1\n");

                client_oblivious_mapper_online(gs.localVertexPos, gs.updateSrcVertexPos[i],
                                    gs.localVertexSvv, updateSrcs[i], plainNumPerOperand, 
                                    iter.cnt(), preprocessId, i);
                preprocessId += 1;

                print_duration(t_Scatter_preparation, "Scatter_preparation");

                // if (i == 1) {
                //     for (int kk=0; kk<gs.localVertexPos.size(); ++kk) {
                //         if (gs.localVertexPos[kk] == 0) {
                //             std::cout<<"Zero pos "<<kk<<std::endl;
                //             auto& x = gs.localVertexSvv[kk];
                //             std::cout<<x[0]<<" "<<x[1]<<"  "<<std::endl;
                //         }
                //     }
                //     for (auto& x : gs.updateSrcVertexPos[i]) std::cout<<x<<" ";
                //     std::cout<<std::endl;
                //     for (auto& x : updateSrcs[i]) std::cout<<x[0]<<" "<<x[1]<<"  ";
                //     std::cout<<std::endl;
                // }

                // Scatter and pre-merging for the mirror vertex of party i
                TaskqHandlerConfig& thc = clientTaskComm.getTaskqHandlerConfig(i);
                thc.sendOperand = false;
                thc.mergeResult = false;
                thc.sendTaskqDigest = false;
                std::vector<Task> taskv;

                auto clientComputeUpdate = [this, &gs, &thc, &clientTaskComm, &taskv, &graph, i, tileIndex](ShareVecVec& updateSrc, ShareVecVec& duplicatedUpdateSvv, const uint32_t dstTid) {
                    // set_up_mpc_channel(true, i);
                    
                    auto t_Scatter_computation = std::chrono::high_resolution_clock::now();

                    uint64_t scatterTaskNum = updateSrc.size();
                    if (scatterTaskNum != gs.localEdgeWeightVecs[dstTid].size()) {
                        printf("Unmatched src num and edge weight num during Scatter!\n");
                        exit(-1);
                    }
                        
                    duplicatedUpdateSvv.clear();
                    this->ScatterComp(updateSrc, gs.updateSrcOutDeg[dstTid], gs.updateDstInDeg[dstTid], duplicatedUpdateSvv, i, sci::ALICE);

                    print_duration(t_Scatter_computation, "Scatter_computation");

                    auto t_premerging = std::chrono::high_resolution_clock::now();

                    this->UpdatePreMergeComp(duplicatedUpdateSvv, gs.updateDstVertexPos[dstTid], i, sci::ALICE);

                    print_duration(t_premerging, "premerging");
                    
                    // close_mpc_channel(true, i);                   
                };

                ShareVecVec duplicatedUpdateSvv;
                std::cout<<tileIndex<<" "<<"Begin update extraction mapping with "<<i<<std::endl;
                if (i == (tileIndex + 1) % tileNum) {
                    clientComputeUpdate(updateSrcs[tileIndex], duplicatedUpdateSvv, tileIndex);
                    auto t_premerged_extraction = std::chrono::high_resolution_clock::now();
                    client_oblivious_mapper_online(gs.updateDstVertexPos[tileIndex], gs.localVertexPos,
                                        duplicatedUpdateSvv, gs.localUpdateSvvs[tileIndex], plainNumPerOperand, 
                                        iter.cnt(), preprocessId, i);
                    preprocessId += 1;
                    print_duration(t_premerged_extraction, "premerged_extraction");
                }

                // printf("H2\n");

                clientComputeUpdate(updateSrcs[i], duplicatedUpdateSvv, i);

                // auto t_premerged_extraction = std::chrono::high_resolution_clock::now();
                // client_oblivious_mapper_online(gs.updateDstVertexPos[i], gs.mirrorVertexPos[i],
                //                     duplicatedUpdateSvv, gs.remoteUpdateSvvs[i], plainNumPerOperand, 
                //                     iter.cnt(), preprocessId, i);
                // preprocessId += 1;
                // print_duration(t_premerged_extraction, "premerged_extraction");
                gs.remoteUpdateSvvs[i].swap(duplicatedUpdateSvv);


                Semaphore& local_update_ready_smp = clientTaskComm.getLocalUpdateReadySmp(i);
                Semaphore& remote_update_ready_smp = serverTaskComm.getRemoteUpdateReadySmp(i);
                remote_update_ready_smp.release();
                local_update_ready_smp.acquire();

                auto t_Gather_preparation = std::chrono::high_resolution_clock::now();

                std::cout<<tileIndex<<" "<<"Begin update extension mapping with "<<i<<std::endl;

                ShareVecVec tmpUpdateSvv;
                client_oblivious_mapper_online(gs.remoteMirrorVertexPos[i], gs.localVertexPos,
                                    gs.localUpdateSvvs[i], tmpUpdateSvv, plainNumPerOperand, 
                                    iter.cnt(), preprocessId, i, true);
                preprocessId += 1;

                gs.localUpdateSvvs[i].clear();
                gs.localUpdateSvvs[i].swap(tmpUpdateSvv);

                print_duration(t_Gather_preparation, "Gather_preparation");

                // Synchronize across all threads
                barrier.wait();

                std::cout<<tileIndex<<" "<<"Begin gather computation mapping with "<<i<<std::endl;
                // Gather
                if (i == (tileIndex + 1) % tileNum) {
                    auto t_Gather_computation = std::chrono::high_resolution_clock::now();
                    // set_up_mpc_channel(true, i);
                    for (int j=0; j<tileNum; ++j) {
                        printf("Client debug here! tile num = %d j = %d gs.localUpdateSvvs.size() = %d\n", tileNum, j, gs.localUpdateSvvs.size());
                        uint64_t gatherTaskNum = gs.localVertexSvv.size();
                        if (gatherTaskNum != gs.localUpdateSvvs[j].size()) {
                            printf("Client Unmatched update num and vertex num during cooperation! %lld %d tile num %d j = %d\n", gatherTaskNum, gs.localUpdateSvvs[j].size(), tileNum, j);
                            exit(-1);
                        }

                        GatherComp(gs.localVertexSvv, gs.localUpdateSvvs[j], gs.isGatherDstVertexDummy[j], gs.localVertexInDeg, iter.cnt(), j, i, sci::ALICE);

                        // if (j == 1 && i == 0)
                        //     for (uint64_t m=0; m<gatherTaskNum; ++m) {
                        //         std::cout<<"2Here "<<m<<" "<<gs.localVertexPos[m]<<" "<<gs.localVertexSvv[m][0]<<" "<<gs.localVertexSvv[m][1]<<" "<<gs.isGatherDstVertexDummy[j][m]<<std::endl;
                        //     }
                    }
                    print_duration(t_Gather_computation, "Gather_computation");

                    auto t_Apply_computation = std::chrono::high_resolution_clock::now();
                    // Apply
                    ShareVecVec curResult;
                    ApplyComp(
                        gs, 
                        iter.cnt(), 
                        gs.localVertexSvv,
                        gs.localVertexInDeg, 
                        curResult, 
                        tileIndex,
                        i, 
                        true
                    );
                    gs.localVertexSvv.swap(curResult);
                    print_duration(t_Apply_computation, "Apply_computation");

                    // close_mpc_channel(true, i);                                        
                }

            });
        }
    }

    for (auto& thrd : threads)
        thrd.join();

    return false;
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::runAlgoKernelServer(std::vector<std::thread>& threads, Ptr<GraphTileType>& graph, CommSyncType& cs, GraphSummary& gs) const {
	TaskComm& clientTaskComm = TaskComm::getClientInstance();
    TaskComm& serverTaskComm = TaskComm::getServerInstance();
	size_t tileNum = serverTaskComm.getTileNum();
	size_t tileIndex = serverTaskComm.getTileIndex();

    uint64_t maxIters = this->maxIters().cnt();

    const uint32_t forwardLayerNum = getForwardLayerNum();
    const uint32_t backwardLayerNum = getBackwardLayerNum();
    const uint32_t epochLayerNum = forwardLayerNum + backwardLayerNum;

    bar_t barrier(tileNum - 1);
	for (int i = 0; i < tileNum; i++) {
		if (i != tileIndex) {
			threads.emplace_back([this, i, &serverTaskComm, &clientTaskComm, &cs, &gs, &barrier, &graph, tileIndex, tileNum, maxIters, forwardLayerNum, backwardLayerNum, epochLayerNum]() {

                uint64_t iter = 0;
                TaskqHandlerConfig& thc = serverTaskComm.getTaskqHandlerConfig(i);
                thc.sendOperand = false;
                thc.mergeResult = false;
                thc.sendTaskqDigest = false;
                std::vector<Task>& taskv = serverTaskComm.getTaskv(i);
                while (iter < maxIters) { // On iteration
                    // set_up_mpc_channel(false, i);
                    if (iter %  epochLayerNum == 0) gs.remoteVertexSvvs[i] = gs.remoteVertexSvvsBackup[i]; // Go back to the first layer

                    // At the first layer of backward pass, we only do apply.
                    if (iter % epochLayerNum != 0 && (iter % epochLayerNum) % forwardLayerNum == 0) {
                        printf("At the first layer of backward pass (iter %lu), we only do apply.\n", iter);
                        if (tileIndex != (i + 1) % tileNum) {
                            cs.recvShareVecVec(gs.remoteVertexSvvs[i], (i + 1) % tileNum, tileIndex);
                            iter++;
                            continue;
                        } else {
                            std::cout<<"Compute Gather Taskv, "<<tileIndex<<" Server, "<<"iter: "<<iter<<" "<<i<<std::endl;
                            // set_up_mpc_channel(false, i);

                            // Apply
                            ShareVecVec curResult;
                            std::vector<uint64_t> zeroDeg(gs.remoteVertexSvvs[i].size(), 0);
                            ApplyComp(
                                gs, 
                                iter, 
                                gs.remoteVertexSvvs[i], 
                                zeroDeg,
                                curResult, 
                                tileIndex,
                                i, 
                                false
                            );
                            gs.remoteVertexSvvs[i].swap(curResult);

                            // Send vertex data share to the other party
                            for (int j=0; j<tileNum; ++j) {
                                if (j != tileIndex && j != i) {
                                    // serverTaskComm.sendShareVecVec(gs.remoteVertexSvvs[i], j);
                                    cs.sendShareVecVec(gs.remoteVertexSvvs[i], tileIndex, j);
                                }
                            }

                            // close_mpc_channel(false, i);
                        }
                        
                        iter++;
                        continue;
                    }

                    if (tileIndex != (i + 1) % tileNum) {
                        cs.recvShareVecVec(gs.remoteVertexSvvs[i], (i + 1) % tileNum, tileIndex);
                    } else {
                        auto t_PreScatterComp = std::chrono::high_resolution_clock::now();
                        std::vector<uint64_t> zeroDeg(gs.remoteVertexSvvs[i].size(), 0);
                        PreScatterComp(
                            gs,
                            gs.remoteVertexSvvs[i], 
                            zeroDeg, 
                            gs.remoteVertexSvvs[i],
                            iter,
                            i, 
                            sci::BOB
                        );
                        print_duration(t_PreScatterComp, "PreScatterComp Server");    
                        // Send vertex data share to the other party
                        for (int j=0; j<tileNum; ++j) {
                            if (j != tileIndex && j != i) {
                                // serverTaskComm.sendShareVecVec(gs.remoteVertexSvvs[i], j);
                                cs.sendShareVecVec(gs.remoteVertexSvvs[i], tileIndex, j);
                            }
                        }                
                    }

                    uint32_t preprocessId = 0;
                    printf("Server iter %lld\n", iter);
                    ShareVecVec coUpdateSrc;
                    ShareVecVec updateSrc;

                    if (tileIndex == (i + 1) % tileNum) {
                        server_oblivious_mapper_online(gs.remoteVertexSvvs[i], coUpdateSrc,
                                        iter, preprocessId, i);
                        preprocessId += 1;
                    }

                    server_oblivious_mapper_online(gs.remoteVertexSvvs[i], updateSrc,
                                    iter, preprocessId, i);
                    preprocessId += 1;
                    // if (i == 0) {
                    //     auto& x = gs.remoteVertexSvvs[i][11750];
                    //     std::cout<<x[0]<<" "<<x[1]<<"  "<<std::endl;;
                    //     for (auto& x : updateSrc) std::cout<<x[0]<<" "<<x[1]<<"  ";
                    //     std::cout<<std::endl;
                    // }

                    auto serverComputeUpdate = [this, &gs, &thc, &serverTaskComm, &taskv, &graph, i, tileIndex](ShareVecVec& updateSrc, ShareVecVec& duplicatedUpdateSvv, bool dstIsLocal) {
                        // set_up_mpc_channel(false, i);
                        // Scatter
                        thc.rotation = 0;
                        uint64_t scatterTaskNum = updateSrc.size();

                        // if (i==0 && scatterTaskNum == 2) {
                        //     for (uint64_t j=0; j<scatterTaskNum; ++j) {
                        //         std::cout<<">> Server Scatter inputs"<<std::endl;
                        //         std::cout<<" "<<updateSrc[j][0]<<" "<<updateSrc[j][1]<<std::endl;
                        //     }                        
                        // }
                        std::vector<uint64_t> zeroDeg(scatterTaskNum, 0);
                        duplicatedUpdateSvv.clear();
                        if (dstIsLocal) { // The dst is local to the src subgraph
                            this->ScatterComp(updateSrc, zeroDeg, zeroDeg, duplicatedUpdateSvv, i, sci::BOB);
                        } else { // The dst is the server's subgraph 
                            this->ScatterComp(updateSrc, zeroDeg, gs.remoteUpdateDstInDeg[i], duplicatedUpdateSvv, i, sci::BOB);                      
                        }

                        // Pre-merge
                        std::vector<uint64_t> zeroPosVec(scatterTaskNum, 0);
                        this->UpdatePreMergeComp(duplicatedUpdateSvv, zeroPosVec, i, sci::BOB);

                        // close_mpc_channel(false, i);
                    };

                    ShareVecVec duplicatedUpdateSvv;
                    if (tileIndex == (i + 1) % tileNum) {
                        serverComputeUpdate(coUpdateSrc, duplicatedUpdateSvv, true);
                        // printf("++++%d\n", gs.remoteUpdateSvvs.size());
                        server_oblivious_mapper_online(duplicatedUpdateSvv, gs.remoteUpdateSvvs[tileIndex], 
                                            iter, preprocessId, i);
                        preprocessId += 1;
                        duplicatedUpdateSvv.clear();
                    }

                    serverComputeUpdate(updateSrc, duplicatedUpdateSvv, false);
                    // server_oblivious_mapper_online(duplicatedUpdateSvv, gs.localUpdateSvvs[i], 
                    //                     iter, preprocessId, i);
                    // preprocessId += 1;
                    gs.localUpdateSvvs[i].swap(duplicatedUpdateSvv);

                    Semaphore& local_update_ready_smp = clientTaskComm.getLocalUpdateReadySmp(i);
                    Semaphore& remote_update_ready_smp = serverTaskComm.getRemoteUpdateReadySmp(i);
                    local_update_ready_smp.release();
                    remote_update_ready_smp.acquire();

                    ShareVecVec tmpUpdateSvv;
                    server_oblivious_mapper_online(gs.remoteUpdateSvvs[i], tmpUpdateSvv,
                                        iter, preprocessId, i);
                    preprocessId += 1;

                    gs.remoteUpdateSvvs[i].clear();
                    gs.remoteUpdateSvvs[i].swap(tmpUpdateSvv);

                    // Synchronize across all threads
                    barrier.wait();

                    std::cout<<"Barrier, "<<tileIndex<<" Server, "<<"iter: "<<iter<<" "<<i<<std::endl;

                    std::vector<ShareVecVec> remoteUpdateSvvs;

                    if (tileIndex != (i + 1) % tileNum) {
                        cs.sendShareVecVec(gs.remoteUpdateSvvs[i], tileIndex, (i + 1) % tileNum);
                    } else {
                        remoteUpdateSvvs.resize(tileNum);
                        remoteUpdateSvvs[tileIndex].swap(gs.remoteUpdateSvvs[i]);
                        remoteUpdateSvvs[i].swap(gs.remoteUpdateSvvs[tileIndex]);
                        for (int j=0; j<tileNum; ++j) {
                            if (j != tileIndex && j != i) {
                                cs.recvShareVecVec(remoteUpdateSvvs[j], j, tileIndex);
                            }
                        }
                    }

                    barrier.wait();

                    if (tileIndex != (i + 1) % tileNum) {
                        // serverTaskComm.sendShareVecVec(gs.remoteUpdateSvvs[i], (i + 1) % tileNum);
                        // std::cout<<tileIndex<<" Server delegates "<<gs.remoteUpdateSvvs[i].size()<<" to "<<(i + 1) % tileNum<<std::endl;
                        // std::cout<<gs.remoteUpdateSvvs[i][0][0]<<" ++ "<<gs.remoteUpdateSvvs[i][0][1]<<std::endl;
                        // clientTaskComm.recvShareVecVec(gs.remoteVertexSvvs[i], (i + 1) % tileNum);
                        // std::cout<<"Received Gather Taskv result, "<<tileIndex<<" Server, "<<"iter: "<<iter<<" "<<i<<std::endl;
                        cs.recvShareVecVec(gs.remoteVertexSvvs[i], (i + 1) % tileNum, tileIndex);
                    } else {
                        std::cout<<"Compute Gather Taskv, "<<tileIndex<<" Server, "<<"iter: "<<iter<<" "<<i<<std::endl;
                        // set_up_mpc_channel(false, i);
                        for (int j=0; j<tileNum; ++j) {
                            uint64_t gatherTaskNum = remoteUpdateSvvs[j].size();
                            if (gatherTaskNum != gs.remoteVertexSvvs[i].size()) {
                                printf("%d Server Unmatched update num and vertex num during cooperation! %d %d %d %d tile num %d\n", tileIndex, gatherTaskNum, gs.remoteVertexSvvs[i].size(), i, j, tileNum);
                                exit(-1);
                            }
                            for (int m=0; m<gatherTaskNum; ++m) {
                                taskv.push_back(this->genGatherTask(gs.remoteVertexSvvs[i][m], remoteUpdateSvvs[j][m], -1, i));
                            }

                            std::vector<bool> zeroIsDummy(gatherTaskNum, false);
                            std::vector<uint64_t> zeroDeg(gatherTaskNum, 0);
                            GatherComp(gs.remoteVertexSvvs[i], remoteUpdateSvvs[j], zeroIsDummy, zeroDeg, iter, j, i, sci::BOB);

                            // if (j == 1 && i == 1) {
                            //     // uint64_t m=2718;
                            //     // std::cout<<"2THere "<<m<<" "<<gs.remoteVertexSvvs[i][m][0]<<" "<<gs.remoteVertexSvvs[i][m][1]<<std::endl;
                            //     // m=8796;
                            //     for (int m=0; m<gatherTaskNum; ++m)
                            //         std::cout<<"2THere "<<m<<" "<<gs.remoteVertexSvvs[i][m][0]<<" "<<gs.remoteVertexSvvs[i][m][1]<<std::endl;
                            // } 
                        }

                        // Apply
                        ShareVecVec curResult;
                        std::vector<uint64_t> zeroDeg(gs.remoteVertexSvvs[i].size(), 0);
                        ApplyComp(
                            gs, 
                            iter, 
                            gs.remoteVertexSvvs[i], 
                            zeroDeg,
                            curResult, 
                            tileIndex,
                            i, 
                            false
                        );
                        gs.remoteVertexSvvs[i].swap(curResult);

                        // Send vertex data share to the other party
                        for (int j=0; j<tileNum; ++j) {
                            if (j != tileIndex && j != i) {
                                // serverTaskComm.sendShareVecVec(gs.remoteVertexSvvs[i], j);
                                cs.sendShareVecVec(gs.remoteVertexSvvs[i], tileIndex, j);
                            }
                        }

                        // close_mpc_channel(false, i);
                    }

                    barrier.wait();

                    std::cout<<"Finish iteration, "<<tileIndex<<" Server, "<<"iter: "<<iter<<" "<<i<<std::endl;

                    // if (tileIndex != (i + 1) % tileNum) {
                    //     serverTaskComm.recvShareVecVec(gs.remoteVertexSvvs[i], (i + 1) % tileNum);
                    //     std::cout<<"Received Gather Taskv result, "<<tileIndex<<" Server, "<<"iter: "<<iter<<" "<<i<<std::endl;                        
                    // } else {
                    //     // Send vertex data share to the other party
                    //     for (int j=0; j<tileNum; ++j) {
                    //         if (j != tileIndex && j != i) {
                    //             clientTaskComm.sendShareVecVec(gs.remoteVertexSvvs[i], j);
                    //         }
                    //     }
                    // }

                    // barrier.wait();

                    // close_mpc_channel(false, i);

                    iter++;
                }

			});
		}
	}
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::
fromScatterTaskvResultToPreMergingTaskv(std::vector<Task>& taskv) const {
    std::vector<Task> new_taskv;
    for (auto& ret : taskv) {
        const auto update = this->getScatterTaskResult(ret);
        const auto dstId = ret.vertexIndex;
        const auto dstTid = ret.dstTid;
        new_taskv.push_back(UpdateType::genTask(update, update, dstId, dstId, dstTid, true, ret.isDummy));
        ret.delete_task_content_buf();
    }
    taskv.clear();
    std::swap(taskv, new_taskv);    
}

template<typename GraphTileType>
void SSEdgeCentricAlgoKernel<GraphTileType>::closeAlgoKernelServer(std::vector<std::thread>& threads) const {
    // TaskComm& clientTaskComm = TaskComm::getClientInstance();
    // clientTaskComm.sendFinish();
    for (auto& thrd : threads)
        thrd.join();
}

}

#endif