#ifndef ALGO_KERNELS_EDGE_CENTRIC_GCN_GCN_H_
#define ALGO_KERNELS_EDGE_CENTRIC_GCN_GCN_H_

#include "graph.h"
#include "ss_vertex_centric_algo_kernel.h"
#include "task.h"
#include "TaskUtil.h"
#include "SCIHarness.h"
#include "SecureAggregation.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/access.hpp>

#include <vector>
#include <map>
#include <math.h>

// #define GCN_LOG

// template <typename Archive> 
//     void serialize(Archive &ar, CipherEntry& ce, const unsigned int version) {
//     ar & ce.data;
//     ar & ce.nac;
//     ar & ce.mac;
// }

#define INF(type) (std::numeric_limits<type>::max()/2)
#define INV_VID -1uL

// #define gnnParam.num_layers 2
// #define gnnParam.num_labels 7
// #define gnnParam.input_dim 1433
// // #define gnnParam.input_dim 80
// #define gnnParam.hidden_dim 16
// #define GLOBLAL_NUM_SAMPLES 2708

// #define gnnParam.num_layers 2
// #define gnnParam.num_labels 3
// // #define gnnParam.input_dim 1433
// #define gnnParam.input_dim 2
// #define gnnParam.hidden_dim 3
// #define GLOBLAL_NUM_SAMPLES 4

/*
 * Graph types definitions.
 */
struct GCNData {
    // # Feature
    std::vector<double> feature;
    int label;

    // GraphGASLite::IterCount activeIter;

    // GCNData(const std::vector<double>& feat, int lab)
    //     : feature(feat), label(lab)
    // {
    //     // Nothing else to do.
    // }

    GCNData(const GraphGASLite::VertexIdx&) {}

    void intoShareVec(ShareVec& sv0, ShareVec& sv1) {
        std::vector<uint64_t> feature_share0, feature_share1;
        size_t feature_length = feature.size();
        feature_share0.resize(feature_length);
        feature_share1.resize(feature_length);
        for (int i=0; i<feature_length; ++i) {
            CryptoUtil::intoShares(feature[i], feature_share0[i], feature_share1[i]);
        }

        sv0.swap(feature_share0);
        sv1.swap(feature_share1);
    }

    void fromShareVec(const ShareVec& sv0, const ShareVec& sv1) {
        size_t feature_length = feature.size();
        for (int i=0; i<feature_length; ++i) {
            feature[i] = CryptoUtil::mergeShareAsDouble(sv0[i], sv1[i]);
        }
    }
};

void intoShareTensor(const DoubleTensor& dt, ShareTensor& st0, ShareTensor& st1) {
    size_t dim0 = dt.size();
    st0.resize(dim0);
    st1.resize(dim0);
    
    size_t dim1 = dt[0].size();
    #pragma omp parallel for
    for (int i=0; i<dim0; ++i) {
        st0[i].resize(dim1);
        st1[i].resize(dim1);            
        for (int j=0; j<dim1; ++j) {
            CryptoUtil::intoShares(dt[i][j], st0[i][j], st1[i][j]);
        }
    }
}

struct GCNUpdate {
    ShareVec embedding;

    GCNUpdate(const ShareVec& embed)
        : embedding(embed) {
        // Nothing else to do.
    }

    GCNUpdate() {}

    friend class boost::serialization::access;
    
    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) {
        ar & embedding;
    }

    static Task genTask(const GCNUpdate& update1, const GCNUpdate& update2, uint64_t vertexIndex, uint64_t srcTid, uint64_t dstTid, bool isFinal=false, bool isDummy=false) {
        GCNVectorAddition* gva = new GCNVectorAddition();

        gva->operands[0] = update1.embedding;
        gva->operands[1] = update2.embedding;     
        
        struct Task task = {GCN_VECTOR_ADDITION, vertexIndex, false, (void*)gva, static_cast<uint64_t>(-1), dstTid, srcTid, 0, isFinal, isDummy};
        return task;
    }

    static GCNUpdate getTaskResult(const struct Task& task) {
        if (!task.finished) {
            throw ResultNotReadyException("Task result not ready yet!\n");
        }
        GCNVectorAddition* gva = (GCNVectorAddition*)task.buf;
        GCNUpdate result = GCNUpdate(gva->operands[0]);
        return result;
    }

    static void getTaskResult(const struct Task& task, ShareVec& sv) {
        if (!task.finished) {
            throw ResultNotReadyException("Task result not ready yet!\n");
        }
        GCNVectorAddition* gva = (GCNVectorAddition*)task.buf;
        sv = gva->operands[0];
    }

    static GCNUpdate splitRandomShareFromUpdate(const uint32_t srcTid, const uint32_t dstTid, GCNUpdate& update) {
        GCNUpdate update_share;
        return update_share;
    }

    static void mergeUpdateShare(const uint32_t dstTid, GCNUpdate& update, GCNUpdate& update_share) {
    }
};

/*
 * Algorithm kernel definition.
 */
template<typename GraphTileType>
class GCNEdgeCentricAlgoKernel : public GraphGASLite::SSEdgeCentricAlgoKernel<GraphTileType> {
public:
    static Ptr<GCNEdgeCentricAlgoKernel> instanceNew(const string& name,
            const GraphGASLite::VertexIdx& src) {
        return Ptr<GCNEdgeCentricAlgoKernel>(new GCNEdgeCentricAlgoKernel(name, src));
    }

protected:
    typedef typename GraphTileType::UpdateType UpdateType;
    typedef typename GraphTileType::VertexType VertexType;
    typedef typename GraphTileType::EdgeType::WeightType EdgeWeightType;
    typedef typename GraphGASLite::SSEdgeCentricAlgoKernel<GraphTileType>::GraphSummary GraphSummary; 
    typedef typename GraphGASLite::SSEdgeCentricAlgoKernel<GraphTileType>::TensorVecMap TensorVecMap; 

    std::pair<UpdateType, bool> scatter(const GraphGASLite::IterCount& iter, Ptr<VertexType>& src, EdgeWeightType& weight) const {
        return std::make_pair(GCNUpdate(), false);
    }

    struct Task genScatterTask(Ptr<VertexType> src, Ptr<VertexType> dst, const ShareVec& vertexData, EdgeWeightType weight, uint64_t srcId, uint64_t srcTid, uint64_t dstId, uint64_t dstTid, bool isDummy=false) const {
        GCNVectorScale* gvs = new GCNVectorScale();
        double srcDeg = 0;
        double dstDeg = 0;

        if (src != nullptr) {
            if ((double)src->outDeg().cnt() == 0) srcDeg = 0;
            else srcDeg = pow((double)src->outDeg().cnt(), -0.5);
        }

        if (dst != nullptr) {
            if ((double)dst->inDeg().cnt() == 0) dstDeg = 0;
            else dstDeg = pow((double)dst->inDeg().cnt(), -0.5);
        }
        
        gvs->writeShareToOperand(vertexData, static_cast<uint64_t>(srcDeg*(1<<SCALER_BIT_LENGTH)), static_cast<uint64_t>(dstDeg*(1<<SCALER_BIT_LENGTH)));

        struct Task task = {GCN_VECTOR_SCALE, dstId, false, (void*)gvs, srcId, dstTid, srcTid};
        task.isDummy = isDummy;
        return task;
    }

    void PreScatterComp(
        GraphSummary& gs,
        const ShareVecVec& vertexSvv, 
        std::vector<uint64_t>& vertexOutDeg, 
        ShareVecVec& scaledVertexSvv,
        uint64_t iter,
        uint64_t coTid, 
        int party
    ) const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        uint32_t epochLayerNum = getEpochLayerNum();
        uint32_t forwardLayerNum = getForwardLayerNum();
        uint32_t backwardLayerNum = getBackwardLayerNum();
        bool isForward = ((iter % epochLayerNum) < forwardLayerNum); 
        uint32_t coForwardLayer = 0;
        if (isForward) coForwardLayer = iter % epochLayerNum;
        else coForwardLayer = forwardLayerNum - 1 - (((iter % epochLayerNum) - forwardLayerNum) / 2);
        bool isClient = (party == sci::ALICE);

        size_t length = vertexSvv.size();
        std::vector<uint64_t> normalizer(length);
        for (int i = 0; i < length; ++i) {
            normalizer[i] = vertexOutDeg[i] == 0 ? 0 : CryptoUtil::encodeDoubleAsFixedPoint(pow((double)vertexOutDeg[i] + 1, -0.5));
        }

        if (isForward) { // FORWARD
            const ShareTensor& weight = isClient? gs.localWeight[coForwardLayer]:gs.remoteWeight[coForwardLayer];
            TensorVecMap& vertexInterData = isClient? gs.localVertexInterDataTVs[coForwardLayer]:gs.remoteVertexInterDataTVs[coForwardLayer];

            TaskComm& clientTaskComm = TaskComm::getClientInstance();
            size_t tileNum = clientTaskComm.getTileNum();
            size_t tileIndex = clientTaskComm.getTileIndex();
            if (isClient && coTid == (tileIndex + 1) % tileNum) vertexInterData["h_t"] = {transpose(vertexSvv)};
            if ((!isClient) && (coTid + 1) % tileNum == tileIndex) vertexInterData["h_t"] = {transpose(vertexSvv)};

            sci::twoPartyGCNMatMul(
                vertexSvv, 
                weight, 
                scaledVertexSvv,
                coTid,
                party
            );
        }
        // The feature is scaled, take care during back layer 

        if ((iter % epochLayerNum) == 0) { // Is the first forward layer
            return;
        }

        sci::twoPartyGCNVectorScale(
            vertexSvv, 
            normalizer, 
            scaledVertexSvv, 
            true,
            coTid, 
            party
        );
    }

    void ScatterComp(
        ShareVecVec& updateSrcSvv, 
        std::vector<uint64_t>& updateSrcOutDeg, 
        std::vector<uint64_t>& updateDstInDeg, 
        ShareVecVec& duplicatedUpdateSvv,
        uint64_t coTid, 
        int party
    ) const {
        TaskComm& clientTaskComm = TaskComm::getClientInstance();
        size_t tileNum = clientTaskComm.getTileNum();
        size_t tileIndex = clientTaskComm.getTileIndex();
        // updateSrcSvv
        // updateSrcOutDeg
        // updateDstInDeg
        // duplicatedUpdateSvv

        // size_t length = updateSrcSvv.size();
        // std::vector<uint64_t> normalizer0(length);
        // std::vector<uint64_t> normalizer1(length);
        // for (int i = 0; i < length; ++i) {
        //     normalizer0[i] = updateSrcOutDeg[i] == 0 ? 0 : CryptoUtil::encodeDoubleAsFixedPoint(pow((double)updateSrcOutDeg[i] + 1, -0.5));
        //     normalizer1[i] = updateDstInDeg[i] == 0 ? 0 : CryptoUtil::encodeDoubleAsFixedPoint(pow((double)updateDstInDeg[i] + 1, -0.5));
        // }

#ifdef GCN_LOG
        // printf(">>>>> Scatter Comp updateSrcSvv: party id %d role %d\n", tileIndex, party);
        // sci::printShareVecVec(updateSrcSvv, coTid, party);
        // printf("<<<<< Scatter Comp updateSrcSvv: party id %d role %d\n", tileIndex, party);        

        // printf(">>>>> Scatter Comp normalizer: party id %d role %d\n", tileIndex, party);
        // sci::printShareVecVec(ShareVecVec(1, normalizer0), coTid, party);
        // sci::printShareVecVec(ShareVecVec(1, normalizer1), coTid, party);
        // printf("<<<<< Scatter Comp normalizer: party id %d role %d\n", tileIndex, party);    
#endif

        // sci::twoPartyGCNVectorScale(
        //     updateSrcSvv, 
        //     normalizer0, 
        //     normalizer1, 
        //     duplicatedUpdateSvv, 
        //     coTid, 
        //     party
        // );
        duplicatedUpdateSvv = updateSrcSvv;

#ifdef GCN_LOG
        // printf(">>>>> Scatter Comp duplicatedUpdateSvv: party id %d role %d\n", tileIndex, party);
        // sci::printShareVecVec(duplicatedUpdateSvv, coTid, party);
        // printf("<<<<< Scatter Comp duplicatedUpdateSvv: party id %d role %d\n", tileIndex, party);
#endif
    }

    void UpdatePreMergeComp(
        ShareVecVec& duplicatedUpdateSvv,
        std::vector<uint64_t>& updateDstVertexPos, 
        uint64_t coTid, 
        int party
    ) const {
        TaskComm& clientTaskComm = TaskComm::getClientInstance();
        size_t tileNum = clientTaskComm.getTileNum();
        size_t tileIndex = clientTaskComm.getTileIndex();
#ifdef GCN_LOG
        printf(">>>>> PreMerge Comp duplicatedUpdateSvv input: party id %d role %d\n", tileIndex, party);
        sci::printShareVecVec(duplicatedUpdateSvv, coTid, party);
        printf("<<<<< PreMerge Comp duplicatedUpdateSvv input: party id %d role %d\n", tileIndex, party);
        // printf(">>>>> PreMerge Comp updateDstVertexPos input: party id %d role %d\n", tileIndex, party);
        // sci::print_vector(updateDstVertexPos);
        // printf("<<<<< PreMerge Comp updateDstVertexPos input: party id %d role %d\n", tileIndex, party);
#endif        
        // duplicatedUpdateSvv
        // updateDstVertexPos
        duplicatedUpdateSvv = prefix_network_aggregate(
            updateDstVertexPos,
            duplicatedUpdateSvv,
            AggregationOp::ADD_AGG,
            coTid,
            party,
            true
        );

#ifdef GCN_LOG
        printf(">>>>> PreMerge Comp duplicatedUpdateSvv output: party id %d role %d\n", tileIndex, party);
        sci::printShareVecVec(duplicatedUpdateSvv, coTid, party);
        printf("<<<<< PreMerge Comp duplicatedUpdateSvv output: party id %d role %d\n", tileIndex, party);
#endif
    }

    UpdateType getScatterTaskResult(const Task& task) const {
        if (!task.finished) {
            throw ResultNotReadyException("Task result not ready yet!\n");
        }
        GCNVectorScale* gvs = (GCNVectorScale*)task.buf;
        GCNUpdate result = GCNUpdate(gvs->embeddingVec);
        return result;      
    }

    bool gather(const GraphGASLite::IterCount& iter, Ptr<VertexType>& dst, const UpdateType& update) const {
        return false;
    }

    struct Task genGatherTask(const GraphGASLite::IterCount& iter, Ptr<VertexType>& dst, const UpdateType& update) const {
        struct Task task;
        return task;        
    }

    struct Task genGatherTask(const ShareVec& vertexData, const ShareVec& update, uint64_t dstId, uint64_t dstTid, bool isDummy=false) const {
        GCNVectorAddition* gva = new GCNVectorAddition();
        gva->writeShareToOperand(vertexData, 0);
        gva->writeShareToOperand(update, 1); // Fix me

        gva->useMask = true;
        gva->operandMask = true;

        struct Task task = {GCN_VECTOR_ADDITION, dstId, false, (void*)gva, static_cast<uint64_t>(-1), dstTid, dstTid};
        task.isDummy = isDummy;
        return task;        
    }

    void GatherComp(
        ShareVecVec& vertexSvv, 
        ShareVecVec& updateSvv, 
        std::vector<bool>& isGatherDstVertexDummy, 
        std::vector<uint64_t>& localVertexInDeg,
        uint64_t iter,
        uint64_t updateSrcTid,
        uint64_t coTid, 
        int party
    ) const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        uint32_t epochLayerNum = getEpochLayerNum();
        uint32_t forwardLayerNum = getForwardLayerNum();
        uint32_t backwardLayerNum = getBackwardLayerNum();
        bool isForward = ((iter % epochLayerNum) < forwardLayerNum); 
        uint32_t coForwardLayer = 0;
        if (isForward) coForwardLayer = iter % epochLayerNum;
        else coForwardLayer = forwardLayerNum - 1 - (((iter % epochLayerNum) - forwardLayerNum) / 2);

        TaskComm& clientTaskComm = TaskComm::getClientInstance();
        size_t tileNum = clientTaskComm.getTileNum();
        size_t tileIndex = clientTaskComm.getTileIndex();

        bool isSigned = true;

#ifdef GCN_LOG
        printf(">>>>> Gather Comp vertexSvv input: party id %d role %d\n", tileIndex, party);
        sci::printShareVecVec(vertexSvv, coTid, party);
        printf("<<<<< Gather Comp vertexSvv input: party id %d role %d\n", tileIndex, party);
        printf(">>>>> Gather Comp updateSvv: party id %d role %d\n", tileIndex, party);
        sci::printShareVecVec(updateSvv, coTid, party);
        printf("<<<<< Gather Comp updateSvv: party id %d role %d\n", tileIndex, party);
        // printf(">>>>> Gather Comp isGatherDstVertexDummy: party id %d role %d\n", tileIndex, party);
        // sci::print_vector(isGatherDstVertexDummy);
        // printf("<<<<< Gather Comp isGatherDstVertexDummy: party id %d role %d\n", tileIndex, party);
#endif            
        size_t length = localVertexInDeg.size();
        // std::vector<uint64_t> normalizer(length);
        // for (int i = 0; i < length; ++i) {
        //     normalizer[i] = localVertexInDeg[i] == 0 ? 0 : CryptoUtil::encodeDoubleAsFixedPoint(pow((double)localVertexInDeg[i] + 1, -0.5));
        // }

        auto t_tmp = std::chrono::high_resolution_clock::now();

        // // If is a backward layer, the vertex scale and update scale are not delayed.
        // // If not, they are delayed to the Apply phase.
        // if ((iter / gnnParam.num_layers) % 2 != 0) {
        //     t_tmp = std::chrono::high_resolution_clock::now();

        //     if (updateSrcTid == 0) {
        //         sci::twoPartyGCNVectorScale(
        //             vertexSvv, 
        //             normalizer, 
        //             vertexSvv, 
        //             isSigned,
        //             coTid, 
        //             party
        //         );
        //     }

        //     if (party == sci::ALICE) print_duration(t_tmp, "vertex-svv-scale");
        //     t_tmp = std::chrono::high_resolution_clock::now();

        //     sci::twoPartyGCNVectorScale(
        //         updateSvv, 
        //         normalizer, 
        //         updateSvv,
        //         isSigned, 
        //         coTid, 
        //         party
        //     );      

        //     if (party == sci::ALICE) print_duration(t_tmp, "update-svv-scale");
        //     t_tmp = std::chrono::high_resolution_clock::now(); 
        // } 

        // vertexSvv
        // updateSvv
        // isGatherDstVertexDummy
        std::vector<bool> cond = isGatherDstVertexDummy;
        for (int i = 0; i < cond.size(); ++i) cond[i] = !cond[i];
        sci::twoPartyGCNCondVectorAddition(
            vertexSvv, 
            updateSvv, 
            cond, 
            vertexSvv,
            coTid, 
            party
        );

        if (party == sci::ALICE) print_duration(t_tmp, "vertex-update-cond-addition");

        // Scale as we have Gathered updates from all parties.
        t_tmp = std::chrono::high_resolution_clock::now();

        if (updateSrcTid == tileNum - 1 && (iter + 1) % epochLayerNum != 0) {
            std::vector<uint64_t> normalizer(length);
            for (int i = 0; i < length; ++i) {
                normalizer[i] = localVertexInDeg[i] == 0 ? 0 : CryptoUtil::encodeDoubleAsFixedPoint(pow((double)localVertexInDeg[i] + 1, -0.5));
            }

            sci::twoPartyGCNVectorScale(
                vertexSvv, 
                normalizer, 
                vertexSvv, 
                isSigned,
                coTid, 
                party
            );
        }   

        if (party == sci::ALICE) print_duration(t_tmp, "vertex-plus-update-svv-scale");
        

#ifdef GCN_LOG
        printf(">>>>> Gather Comp vertexSvv output: party id %d role %d\n", tileIndex, party);
        sci::printShareVecVec(vertexSvv, coTid, party);
        printf("<<<<< Gather Comp vertexSvv output: party id %d role %d\n", tileIndex, party);
#endif
    }

    void writeGatherTaskResult(const Task& task, ShareVec& dst) const {
        if (!task.finished) {
            throw ResultNotReadyException("Task result not ready yet!\n");
        }
        GCNVectorAddition* gva = (GCNVectorAddition*)task.buf;
        dst = gva->operands[0];
    }

    void writeGatherTaskResult(const Task& task, Ptr<VertexType>& dst) const {
    }

    std::vector<Task> genApplyTaskVec(GraphSummary& gs, uint64_t iter, const std::vector<ShareVec>& vertexDataVec, uint64_t dstTid, bool isClient, bool isDummy=false) const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        uint64_t vecSize = vertexDataVec.size();
        std::vector<Task> taskVec;
        
        return taskVec;        
    }

    void ApplyComp(GraphSummary& gs, uint64_t iter, const std::vector<ShareVec>& vertexDataVec, const std::vector<uint64_t>& localVertexInDeg, std::vector<ShareVec>& dstVec, uint64_t tileIndex, uint64_t dstTid, bool isClient) const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        uint32_t epochLayerNum = getEpochLayerNum();
        uint32_t forwardLayerNum = getForwardLayerNum();
        uint32_t backwardLayerNum = getBackwardLayerNum();
        bool isForward = ((iter % epochLayerNum) < forwardLayerNum); 
        uint32_t coForwardLayer = 0;
        if (isForward) coForwardLayer = iter % epochLayerNum;
        else coForwardLayer = forwardLayerNum - 1 - (((iter % epochLayerNum) - forwardLayerNum) / 2);

        uint64_t vecSize = vertexDataVec.size();
        TaskComm& clientTaskComm = TaskComm::getClientInstance();
        TaskComm& serverTaskComm = TaskComm::getServerInstance();
        size_t tileNum = clientTaskComm.getTileNum();
        // printf("H1.0\n");
        // printf("H1.1\n");
        int party;
        if (isClient) party = sci::ALICE;
        else party = sci::BOB; 

        size_t length = localVertexInDeg.size();
        std::vector<uint64_t> normalizer(length);
        for (int i = 0; i < length; ++i) {
            normalizer[i] = localVertexInDeg[i] == 0 ? 0 : CryptoUtil::encodeDoubleAsFixedPoint(pow((double)localVertexInDeg[i] + 1, -0.5));
        }

        if (isForward) { // FORWARD
            const ShareTensor& weight = isClient? gs.localWeight[coForwardLayer]:gs.remoteWeight[coForwardLayer];
            TensorVecMap& vertexInterData = isClient? gs.localVertexInterDataTVs[coForwardLayer]:gs.remoteVertexInterDataTVs[coForwardLayer];

            // printf("H1.2\n");
            if (iter % epochLayerNum != forwardLayerNum - 1) { // GCN_FORWARD_NN
                vertexInterData["z"] = {vertexDataVec};
                ShareTensor new_h;
                sci::twoPartyGCNRelu(vertexDataVec, new_h, dstTid, party);

#ifdef GCN_LOG
                printf(">>>>> Apply Comp forward, z, new_h: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(vertexDataVec, dstTid, party);
                printf("------------\n");
                sci::printShareVecVec(new_h, dstTid, party);
                printf("<<<<< Apply Comp forward, z, new_h: party id %d role %d\n", tileIndex, party);
#endif
                dstVec.swap(new_h);
            } else { // GCN_FORWARD_PREDICTION
                uint64_t trainSetSize = (uint64_t)(vecSize * gnnParam.train_ratio);
                uint64_t valSetSize = (uint64_t)(vecSize * gnnParam.val_ratio);
                uint64_t testSetSize = vecSize - trainSetSize - valSetSize;
#ifdef GCN_LOG
                printf(">> trainSetSize %lu\n", trainSetSize);
                printf(">> valSetSize %lu\n", valSetSize);
                printf(">> testSetSize %lu\n", testSetSize);
#endif
                vertexInterData["z"] = {vertexDataVec};
                vertexInterData["p"] = std::vector<ShareTensor>(1);
                ShareTensor p;
                ShareTensor p_minus_y;
                if (isClient) {
                    ShareVecVec label;
                    for (int i=0; i<vecSize; ++i) {
                        label.push_back(toShareVec(gs.localVertexVec[i]->data().label, gnnParam.num_labels));
                        // printf("vid %lu\n", (uint64_t)gs.localVertexVec[i]->vid());
                    }
                    sci::twoPartyGCNForwardNNPredictionWithoutWeight(vertexDataVec, label, p, p_minus_y, dstTid, party);
#ifdef GCN_LOG
                    printf(">>>>> Apply Comp prediction, input, weight, p: party id %d role %d\n", tileIndex, party);
                    sci::printShareVecVec(vertexDataVec, dstTid, party);
                    printf("------------\n");
                    sci::printShareVecVec(weight, dstTid, party);
                    printf("------------\n");
                    sci::printShareVecVec(p, dstTid, party);
                    printf("<<<<< Apply Comp prediction, input, weight, p: party id %d role %d\n", tileIndex, party);
#endif
                } else {
                    ShareVecVec zero_label;
                    zero_label.resize(vecSize, std::vector<uint64_t>(gnnParam.num_labels, 0));
                    sci::twoPartyGCNForwardNNPredictionWithoutWeight(vertexDataVec, zero_label, p, p_minus_y, dstTid, party);
#ifdef GCN_LOG
                    printf(">>>>> Apply Comp prediction, input, weight, p: party id %d role %d\n", tileIndex, party);
                    sci::printShareVecVec(vertexDataVec, dstTid, party);
                    printf("------------\n");
                    sci::printShareVecVec(weight, dstTid, party);
                    printf("------------\n");
                    sci::printShareVecVec(p, dstTid, party);
                    printf("<<<<< Apply Comp prediction, input, weight, p: party id %d role %d\n", tileIndex, party);
#endif
                }

                DoubleTensor plainP;
                sci::getPlainShareVecVec(p, plainP, dstTid, party);
                if (isClient) {
#ifdef GCN_LOG
                    printf(">>>>> Apply Comp prediction, y, loss, accuracy: party id %d role %d\n", tileIndex, party);
                    sci::print_vector_of_vector(plainP, 10);
                    printf("--------\n");
#endif
                    DoubleTensor y(vecSize, std::vector<double>(gnnParam.num_labels, 0.0));
                    for (int i=0; i<vecSize; ++i) {
                        y[i][gs.localVertexVec[i]->data().label] = 1.0;
                        for (int j=0; j<gnnParam.num_labels; ++j) {
                            if (plainP[i][j] == 0) plainP[i][j] = 0.001;
                        }
                    }
                    // sci::print_vector_of_vector(y);
                    printf("--------\n");
                    printf("cross-entropy-loss = %lf\n", sci::cross_entropy_loss(y, plainP));
                    DoubleTensor trainingY = DoubleTensor(y.begin(), y.begin() + trainSetSize);
                    DoubleTensor testY = DoubleTensor(y.begin() + trainSetSize + valSetSize, y.end());
                    DoubleTensor trainingPlainP = DoubleTensor(plainP.begin(), plainP.begin() + trainSetSize);
                    DoubleTensor testPlainP = DoubleTensor(plainP.begin() + trainSetSize + valSetSize, plainP.end());
                    std::vector<bool> trainingIsBorder = std::vector<bool>(gs.isLocalVertexBorder.begin(), gs.isLocalVertexBorder.begin() + trainSetSize);
                    std::vector<bool> testIsBorder = std::vector<bool>(gs.isLocalVertexBorder.begin() + trainSetSize + valSetSize, gs.isLocalVertexBorder.end());
                    printf("full set accuracy = %lf\n", sci::accuracy(y, plainP));
                    printf("training set accuracy = %lf\n", sci::accuracy(trainingY, trainingPlainP));
                    printf("border training set accuracy = %lf\n", sci::accuracy(trainingY, trainingPlainP, trainingIsBorder));
                    printf("test set accuracy = %lf\n", sci::accuracy(testY, testPlainP));
                    printf("border test set accuracy = %lf\n", sci::accuracy(testY, testPlainP, testIsBorder));
                    printf("the number of vertices is %lu, the number of border vertices is %lu\n", y.size(), sci::count_true(gs.isLocalVertexBorder));
                    // printf("<<<<< Apply Comp prediction, y, loss, accuracy: party id %d role %d\n", tileIndex, party);
                }

                vertexInterData["p"][0].swap(p);
                // Preserve the gradients of training set only
                // printf("trainSetSize %lu, vecSize %lu\n", trainSetSize, vecSize);
                for (int i=trainSetSize; i<vecSize; ++i) {
                    p_minus_y[i] = std::vector<uint64_t>(p_minus_y[0].size(), 0);
                }
                dstVec.swap(p_minus_y);   
            }
        } else { // BACKWARD
            ShareTensor weightT = isClient? gs.localWeight[coForwardLayer]:gs.remoteWeight[coForwardLayer];
            ShareTensor& weightRef = isClient? gs.localWeight[coForwardLayer]:gs.remoteWeight[coForwardLayer];
            ShareTensor& coWeightRef = isClient? gs.remoteWeight[coForwardLayer]:gs.localWeight[coForwardLayer];
            weightT = transpose(weightT);
            TensorVecMap& vertexInterData = isClient? gs.localVertexInterDataTVs[coForwardLayer]:gs.remoteVertexInterDataTVs[coForwardLayer];
            TensorVecMap& coVertexInterData = isClient? gs.remoteVertexInterDataTVs[coForwardLayer]:gs.localVertexInterDataTVs[coForwardLayer];
            bool isFirstOfTwo = (((iter % epochLayerNum) - forwardLayerNum) % 2 == 0);
            if (coForwardLayer == forwardLayerNum - 1) { // two layers of GCN_BACKWARD_NN_INIT
                vertexInterData["d"] = std::vector<ShareTensor>(1);
                ShareTensor d;
                ShareTensor g;
#ifdef GCN_LOG
                printf(">>>>> Apply Comp p_minus_y: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(vertexDataVec, dstTid, party);
                printf("<<<<< Apply Comp p_minus_y: party id %d role %d\n", tileIndex, party);
                printf(">>>>> Apply Comp weight_t: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(weightT, dstTid, party);
                printf("<<<<< Apply Comp weight_t: party id %d role %d\n", tileIndex, party);
#endif
                if (isFirstOfTwo) {
                    sci::twoPartyGCNMatMul(vertexDataVec, weightT, g, dstTid, party);
                    vertexInterData["g"] = {g};
                    dstVec = vertexDataVec;
                    return;
                }

                sci::twoPartyGCNMatMul(vertexInterData["h_t"][0], vertexDataVec, d , dstTid, party);

                uint64_t trainSetSize = (uint64_t)(vecSize * gnnParam.train_ratio);
                double gradientScaler = (double) 1 / trainSetSize;
                // double gradientScaler = 1.0;
                sci::twoPartyGCNMatrixScale(d, static_cast<uint64_t>(gradientScaler * (1<<SCALER_BIT_LENGTH)), d, dstTid, party);

                sci::twoPartyGCNApplyGradient(weightRef, d, static_cast<uint64_t>(gs.learningRate * (1<<SCALER_BIT_LENGTH)), weightRef, dstTid, party);

                // double weightScaler = (double) 1 / tileNum;
                // sci::twoPartyGCNMatrixScale(weightRef, static_cast<uint64_t>(weightScaler * (1<<SCALER_BIT_LENGTH)), weightRef, dstTid, party);

                vertexInterData["d"][0].swap(d);
                dstVec.swap(vertexInterData["g"][0]);
#ifdef GCN_LOG
                printf(">>>>> Apply Comp d: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(vertexInterData["d"][0], dstTid, party);
                printf("<<<<< Apply Comp d: party id %d role %d\n", tileIndex, party);
#endif
            } else { // GCN_BACKWARD_NN
                vertexInterData["d"] = std::vector<ShareTensor>(1);
                ShareTensor d;
                ShareTensor g;
#ifdef GCN_LOG
                printf(">>>>> Apply Comp p_minus_y: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(vertexDataVec, dstTid, party);
                printf("<<<<< Apply Comp p_minus_y: party id %d role %d\n", tileIndex, party);
                printf(">>>>> Apply Comp weight_t: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(weightT, dstTid, party);
                printf("<<<<< Apply Comp weight_t: party id %d role %d\n", tileIndex, party);
#endif
                bool isFirstLayer = false;
                if (coForwardLayer == 0) isFirstLayer = true; 
                if (isFirstOfTwo) {
                    sci::twoPartyGCNBackwardNNWithoutAH(vertexDataVec, vertexInterData["z"][0], weightT, dstVec, g, isFirstLayer, dstTid, party);
                    vertexInterData["g"] = {g};
                    return;
                }

                sci::twoPartyGCNMatMul(vertexInterData["h_t"][0], vertexDataVec, d , dstTid, party);
#ifdef GCN_LOG
                printf(">>>>> Apply Comp d: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(d, dstTid, party);
                printf("<<<<< Apply Comp d: party id %d role %d\n", tileIndex, party);
                printf(">>>>> Apply Comp g: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(vertexInterData["g"][0], dstTid, party);
                printf("<<<<< Apply Comp g: party id %d role %d\n", tileIndex, party);
#endif

                uint64_t trainSetSize = (uint64_t)(vecSize * gnnParam.train_ratio);
                double gradientScaler = (double) 1 / trainSetSize;
                // double gradientScaler = 1.0;
                sci::twoPartyGCNMatrixScale(d, static_cast<uint64_t>(gradientScaler * (1<<SCALER_BIT_LENGTH)), d, dstTid, party);

#ifdef GCN_LOG
                printf(">>>>> Apply Comp weight, d, new weight: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(weightRef, dstTid, party);
                printf("---------\n");
#endif
                sci::twoPartyGCNApplyGradient(weightRef, d, static_cast<uint64_t>(gs.learningRate * (1<<SCALER_BIT_LENGTH)), weightRef, dstTid, party);

                // double weightScaler = (double) 1 / tileNum;
                // sci::twoPartyGCNMatrixScale(weightRef, static_cast<uint64_t>(weightScaler * (1<<SCALER_BIT_LENGTH)), weightRef, dstTid, party);

                vertexInterData["d"][0].swap(d);
                dstVec.swap(vertexInterData["g"][0]);

#ifdef GCN_LOG
                sci::printShareVecVec(vertexInterData["d"][0], dstTid, party);
                printf("---------\n");
                sci::printShareVecVec(weightRef, dstTid, party);
                printf("---------\n");
                printf("<<<<< Apply Comp weight, d, new weight: party id %d role %d\n", tileIndex, party);
#endif
            }

            Semaphore& remote_weight_ready_smp = serverTaskComm.getRemoteWeightReadySmp();
            Semaphore& weight_avg_finished_smp = clientTaskComm.getWeightAvgFinishedSmp();
            if (isClient) {
                remote_weight_ready_smp.acquire();
                // ShareVecVec debugD;
                std::vector<ShareVecVec> weightFromTheOtherParties(tileNum);
                if (tileIndex == 0 || tileIndex == 1) {
                    for (int i = 0; i < tileNum; ++i) {
                        if (i != tileIndex && i != 1-tileIndex) {
                            if (tileIndex == 0) clientTaskComm.recvShareVecVec(weightFromTheOtherParties[i], i);
                            if (tileIndex == 1) serverTaskComm.recvShareVecVec(weightFromTheOtherParties[i], i);
                            sci::plaintext_add_matrix_in_place(weightRef, weightFromTheOtherParties[i]);
                        }
                    }
                    // debugD = sci::plaintext_add_matrix(vertexInterData["d"][0], coVertexInterData["d"][0]);
                    sci::plaintext_add_matrix_in_place(weightRef, coWeightRef);
                    double weightScaler = (double) 1 / tileNum;
                    sci::twoPartyGCNMatrixScale(weightRef, static_cast<uint64_t>(weightScaler * (1<<SCALER_BIT_LENGTH)), weightRef, 1-tileIndex, tileIndex + 1);
                    coWeightRef = weightRef;

                    for (int i = 0; i < tileNum; ++i) {
                        if (i != tileIndex && i != 1-tileIndex) {
                            if (tileIndex == 0) clientTaskComm.sendShareVecVec(weightRef, i);
                            if (tileIndex == 1) serverTaskComm.sendShareVecVec(weightRef, i);
                        }
                    }
                } else {
                    clientTaskComm.sendShareVecVec(weightRef, 1);
                    serverTaskComm.sendShareVecVec(coWeightRef, 0);
                    clientTaskComm.recvShareVecVec(weightRef, 1);
                    serverTaskComm.recvShareVecVec(coWeightRef, 0);
                }

                weight_avg_finished_smp.release();

#ifdef GCN_LOG
                // printf(">>>>> Apply Comp added d: party id %d role %d\n", tileIndex, party);
                // sci::printShareVecVec(debugD, dstTid, party);
                // printf("<<<<< Apply Comp added d: party id %d role %d\n", tileIndex, party);
                printf(">>>>> Apply Comp added weight: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(coWeightRef, dstTid, party);
                printf("<<<<< Apply Comp added weight: party id %d role %d\n", tileIndex, party);
#endif
            } else {
                remote_weight_ready_smp.release();
                weight_avg_finished_smp.acquire();
                // ShareVecVec debugD = sci::plaintext_add_matrix(vertexInterData["d"][0], coVertexInterData["d"][0]);
#ifdef GCN_LOG
                // printf(">>>>> Apply Comp added d: party id %d role %d\n", tileIndex, party);
                // sci::printShareVecVec(debugD, dstTid, party);
                // printf("<<<<< Apply Comp added d: party id %d role %d\n", tileIndex, party);
                printf(">>>>> Apply Comp added weight: party id %d role %d\n", tileIndex, party);
                sci::printShareVecVec(coWeightRef, dstTid, party);
                printf("<<<<< Apply Comp added weight: party id %d role %d\n", tileIndex, party);
#endif
            }

        } 

#ifdef GCN_LOG
        printf(">>>>> Apply Comp: party id %d role %d\n", tileIndex, party);
        sci::printShareVecVec(dstVec, dstTid, party);
        printf("<<<<< Apply Comp: party id %d role %d\n", tileIndex, party);    
#endif
    }

    void writeApplyTaskVecResult(GraphSummary& gs, uint64_t iter, const std::vector<Task>& taskVec, std::vector<ShareVec>& dstVec, bool isClient) const {
    }

    void onAlgoKernelStart(Ptr<GraphTileType>& graph) const {
    }

    static std::vector<double> normalizeFeatureVec(const std::vector<double>& raw, double inDeg) {
        // Compute the sum of the features of the node
        double sum = 0.0;
        size_t n_features = raw.size();
        for (int j = 0; j < n_features; j++) {
            sum += raw[j];
        }
        if (sum == 0.0) sum = 1.0;
        // Normalize the features by dividing by the sum
        std::vector<double> normalized(n_features, 0.0);
        for (int j = 0; j < n_features; j++) {
            // normalized[j] = raw[j] / sum;
            normalized[j] = raw[j];
            normalized[j] *= pow(inDeg + 1.0, -0.5);
        }      
        return normalized;  
    }

    // GCN weight initialization function using the Glorot method
    static std::vector<std::vector<double>> initWeight(int dim0, int dim1) {
        std::vector<std::vector<double>> W(dim0, std::vector<double>(dim1, 0.0));

        std::srand(42);

        double limit = std::sqrt(6.0 / (dim0 + dim1));

        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                W[i][j] = (double) std::rand() / RAND_MAX * 2 * limit - limit;
            }
        }

        return W;
    }

    void onAlgoKernelStart(Ptr<GraphTileType>& graph, GraphSummary& gs) const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        // Feature normalization
        for (auto vIter = graph->vertexIter(); vIter != graph->vertexIterEnd(); ++vIter) {
            auto v = vIter->second;
            auto& data = v->data();
            double inDeg = (double)(v->inDeg().cnt());
            data.feature = normalizeFeatureVec(data.feature, inDeg);
        }

        // Initilize weight matrix
        gs.plainWeight.resize(gnnParam.num_layers);
        // gs.plainWeight[0].resize(gnnParam.input_dim);
        // for (int i=0; i<gs.plainWeight[0].size(); ++i) {
        //     gs.plainWeight[0][i].resize(gnnParam.hidden_dim, 0.5);
        // }
        // gs.plainWeight[1].resize(gnnParam.hidden_dim);
        // for (int i=0; i<gs.plainWeight[1].size(); ++i) {
        //     gs.plainWeight[1][i].resize(gnnParam.num_labels, 0.5);
        // }
        gs.plainWeight[0] = initWeight(gnnParam.input_dim, gnnParam.hidden_dim);
        gs.plainWeight[1] = initWeight(gnnParam.hidden_dim, gnnParam.num_labels);

        gs.localWeight.resize(gnnParam.num_layers);
        gs.remoteWeight.resize(gnnParam.num_layers);
        // printf("H1\n");
        intoShareTensor(gs.plainWeight[0], gs.localWeight[0], gs.remoteWeight[0]);
        // printf("H2\n");
        intoShareTensor(gs.plainWeight[1], gs.localWeight[1], gs.remoteWeight[1]);
        // printf("H3\n");

        gs.learningRate = gnnParam.learning_rate;
        gs.globalNumSamples = gnnParam.num_samples;
    }

    void onAlgoKernelEnd(Ptr<GraphTileType>& graph) const {
        // Nothing to do
    }

    uint32_t getPlainNumPerOperand() const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        return std::max(gnnParam.input_dim, gnnParam.hidden_dim); // Fix me
    }

    uint32_t getPlainNumPerOperand(uint64_t layer) const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        layer = layer % ((1 + 2)*gnnParam.num_layers);
        uint32_t ret = 0;
        switch (layer) {
            case 0:
                ret = gnnParam.hidden_dim;
                break;
            case 1:
                ret = gnnParam.num_labels;
                break;
            case 2:
                ret = gnnParam.num_labels;
                break;
            case 3:
                ret = gnnParam.num_labels;
                break;
            case 4:
                ret = gnnParam.hidden_dim;
                break;
            case 5:
                ret = gnnParam.hidden_dim;
                break;
            default:
                printf("Illegal layer in getPlainNumPerOperand\n");
                exit(-1);                
        }

        return ret;
    }

    uint32_t getForwardLayerNum() const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        return gnnParam.num_layers;
    }

    uint32_t getBackwardLayerNum() const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        return 2 * gnnParam.num_layers;
    }

    uint32_t getEpochLayerNum() const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        return 3 * gnnParam.num_layers;
    }

    std::vector<uint32_t> getDimensionVec() const {
        GNNParam& gnnParam = GNNParam::getGNNParam();
        std::vector<uint32_t> dimensions = {gnnParam.hidden_dim, gnnParam.num_labels, 0, gnnParam.num_labels, gnnParam.hidden_dim, gnnParam.hidden_dim};
        return dimensions;
    }

protected:
    GCNEdgeCentricAlgoKernel(const string& name, const GraphGASLite::VertexIdx& src)
        : GraphGASLite::SSEdgeCentricAlgoKernel<GraphTileType>(name),
          src_(src)
    {
        // Nothing else to do.
    }

private:
    const GraphGASLite::VertexIdx src_;
};

#endif // ALGO_KERNELS_EDGE_CENTRIC_GCN_GCN_H_

