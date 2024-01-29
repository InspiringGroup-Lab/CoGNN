#ifndef KERNEL_HARNESS_H_
#define KERNEL_HARNESS_H_

#include "harness.h"
#include "gcn.h"

#include <sstream>

typedef GraphGASLite::GraphTile<GCNData, GCNUpdate> Graph;
typedef GCNEdgeCentricAlgoKernel<Graph> Kernel;

const char appName[] = "gcn";

class AppArgs : public GenericArgs<uint64_t> {
public:
    AppArgs() : GenericArgs<uint64_t>() {
        std::get<0>(argTuple_) = srcDefault;
    };

    const ArgInfo* argInfoList() const {
        static const ArgInfo list[] = {
            {"", "[src]", "Source vertex index (default " + std::to_string(srcDefault) + ")."},
        };
        return list;
    }

private:
    static constexpr uint64_t srcDefault = 0;
};

// #define VDATA(vd) \
//     (vd.distance >= INF(uint32_t) || vd.predecessor == INV_VID ? "Can not reach <- none": \
//     std::to_string(vd.distance) + \
//     "\t<- " + \
//     std::to_string(vd.predecessor))

void readVertexDataLine(std::istringstream& iss, GCNData& data) {
    GNNParam& gnnParam = GNNParam::getGNNParam();
    data.feature.resize(gnnParam.input_dim);
    for (int i=0; i<gnnParam.input_dim; ++i) {
        iss >> data.feature[i];
    }
    iss >> data.label;
}

#endif // KERNEL_HARNESS_H_
