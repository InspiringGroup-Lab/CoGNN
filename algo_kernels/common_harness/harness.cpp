#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <vector>
#include <condition_variable>
#include <chrono>
#include <ctime>
#include "engine.h"
#include "graph_io_util.h"
#include "graph.h"
#include "graph_common.h"

// Kernel harness header.
#include "kernel_harness.h"

#include "TaskqHandler.h"
#include "TaskUtil.h"

template<typename GraphTileType>
void loadVertexData(Ptr<GraphTileType> graph, const string& vertexDataFileName) {
    std::ifstream infile(vertexDataFileName, std::ifstream::in);
    if (!infile.is_open()) {
        throw FileException(vertexDataFileName);
    }
    string line;
    // int cnt = 0;
    while (GraphGASLite::GraphIOUtil::nextEffectiveLine(infile, line)) {
        // printf("line %d\n", cnt);
        std::istringstream iss(line);
        uint64_t vid = 0;

        if (!(iss >> vid)) {
            throw FileException(vertexDataFileName);
        }

        if (graph->hasVertex(vid)) {
            // Local destination.
            auto v = graph->vertex(vid);
            readVertexDataLine(iss, v->data());
        }
        // } else {
        //     printf("%lld\n", vid);
        //     throw FileException(vertexDataFileName);
        // }
    }
}

int main(int argc, char* argv[]) {
    /* Parse arguments. */

    size_t threadCount;
    size_t graphTileCount;
    size_t tileIndex; // The tile number is actually the thread number
    uint64_t maxIters;
    uint32_t numParts;
    bool undirected;
    bool noPreprocess;
    bool isCluster;
    bool isNoDummyEdge;

    std::string edgelistFile;
    std::string vertexlistFile;
    std::string partitionFile;
    std::string outputFile;
    std::string setting;
    std::string GNNConfigFile;

    AppArgs appArgs;

    int argRet = algoKernelArgs(argc, argv,
            threadCount, graphTileCount, tileIndex, maxIters, numParts, setting, noPreprocess, isCluster, isNoDummyEdge, undirected,
            edgelistFile, vertexlistFile, partitionFile, outputFile, GNNConfigFile, appArgs);

    if (argRet) {
        algoKernelArgsPrintHelp(appName, appArgs);
        return argRet;
    }

	// /* Load and initialize the signed enclave */
	// if (open_enclave(tileIndex) != 0) {
	// 	return -1;
	// }

    GNNParam& gnnParam = GNNParam::getGNNParam();
    gnnParam.readConfig(GNNConfigFile);

    std::unordered_map<GraphGASLite::VertexIdx, GraphGASLite::TileIdx, std::hash<GraphGASLite::VertexIdx::Type> > tidMap;

    /* Make engine and load input. */

    GraphGASLite::Engine<Graph> engine;
    engine.graphTileIs(GraphGASLite::GraphIOUtil::graphTilesFromEdgeList<Graph>(
                threadCount, tileIndex, edgelistFile, partitionFile, 1, undirected, graphTileCount/threadCount, true, tidMap));
    
    engine.tileIndexIs(tileIndex);

    loadVertexData<Graph>(engine.graphTile(tileIndex), vertexlistFile);

    std::cout << "Graph loaded from " << edgelistFile <<
        (partitionFile.empty() ? "" : string(" and ") + partitionFile) <<
        " with " << graphTileCount << " graph tiles, " << 
        "into " << threadCount << " tiles." <<
        " Treated as " << (undirected ? "undirected" : "directed") << " graph." <<
        "Current tile is the No." << tileIndex << " tile." <<
        std::endl;

    /* Make algorithm kernel. */

    auto kernel = appArgs.algoKernel<Kernel>(appName);
    kernel->verboseIs(true);
    kernel->maxItersIs(maxIters);
    kernel->numPartsIs(numParts);
    kernel->tidMapIs(tidMap);
    kernel->curTidIs(tileIndex);
    engine.algoKernelNew(kernel);

    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    cryptoUtil.tileIndexIs(tileIndex);
    cryptoUtil.setUpPaillierCipher();
    // #ifdef USE_FHE
    // cryptoUtil.setUpFHECipher();
    // #endif

    TaskComm& clientTaskComm = TaskComm::getClientInstance();
    clientTaskComm.tileNumIs(threadCount);
    clientTaskComm.tileIndexIs(tileIndex);
    clientTaskComm.settingIs(setting);
    clientTaskComm.noPreprocessIs(noPreprocess);
    clientTaskComm.isClusterIs(isCluster);
    clientTaskComm.isNoDummyEdgeIs(isNoDummyEdge);

    TaskComm& serverTaskComm = TaskComm::getServerInstance();
    serverTaskComm.tileNumIs(threadCount);
    serverTaskComm.tileIndexIs(tileIndex);
    serverTaskComm.settingIs(setting);
    serverTaskComm.noPreprocessIs(noPreprocess);
    serverTaskComm.isClusterIs(isCluster);
    serverTaskComm.isNoDummyEdgeIs(isNoDummyEdge);

    // sci::setUpSCIChannel();

    std::thread clientSetupThread([&clientTaskComm](){
        clientTaskComm.setUp(true);
    });

    std::thread serverSetupThread([&serverTaskComm](){
        serverTaskComm.setUp(false);
    });

    clientSetupThread.join();
    serverSetupThread.join();

    sci::setUpSCIChannel();
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < threadCount; ++i) {
        if (i != tileIndex) {
            threads.emplace_back([i] () {
                set_up_mpc_channel(true, i);
            });
            threads.emplace_back([i] () {
                set_up_mpc_channel(false, i);
            });
        }
    }
    for (auto& thrd : threads) thrd.join();
    threads.clear();


    std::cout << "Algorithm kernel named " << appName <<
        " is " << algoKernelTagName(kernel->tag()) << ", " <<
        "with max iterations " << maxIters << " and number of partitions " << numParts << "." <<
        std::endl;

    std::cout << "Application parameters: " << appArgs << "." << std::endl;

    /* Run. */

    engine();

    /* Output. */
#ifdef VDATA
    if (!outputFile.empty()) {
        std::cout << "Output to " << outputFile << "." << std::endl;
        std::ofstream ofs(outputFile);
        auto g = engine.graphTile(tileIndex);
        for (auto vIter = g->vertexIter(); vIter != g->vertexIterEnd(); ++vIter) {
            auto v = vIter->second;
            ofs << v->vid() << "\t" << VDATA(v->data()) << std::endl;
        }
        ofs.close();
    }
#endif

    // close_enclave();
    clientTaskComm.closeChannels();

    // std::vector<std::condition_variable>& client_computeTaskq_cv = clientTaskComm.getClientComputeTaskqCv();
	// for (int i = 0; i < threadCount; i++) {
	// 	if (i != tileIndex) {
    //         client_computeTaskq_cv[i].notify_one();
    //     }
    // }

    // for (auto& thrd : client_computeTaskq_threads)
    //     thrd.join();

    serverTaskComm.closeChannels();

    return 0;
}

