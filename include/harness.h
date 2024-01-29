#ifndef HARNESS_H_
#define HARNESS_H_

#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include "graph_common.h"


/**
 * Argument information.
 */
struct ArgInfo {
    string optchar;
    string placeholder;
    string help;
};

constexpr uint64_t maxItersDefault = 1000;
constexpr uint32_t numPartsDefault = 16;

const ArgInfo optInfoList[] = {
    {"-t", "<threads>", "Number of threads (required)."},
    {"-g", "<gtiles>", "Number of graph tiles (required). Should be multiplier of threads."},
    {"-m", "[maxiter]", "Maximum iteration number (default " + std::to_string(maxItersDefault) + ")."},
    {"-p", "[numParts]", "Number of partitions per thread (default " + std::to_string(numPartsDefault) + ")."},
    {"-u", "", "Undirected graph (default directed)."},
    {"-h", "", "Print this help message."},
};

const ArgInfo comArgInfoList[] = {
    {"", "<edgelistFile>", "Input graph edge list file path (required)."},
    {"", "[partitionFile]", "Input graph partition file path."},
    {"", "[outputFile]", "Output result file path."},
};


/**
 * Print command line arguments help message.
 */
template <typename AppArgs>
void algoKernelArgsPrintHelp(const char* appName, const AppArgs& appArgs) {
    const auto appArgInfoList = appArgs.argInfoList();

    std::cerr << std::endl;

    std::cerr << "Usage: " << appName;
    std::cerr << ' ' << "[options]";
    for (const auto& comArgInfo : comArgInfoList) std::cerr << ' ' << comArgInfo.placeholder;
    for (size_t idx = 0; idx < appArgs.argCount; idx++) std::cerr << ' ' << appArgInfoList[idx].placeholder;
    std::cerr << std::endl;

    constexpr int w = 15;

    std::cerr << "Options:" << std::endl;
    for (const auto& optInfo : optInfoList) {
        std::cerr << '\t' << optInfo.optchar << ' '
            << std::left << std::setw(w) << optInfo.placeholder << ' '
            << optInfo.help << std::endl;
    }
    std::cerr << std::endl;

    std::cerr << "Common arguments:" << std::endl;
    for (const auto& comArgInfo : comArgInfoList) {
        assert(comArgInfo.optchar.empty());
        std::cerr << '\t'
            << std::left << std::setw(w + 3) << comArgInfo.placeholder << ' '
            << comArgInfo.help << std::endl;
    }
    std::cerr << std::endl;

    std::cerr << "App-specific arguments:" << std::endl;
    for (size_t idx = 0; idx < appArgs.argCount; idx++) {
        const auto& appArgInfo = appArgInfoList[idx];
        assert(appArgInfo.optchar.empty());
        std::cerr << '\t'
            << std::left << std::setw(w + 3) << appArgInfo.placeholder << ' '
            << appArgInfo.help << std::endl;
    }
    std::cerr << std::endl;
}


/**
 * Parse command line arguments to algo kernel parameters.
 */
template <typename AppArgs>
int algoKernelArgs(int argc, char** argv,
        size_t& threadCount, size_t& graphTileCount, size_t& tileIndex,
        uint64_t& maxIters, uint32_t& numParts, string& setting, bool& noPreprocess, bool& isCluster, bool& isNoDummyEdge, bool& undirected,
        string& edgelistFile, string& vertexlistFile, string& partitionFile, string& outputFile, string& GNNConfigFile,
        AppArgs& appArgs) {

    threadCount = 0;
    graphTileCount = 0;
    tileIndex = 0;
    maxIters = maxItersDefault;
    numParts = numPartsDefault;
    noPreprocess = false;
    uint32_t noPreprocessFlag; 
    uint32_t isClusterFlag; 
    uint32_t isNoDummyEdgeFlag;
    undirected = false;
    isCluster = false;
    isNoDummyEdge = false;

    edgelistFile = "";
    vertexlistFile = "";
    partitionFile = "";
    outputFile = "";
    GNNConfigFile = "";
    setting = "";

    appArgs = AppArgs();

    /* Common options. */

    int ch;
    opterr = 0; // Reset potential previous errors.
    while ((ch = getopt(argc, argv, "t:g:i:m:p:s:n:c:r:uh")) != -1) {
        switch (ch) {
            case 't':
                std::stringstream(optarg) >> threadCount;
                break;
            case 'g':
                std::stringstream(optarg) >> graphTileCount;
                break;
            case 'i':
                std::stringstream(optarg) >> tileIndex;
                break;
            case 'm':
                std::stringstream(optarg) >> maxIters;
                break;
            case 'p':
                std::stringstream(optarg) >> numParts;
                break;
            case 's':
                std::stringstream(optarg) >> setting;
            case 'n':
                std::stringstream(optarg) >> noPreprocessFlag;
                if (noPreprocessFlag == 1)
                    noPreprocess = true;
                break;
            case 'c':
                std::stringstream(optarg) >> isClusterFlag;
                if (isClusterFlag == 1)
                    isCluster = true;
                break;
            case 'r':
                std::stringstream(optarg) >> isNoDummyEdgeFlag;
                if (isNoDummyEdgeFlag == 1)
                    isNoDummyEdge = true;
                break;
            case 'u':
                undirected = true;
                break;
            case 'h':
            default:
                return -1;
        }
    }

    if (threadCount == 0 || graphTileCount == 0) {
        std::cerr << "Must specify number of threads and number of graph tiles." << std::endl;
        return -1;
    }
    if (graphTileCount % threadCount != 0) {
        std::cerr << "Number of threads must be a divisor of number of graph tiles." << std::endl;
        return -1;
    }

    argc -= optind;
    argv += optind;

    /* Common arguments. */

    if (argc < 1) {
        std::cerr << "Must specify an input edge list file." << std::endl;
        return -1;
    }
    edgelistFile = argv[0];
    argc -= 1;
    argv += 1;

    vertexlistFile = argv[0];
    argc -= 1;
    argv += 1;

    if (argc > 0) {
        partitionFile = argv[0];
        argc -= 1;
        argv += 1;
    }

    if (argc > 0) {
        outputFile = argv[0];
        argc -= 1;
        argv += 1;
    }

    if (argc > 0) {
        GNNConfigFile = argv[0];
        argc -= 1;
        argv += 1;
    }

    /* App-specific arguments. */

    appArgs.argIs(argc, argv);

    if (!appArgs.isValid()) {
        std::cerr << "Invalid app-specific argument." << std::endl;
        return -1;
    }

    return 0;
}


/**
 * Generic arguments.
 */
template<typename... ArgTypes>
class GenericArgs {
public:
    static constexpr size_t argCount = sizeof...(ArgTypes);

    const std::tuple<ArgTypes...>& argTuple() const { return argTuple_; }

    template<size_t N>
    const typename std::tuple_element<N, std::tuple<ArgTypes...>>::type arg() const {
        return std::get<N>(argTuple_);
    }

    friend std::ostream& operator<<(std::ostream& os, const GenericArgs& args) {
        StreamOutputFunc func(os);
        args.foreach(func);
        return os;
    }

    template<typename KernelType>
    Ptr<KernelType> algoKernel(const string& kernelName) {
        return dispatch(KernelType::instanceNew, typename GenSeq<sizeof...(ArgTypes)>::Type(), kernelName);
    }

    virtual void argIs(int argc, char* argv[]) {
        ArgReadFunc func(argc, argv);
        foreach(func);
    }

    virtual const ArgInfo* argInfoList() const = 0;

    virtual bool isValid() const { return true; }

protected:
    std::tuple<ArgTypes...> argTuple_;

protected:
    /**
     * Unpack tuple to variadic function arguments for function dispatch.
     *
     * http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
     */
    template<int ...>
    struct Seq { };

    template<int N, int... S>
    struct GenSeq : GenSeq<N-1, N-1, S...> { };

    template<int... S>
    struct GenSeq<0, S...> { typedef Seq<S...> Type; };

    // Be careful about the position of BoundArgs. Generally variadic pack is at the end.
    template<typename F, int... S, typename... BoundArgs>
    auto dispatch(F func, Seq<S...>, BoundArgs... bndArgs)
    ->decltype(func(bndArgs..., std::get<S>(argTuple_)...)) {
        return func(bndArgs..., std::get<S>(argTuple_)...);
    }


    /**
     * foreach semantic: apply functor to each of the tuple element.
     *
     * http://stackoverflow.com/questions/1198260/iterate-over-tuple
     */
    template<size_t I = 0, typename F>
    typename std::enable_if<I <  sizeof...(ArgTypes), void>::type
    foreach(F& func) {
        func(std::get<I>(argTuple_));
        foreach<I+1, F>(func);
    }

    template<size_t I = 0, typename F>
    typename std::enable_if<I >= sizeof...(ArgTypes), void>::type
    foreach(F&) { }

    template<size_t I = 0, typename F>
    typename std::enable_if<I <  sizeof...(ArgTypes), void>::type
    foreach(F& func) const {
        func(std::get<I>(argTuple_));
        foreach<I+1, F>(func);
    }

    template<size_t I = 0, typename F>
    typename std::enable_if<I >= sizeof...(ArgTypes), void>::type
    foreach(F&) const { }

protected:
    /**
     * Functors for foreach.
     */

    struct StreamOutputFunc {
    private:
        std::ostream* os_;
        int idx_;

    public:
        StreamOutputFunc(std::ostream& os) : os_(&os), idx_(0) { }

        template<typename T>
        void operator()(const T& arg) {
            (*os_) << (idx_ == 0 ? "" : ", ") << arg;
            idx_++;
        }
    };

    struct ArgReadFunc {
    private:
        const int argc_;
        std::vector<char*> argv_;
        int idx_;

    public:
        ArgReadFunc(const int argc, char* const argv[])
            : argc_(argc), idx_(0)
        {
            for (int i = 0; i < argc_; i++) {
                argv_.push_back(argv[i]);
            }
        }

        template<typename T>
        void operator()(T& arg) {
            if (idx_ >= argc_) return;
            std::stringstream(argv_[idx_]) >> arg;
            idx_++;
        }
    };
};


#endif // HARNESS_H_