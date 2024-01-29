#include "gtest/gtest.h"
#include "engine.h"
#include "graph_io_util.h"
#include "test_graph_types.h"

using namespace GraphGASLite;

class AK1 : public EdgeCentricAlgoKernel<TestGraphTile> {
public:
    AK1() : EdgeCentricAlgoKernel<TestGraphTile>("ak1") {
        maxItersIs(10);
    }
protected:
    std::pair<TestUpdate, bool> scatter(const IterCount&, Ptr<VertexType>&, EdgeWeightType&) const {
        return std::make_pair<TestUpdate, bool>(0, false);
    }
    bool gather(const IterCount&, Ptr<VertexType>&, const TestUpdate&) const {
        return false;
    }
};

class AK2 : public EdgeCentricAlgoKernel<TestGraphTile> {
public:
    AK2() : EdgeCentricAlgoKernel<TestGraphTile>("ak2") {
        maxItersIs(15);
    }
protected:
    std::pair<TestUpdate, bool> scatter(const IterCount&, Ptr<VertexType>&, EdgeWeightType&) const {
        return std::make_pair<TestUpdate, bool>(0, false);
    }
    bool gather(const IterCount&, Ptr<VertexType>&, const TestUpdate&) const {
        return false;
    }
};

class EngineTest : public ::testing::Test {
public:
    typedef Engine<TestGraphTile> EngineType;

protected:
    virtual void SetUp() {
        engine_ = new EngineType();
        engine_->graphTileIs(GraphIOUtil::graphTilesFromEdgeList<TestGraphTile>(
                    2, "test_graphs/small.dat", "test_graphs/small.part", 0, false, 1, true, 0));
        engine_->algoKernelNew(Ptr<AK1>(new AK1));
        engine_->algoKernelNew(Ptr<AK2>(new AK2));
    }

    virtual void TearDown() {
        delete engine_;
    }

    EngineType* engine_;
};

TEST_F(EngineTest, graphTile) {
    ASSERT_EQ(2, engine_->graphTileCount());
    for (size_t idx = 0; idx < engine_->graphTileCount(); idx++) {
        auto g = engine_->graphTile(idx);
        ASSERT_EQ(idx, static_cast<size_t>(g->tid()));
    }
}

TEST_F(EngineTest, graphTileNew) {
    auto count = engine_->graphTileCount();
    engine_->graphTileNew(Ptr<TestGraphTile>(new TestGraphTile(count)));
    ASSERT_EQ(count + 1, engine_->graphTileCount());
}

TEST_F(EngineTest, graphTileNewInvalidTid) {
    auto count = engine_->graphTileCount();
    try {
        engine_->graphTileNew(Ptr<TestGraphTile>(new TestGraphTile(5)));
    } catch (InvalidArgumentException& e) {
        ASSERT_EQ(count, engine_->graphTileCount());
        return;
    }

    // Never reached.
    ASSERT_TRUE(false);
}

TEST_F(EngineTest, graphTileIs) {
    std::vector<Ptr<TestGraphTile>> tiles{Ptr<TestGraphTile>(new TestGraphTile(0))};
    engine_->graphTileIs(tiles);
    ASSERT_EQ(1, engine_->graphTileCount());
    ASSERT_EQ(1, tiles.size());
}

TEST_F(EngineTest, graphTileIsInvalidTid) {
    auto count = engine_->graphTileCount();
    std::vector<Ptr<TestGraphTile>> tiles{Ptr<TestGraphTile>(new TestGraphTile(5))};
    try {
        engine_->graphTileIs(tiles);
    } catch (InvalidArgumentException& e) {
        ASSERT_EQ(count, engine_->graphTileCount());
        return;
    }

    // Never reached.
    ASSERT_TRUE(false);
}

TEST_F(EngineTest, algoKernel) {
    size_t count = 0;
    for (auto akIter = engine_->algoKernelConstIter(); akIter != engine_->algoKernelConstIterEnd(); ++akIter) {
        count++;
    }
    ASSERT_EQ(2, count);
    ASSERT_EQ(2, engine_->algoKernelCount());
}

TEST_F(EngineTest, algoKernelDel) {
    auto akIter = engine_->algoKernelIter();
    while (akIter != engine_->algoKernelIterEnd()) {
        auto name = (*akIter)->name();
        if (name == "ak1") {
            akIter = engine_->algoKernelDel(akIter);
            continue;
        }
        ++akIter;
    }

    ASSERT_EQ(1, engine_->algoKernelCount());
    for (auto akIter = engine_->algoKernelIter(); akIter != engine_->algoKernelIterEnd(); ++akIter) {
        ASSERT_FALSE((*akIter)->name() == "ak1");
    }
}

