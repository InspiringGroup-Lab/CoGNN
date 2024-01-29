#include "gtest/gtest.h"
#include "graph.h"
#include "graph_io_util.h"
#include "test_graph_types.h"

using namespace GraphGASLite;

class GraphTest : public ::testing::Test {
public:
    typedef TestGraphTile::VertexType TestVertexType;

protected:
    virtual void SetUp() {
        graphs_ = GraphIOUtil::graphTilesFromEdgeList<TestGraphTile>(
                    2, "test_graphs/small.dat", "test_graphs/small.part", 0, false, 1, false, 0);
        // Do not finalize the graph.
    }

    void degreeSync() {
        for (auto& g : graphs_) {
            for (auto mvIter = g->mirrorVertexIter(); mvIter != g->mirrorVertexIterEnd(); ++mvIter) {
                auto mv = mvIter->second;
                auto vid = mv->vid();
                auto masterTid = mv->masterTileId();
                graphs_[masterTid]->vertex(vid)->inDegInc(mv->accDeg());
                mv->accDegDel();
            }
        }
    }

    std::vector<Ptr<TestGraphTile>> graphs_;
};

TEST_F(GraphTest, tid) {
    ASSERT_EQ(0, static_cast<TileIdx::Type>(graphs_[0]->tid()));
    ASSERT_EQ(1, static_cast<TileIdx::Type>(graphs_[1]->tid()));
}

TEST_F(GraphTest, vertex) {
    ASSERT_EQ(0, graphs_[0]->vertex(0)->vid());
}

TEST_F(GraphTest, vertexNull) {
    ASSERT_EQ(nullptr, graphs_[0]->vertex(11));
}

TEST_F(GraphTest, vertexCount) {
    ASSERT_EQ(2, graphs_[0]->vertexCount());
    ASSERT_EQ(2, graphs_[1]->vertexCount());
}

TEST_F(GraphTest, vertexNew) {
    auto count = graphs_[0]->vertexCount();
    graphs_[0]->vertexNew(10, 0);
    ASSERT_EQ(count + 1, graphs_[0]->vertexCount());
}

TEST_F(GraphTest, vertexNewKeyInUse) {
    auto count = graphs_[0]->vertexCount();
    try {
        graphs_[0]->vertexNew(0, 0);
    } catch (KeyInUseException& e) {
        ASSERT_EQ(count, graphs_[0]->vertexCount());
        return;
    }

    // Never reached.
    ASSERT_TRUE(false);
}

TEST_F(GraphTest, vertexIter) {
    auto g = graphs_[0];
    for (TestGraphTile::VertexIter vIter = g->vertexIter(); vIter != g->vertexIterEnd(); ++vIter) {
        auto v = vIter->second;
        v->data().x_ = 1;
    }
    for (TestGraphTile::VertexConstIter vIter = g->vertexIter(); vIter != g->vertexIterEnd(); ++vIter) {
        const auto v = vIter->second;
        ASSERT_LT(std::abs(v->data().x_ - 1), 1e-3);
    }
}

TEST_F(GraphTest, mirrorVertex) {
    auto mv = graphs_[0]->mirrorVertex(2);
    ASSERT_EQ(2, mv->vid());
    ASSERT_EQ(1, mv->masterTileId());
    ASSERT_EQ(nullptr, graphs_[0]->mirrorVertex(0));
}

TEST_F(GraphTest, mirrorVertexIter) {
    auto g = graphs_[0];
    size_t count = 0;
    for (auto mvIter = g->mirrorVertexIter(); mvIter != g->mirrorVertexIterEnd(); ++mvIter) {
        auto mv = mvIter->second;
        ASSERT_EQ(1, mv->masterTileId());
        count++;
    }
    ASSERT_EQ(2, count);
}

TEST_F(GraphTest, edgeCount) {
    auto g = graphs_[0];
    ASSERT_EQ(3, g->edgeCount());
}

TEST_F(GraphTest, edgeIter) {
    auto g = graphs_[0];
    for (auto eIter = g->edgeIter(); eIter != g->edgeIterEnd(); ++eIter) {
        auto& e = *eIter;
        auto srcId = e.srcId();
        ASSERT_NE(nullptr, g->vertex(srcId));
    }
}

TEST_F(GraphTest, edgeNew) {
    auto g = graphs_[0];
    g->edgeNew(1, 0, 0, 10);

    auto eIter = g->edgeIter();
    for (; eIter != g->edgeIterEnd(); ++eIter) {
        auto& e = *eIter;
        if (e.weight() > 9) {
            ASSERT_EQ(1, e.srcId());
            ASSERT_EQ(0, e.dstId());
            break;
        }
    }
    // Found the new edge.
    ASSERT_FALSE(eIter == g->edgeIterEnd());
}

TEST_F(GraphTest, edgeSorted) {
    auto g = graphs_[0];
    g->edgeSortedIs(true);
    ASSERT_TRUE(g->edgeSorted());
    g->edgeNew(1, 0, 0, 10);
    ASSERT_FALSE(g->edgeSorted());
    g->edgeSortedIs(true);
    ASSERT_TRUE(g->edgeSorted());
}

TEST_F(GraphTest, degreeBeforeSync) {
    auto v0 = graphs_[0]->vertex(0);
    auto v1 = graphs_[0]->vertex(1);
    auto v2 = graphs_[1]->vertex(2);
    auto v3 = graphs_[1]->vertex(3);

    ASSERT_EQ(DegreeCount(1), v0->outDeg());
    ASSERT_EQ(DegreeCount(2), v1->outDeg());
    ASSERT_EQ(DegreeCount(2), v2->outDeg());
    ASSERT_EQ(DegreeCount(1), v3->outDeg());

    ASSERT_EQ(DegreeCount(0), v0->inDeg());
    ASSERT_EQ(DegreeCount(1), v1->inDeg());
    ASSERT_EQ(DegreeCount(0), v2->inDeg());
    ASSERT_EQ(DegreeCount(1), v3->inDeg());

    auto mv0 = graphs_[1]->mirrorVertex(0);
    ASSERT_TRUE(nullptr == graphs_[1]->mirrorVertex(1));
    auto mv2 = graphs_[0]->mirrorVertex(2);
    auto mv3 = graphs_[0]->mirrorVertex(3);
    ASSERT_EQ(DegreeCount(2), mv0->accDeg());
    ASSERT_EQ(DegreeCount(1), mv2->accDeg());
    ASSERT_EQ(DegreeCount(1), mv3->accDeg());
}

TEST_F(GraphTest, finalizedFail) {
    auto g = graphs_[0];
    ASSERT_FALSE(g->finalized());
    try {
        g->finalizedIs(true);
    } catch (PermissionException& e) {
        ASSERT_FALSE(g->finalized());
        return;
    }

    // Never reached.
    ASSERT_TRUE(false);
}

TEST_F(GraphTest, degree) {
    degreeSync();

    auto v0 = graphs_[0]->vertex(0);
    auto v1 = graphs_[0]->vertex(1);
    auto v2 = graphs_[1]->vertex(2);
    auto v3 = graphs_[1]->vertex(3);

    ASSERT_EQ(DegreeCount(1), v0->outDeg());
    ASSERT_EQ(DegreeCount(2), v1->outDeg());
    ASSERT_EQ(DegreeCount(2), v2->outDeg());
    ASSERT_EQ(DegreeCount(1), v3->outDeg());

    ASSERT_EQ(DegreeCount(2), v0->inDeg());
    ASSERT_EQ(DegreeCount(1), v1->inDeg());
    ASSERT_EQ(DegreeCount(1), v2->inDeg());
    ASSERT_EQ(DegreeCount(2), v3->inDeg());
}

TEST_F(GraphTest, finalized) {
    degreeSync();

    auto g = graphs_[0];
    ASSERT_FALSE(g->finalized());
    g->finalizedIs(true);
    ASSERT_TRUE(g->finalized());
    ASSERT_TRUE(g->edgeSorted());
}

TEST_F(GraphTest, changeAfterinalized) {
    degreeSync();
    auto g = graphs_[0];
    g->finalizedIs(true);

    // Accessor is allowed.
    ASSERT_EQ(2, g->vertexCount());
    ASSERT_TRUE(nullptr != g->vertex(0));

    // Mutator is disallowed.
    try {
        graphs_[0]->vertexNew(10, 0);
    } catch (PermissionException& e) {
        goto try2;
    }

    // Never reached.
    ASSERT_TRUE(false);

try2: try {
        g->edgeNew(1, 0, 0, 10);
    } catch (PermissionException& e) {
        return;
    }

    // Never reached.
    ASSERT_TRUE(false);
}

