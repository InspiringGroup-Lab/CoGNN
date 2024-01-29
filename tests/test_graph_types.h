#ifndef TESTS_TEST_GRAPH_TYPES_H_
#define TESTS_TEST_GRAPH_TYPES_H_

#include "graph.h"

using namespace GraphGASLite;

class TestData {
public:
    TestData(const VertexIdx&, const double x) : x_(x) {}

    double x_;
};

class TestUpdate {
public:
    TestUpdate(const double x = 0) : x_(x) {}
    inline TestUpdate& operator+=(const TestUpdate& update) {
        this->x_ = std::min(this->x_, update.x_);
        return *this;
    }

    double x_;
};

typedef GraphTile<TestData, TestUpdate> TestGraphTile;

#endif
