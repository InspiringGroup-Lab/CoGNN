#include "gtest/gtest.h"
#include "utils/thread_pool.h"
#include "comm_sync.h"
#include <cmath>

using namespace GraphGASLite;

class CommSyncTest : public ::testing::Test {
public:
    typedef CommSync<uint32_t, double> CommSyncType;
    typedef typename CommSyncType::KeyValue KeyValType;
    typedef std::function<void(uint32_t, CommSyncType*)> ThreadFuncType;

protected:
    virtual void SetUp() {
        pool_ = new ThreadPool(threadCount_);
        cs_ = new CommSyncType(threadCount_, CommSyncType::KeyValue(-1u, 0.));
    }

    virtual void TearDown() {
        delete pool_;
        delete cs_;
    }

    void RunTask(ThreadFuncType tf) {
        for (uint32_t tid = 0; tid < threadCount_; tid++) {
            pool_->add_task(std::bind(tf, tid, cs_));
        }
        pool_->wait_all();
    }

    const uint32_t threadCount_ = 8;
    ThreadPool* pool_;
    CommSyncType* cs_;
};

TEST_F(CommSyncTest, barrier) {
    lock_t lock;
    uint32_t curIter = -1;
    uint32_t arrivedThreads = 0;
    auto tf = [&](uint32_t tid, CommSyncType* cs) {
        for (uint32_t iter = 0; iter < 4; iter++) {
            // Sleep for different periods.
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * tid * iter));

            // Check always in the same iteration.
            mutex_begin(uniqLock, lock);
            if (arrivedThreads == 0) {
                // Enter next iteration.
                curIter += 1;
            }
            arrivedThreads = (arrivedThreads + 1) % threadCount_;
            ASSERT_EQ(iter, curIter);
            mutex_end();

            cs->barrier(tid);
        }
    };
    RunTask(tf);
}

TEST_F(CommSyncTest, barrierAND) {
    std::vector<bool> inputs(threadCount_, true);
    bool expOutput = true;

    auto tf = [&](uint32_t tid, CommSyncType* cs) {
        for (uint32_t iter = 0; iter < 4; iter++) {
            auto output = cs->barrierAND(tid, inputs[tid]);
            ASSERT_EQ(expOutput, output);
        }
    };

    RunTask(tf);

    inputs[2] = false;
    expOutput = false;
    RunTask(tf);

    inputs[2] = true;
    inputs[5] = false;
    expOutput = false;
    RunTask(tf);
}

TEST_F(CommSyncTest, comm) {

    auto tf = [this](uint32_t tid, CommSyncType* cs) {

        cs->keyValProdDelAll(tid);

        // Send.
        for (uint32_t dstId = 0; dstId < threadCount_; dstId++) {
            // Each thread sends to thread \c dstId <tt>tid * dstId</tt> pairs
            // of key-values. Key is \c tid, value is <tt>(dstId, 0.1 * i)</tt>.
            for (uint32_t i = 0; i < tid * dstId; i++) {
                cs->keyValNew(tid, dstId, tid, 0.1 * i);
            }
            cs->endTagNew(tid, dstId);
        }

        // Receive.
        // Received key-value pair count.
        std::vector<uint32_t> kvCounts(threadCount_, 0);
        // Sum of the received values.
        double sum = 0;
        while (true) {
            auto rd = cs->keyValPartitions(tid, threadCount_,
                [](uint32_t k){ return static_cast<size_t>(k); });
            const auto& prtns = rd.first;
            auto status = rd.second;

            if (status == CommSyncType::RECV_NONE) {
                // Sleep shortly to wait for data.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            ASSERT_EQ(threadCount_, prtns.size());

            // For each subpartition ...
            for (uint32_t idx = 0; idx < threadCount_; idx++) {
                // For each update ...
                for (const auto& u : prtns[idx]) {
                    kvCounts[u.key()]++;
                    sum += u.val();
                }
            }

            // Finish receiving.
            if (status == CommSyncType::RECV_FINISHED) break;
        }

        cs->keyValConsDelAll(tid);

        for (uint32_t srcId = 0; srcId < threadCount_; srcId++) {
            ASSERT_EQ(srcId * tid, kvCounts[srcId]);
        }

        double expect = 0;
        for (uint32_t srcId = 0; srcId < threadCount_; srcId++) {
            uint32_t num = srcId * tid;
            expect += 0.1 * (0 + num-1) * num / 2.;
        }
        ASSERT_LT(std::fabs(sum - expect), 1e-6);
    };

    // Do communication three times in a row to test *DelAll.
    RunTask(tf);
    RunTask(tf);
    RunTask(tf);
}

