#ifndef UTILS_THREAD_POOL_H_
#define UTILS_THREAD_POOL_H_
/**
 * Thread pool.
 */
#include <queue>
#include "log.h"
#include "threads.h"

namespace GraphGASLite {

typedef std::function<void()> TaskType;

class TaskQueue {
    public:
        TaskQueue() : stop(false) {}
        ~TaskQueue() {}

        // Move
        TaskQueue(TaskQueue&&) = default;
        TaskQueue& operator=(TaskQueue&&) = default;
        // No copy
        TaskQueue(const TaskQueue&) = delete;
        TaskQueue& operator=(const TaskQueue&) = delete;

        void enqueue(const TaskType& task) {
            mutex_begin(uqlk, lock_);
            if (stop) {
                throw std::runtime_error("TaskQueue: enqueue on stopped task queue!");
            }
            if (!task) {
                throw std::runtime_error("TaskQueue: enqueue empty task!");
            }
            queue_.push(task);
            mutex_end();
            enqueue_.notify_one();
        }

        // blocked
        TaskType dequeue() {
            mutex_begin(uqlk, lock_);
            enqueue_.wait(uqlk, [this]{ return !queue_.empty() || stop; });
            if (stop && queue_.empty()) {
                // empty task
                return std::function<void()>();
            }
            TaskType task = queue_.front();
            queue_.pop();
            return task;
            mutex_end();
        }

        void close() {
            mutex_begin(uqlk, lock_);
            stop = true;
            mutex_end();
            enqueue_.notify_all();
        }

    private:
        std::queue<TaskType> queue_;
        lock_t lock_;
        cond_t enqueue_;
        bool stop;
};

class ThreadPool {
    public:
        typedef uint32_t tid_t;
        static const tid_t INV_TID = ((uint32_t)-1);

        ThreadPool(uint32_t num_workers) :
            num_workers_(num_workers), queues_(num_workers_),
            num_tasks_(0), cur_worker_(0) {

            for (tid_t tid = 0; tid < num_workers_; tid++) {
                workers_.emplace_back(&ThreadPool::worker_func, this, tid);
            }
        }

        ~ThreadPool() {
            for (auto& q : queues_) {
                q.close();
            }
            for (auto& w : workers_) {
                w.join();
            }
        }

        // Move
        ThreadPool(ThreadPool&&) = default;
        ThreadPool& operator=(ThreadPool&&) = default;
        // No copy
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        void add_task(const TaskType& task, tid_t tid = INV_TID) {
            if (tid == INV_TID) {
                tid = next_worker();
            }

            mutex_begin(uqlk, st_lk_);
            num_tasks_++;
            mutex_end();

            queues_[tid].enqueue(task);
        }

        void wait_all() {
            mutex_begin(uqlk, st_lk_);
            task_done_.wait(uqlk, [this]{ return num_tasks_ == 0; });
            mutex_end();
        }

    private:
        uint32_t num_workers_;
        std::vector<thread_t> workers_;

        std::vector<TaskQueue> queues_;

        // Track number of pending tasks
        lock_t st_lk_;
        cond_t task_done_;
        uint32_t num_tasks_;

        // For task assignment
        tid_t cur_worker_;

    private:

        tid_t next_worker() {
            return cur_worker_ = (cur_worker_ + 1) % num_workers_;
        }

        void worker_func(tid_t tid) {
            while (true) {
                TaskType task = queues_[tid].dequeue();
                if (!task) return;

                task();

                mutex_begin(uqlk, st_lk_);
                assert(num_tasks_ > 0);
                num_tasks_--;
                mutex_end();
                task_done_.notify_all();
            }
        }
};

}

#endif // UTILS_THREAD_POOL_H_

