#ifndef UTILS_THREADS_H_
#define UTILS_THREADS_H_
/**
 * Wrappers for thread support routines,
 * including thread manipulation and synchronization primitives.
 *
 * Use c++11 primitives and routines.
 */
#include <condition_variable>
#include <mutex>
#include <thread>
#include <functional>

class barrier;

using thread_t = std::thread;
using lock_t = std::mutex;
using cond_t = std::condition_variable;
using bar_t = barrier;


/* Threads */
/*  Use constructor to create a thread and execute the function:
 *  template< class Function, class... Args >
 *  explicit thread( Function&& f, Args&&... args );
 *
 *  Use member function join() to join.
 */


/* Mutexes */
/*  Use member functions lock(), try_lock(), and unlock().
 */

/*  General-purpose mutex ownership wrapper.
 */
#define mutex_begin(uniq_lk, lk) \
    { std::unique_lock<lock_t> uniq_lk(lk);

#define mutex_end() \
    }


/* Condition variables */
/*  Use member functions:
 *  void wait( std::unique_lock<std::mutex>& lock );
 *  template< class Predicate >
 *  void wait( std::unique_lock<std::mutex>& lock, Predicate pred );
 *  void notify_one();
 *  void notify_all();
 */


/* Barrier */
class barrier {
    public:
        static constexpr int SERIAL_LAST_THREAD = 1;

        /**
         * Construct the barrier.
         *
         * @param threadCount   The number of threads involved in the barrier.
         */
        explicit barrier(const std::size_t threadCount)
            : threadCount_(threadCount), remain_(threadCount), barCount_(0)
        {
            // Nothing else to do.
        }

        /**
         * Wait on the barrier.
         *
         * @param onSerialPoint     A callback to be called at the barrier serial point.
         *
         * @return      SERIAL_LAST_THREAD if the thread is the last one arriving
         *              at the barrier, or 0 otherwise.
         */
        int wait(const std::function<void(void)>& onSerialPoint = [](){}) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                auto curBarCount = barCount_;
                remain_--;
                if (remain_) {
                    // Not all threads have arrived, wait.
                    // Wait predicate should not depend on \c remain_, since it must
                    // be reset at the time all threads arriving at the barrier.
                    cv_.wait(lock, [this, &curBarCount]{ return barCount_ != curBarCount; });
                    return 0;
                }
            }
            // Only executed by the very last thread arriving at the barrier.
            // Reset \c remain_.
            remain_ = threadCount_;
            // Increase \c barCount_, which is used as the wait predicate.
            barCount_++;
            // Call callback function at serial point.
            if (onSerialPoint) onSerialPoint();
            // Notify all after updating wait predicate, should not hold the mutex.
            cv_.notify_all();
            return SERIAL_LAST_THREAD;
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        const std::size_t threadCount_;
        std::size_t remain_;
        std::size_t barCount_;
};

#endif // UTILS_THREADS_H_

