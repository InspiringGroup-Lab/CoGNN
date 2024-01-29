#ifndef UTILS_STREAM_H_
#define UTILS_STREAM_H_
/**
 * Generic FIFO stream.
 *
 * Implemented by std::vector.
 */
#include <algorithm>    // for std::sort
#include <vector>

template <typename Data>
class Stream {
    public:
        typedef typename std::vector<Data>::iterator StreamIter;
        typedef typename std::vector<Data>::const_iterator StreamConstIter;

    public:
        Stream(size_t num = 16) {
            stream.reserve(num);
        }

        ~Stream() {}

        /* Copy and move */

        Stream(const Stream<Data>&) = delete;
        Stream<Data>& operator=(const Stream<Data>&) = delete;

        Stream(Stream<Data>&& s) { stream.swap(s.stream); }
        Stream<Data>& operator=(Stream<Data>&& s) {
            // avoid self assign
            if (this == &s) return *this;
            stream.swap(s.stream);
            return *this;
        }

        /* Member access */

        const Data* data() const { return stream.data(); }
        Data* data() { return stream.data(); }
        size_t size() const { return stream.size(); }
        size_t byte_size() const { return size() * sizeof(Data); }

        Data& operator[](size_t idx) { return stream[idx]; }
        const Data& operator[](size_t idx) const { return stream[idx]; }

        StreamIter begin() { return stream.begin(); }
        StreamIter end() { return stream.end(); }

        StreamConstIter begin() const { return stream.begin(); }
        StreamConstIter end() const { return stream.end(); }

        StreamConstIter cbegin() const { return stream.cbegin(); }
        StreamConstIter cend() const { return stream.cend(); }

        /* Modifiers */

        void reset(size_t num = 16) {
            if (num != stream.capacity()) {
                // Non-binding request, as shrink_to_fit() is non-binding.
                stream.resize(num);
                stream.shrink_to_fit();
            }
            stream.clear();
        }

        void swap(Stream<Data>& s) {
            stream.swap(s.stream);
        }

        void put(const Data& d) {
            // The growth of the STL vector is implementation dependent, but it
            // usually grows exponentially as a nearly-optimal solution.
            stream.push_back(d);
        }

        void put(Data&& d) {
            stream.push_back(std::forward<Data>(d));
        }

        void sort() {
            std::sort(stream.begin(), stream.end());
        }

    private:
        std::vector<Data> stream;
};

#endif // UTILS_STREAM_H_

