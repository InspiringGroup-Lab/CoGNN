#ifndef UTILS_BASIC_AL_H_
#define UTILS_BASIC_AL_H_
/**
 * Basic arithmetic and logic operations.
 * from zsim bithacks.h, https://github.com/s5z/zsim
 */
#include <cstdint>

/* "Bithack" operations, see the bithacks website
 * http://graphics.stanford.edu/~seander/bithacks.html
 */
// Integer log2 --- called ilog2 because cmath defines log2 for floats/doubles,
// and promotes int calls to use FP
template<typename T> static inline uint32_t ilog2(T val);
// Only specializations of unsigned types (no calling these with ints)
// __builtin_clz is undefined for 0 (internally, this uses bsr in x86-64)
template<> uint32_t ilog2<uint32_t>(uint32_t val) {
    return val? 31 - __builtin_clz(val) : 0;
}
template<> uint32_t ilog2<uint64_t>(uint64_t val) {
    return val? 63 - __builtin_clzl(val) : 0;
}

template<typename T>
static inline bool isPow2(T val) {
    return val && !(val & (val - 1));
}

/* Max and min: These work with side-effects, are type-safe, and gcc recognizes this pattern and uses
 * conditional moves (i.e., predication --> no unpredictable branches and great preformance)
 */
#ifdef MAX
#undef MAX
#endif
#define MAX(x, y) ({ __typeof__(x) xx = (x); __typeof__(y) yy = (y); (xx > yy)? xx : yy;})

#ifdef MIN
#undef MIN
#endif
#define MIN(x, y) ({ __typeof__(x) xx = (x); __typeof__(y) yy = (y); (xx < yy)? xx : yy;})

/* Some variadic template magic for max/min with N args.
 *
 * Type-wise, you can compare multiple types (e.g., maxN(1, -7, 3.3)), but the
 * output type is the first arg's type (e.g., returns 3)
 */
template <typename T> static inline T maxN(T a) { return a; }
template <typename T, typename U, typename ... V> static inline T maxN(T a, U b, V... c) {
    return maxN(((a > b)? a : b), c...);
}

template <typename T> static inline T minN(T a) { return a; }
template <typename T, typename U, typename ... V> static inline T minN(T a, U b, V... c) {
    return minN(((a < b)? a : b), c...);
}

#endif // UTILS_BASIC_AL_H_

