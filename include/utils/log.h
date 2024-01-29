#ifndef UTILS_LOG_H_
#define UTILS_LOG_H_
/**
 * Generic logging/info/warn/panic routines.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#define PANIC_EXIT_CODE (112)

/* Logging */

// // Generic logging, thread-unsafe
// #define __panic(logFdErr, logHeader, ...) \
// { \
//     fprintf(logFdErr, "%sPanic on %s:%d: ", logHeader, __FILE__, __LINE__); \
//     fprintf(logFdErr, __VA_ARGS__); \
//     fprintf(logFdErr, "\n"); \
//     fflush(logFdErr); \
//     exit(PANIC_EXIT_CODE); \
// }

// #define __warn(logFdErr, logHeader, ...) \
// { \
//     fprintf(logFdErr, "%sWARN: ", logHeader); \
//     fprintf(logFdErr, __VA_ARGS__); \
//     fprintf(logFdErr, "\n"); \
//     fflush(logFdErr); \
// }

// #define __info(logFdOut, logHeader, ...) \
// { \
//     fprintf(logFdOut, "%s", logHeader); \
//     fprintf(logFdOut, __VA_ARGS__); \
//     fprintf(logFdOut, "\n"); \
//     fflush(logFdOut); \
// }

// // Basic logging, thread-unsafe, print to stdout/stderr, no header
// #define panic(...) __panic(stderr, "", __VA_ARGS__)
// #define warn(...)  __warn(stderr, "", __VA_ARGS__)
// #define info(...)  __info(stdout, "", __VA_ARGS__)

// // Logging class, thread-safe, support redirection to files, support header
// class Logger {
//     public:
//         Logger(const char* header = "", const char* file = nullptr) : logHeader(header) {
//             if (file) {
//                 fd = fopen(file, "a");
//                 if (fd == NULL) {
//                     perror("fopen() failed");
//                     // We can panic in InitLog (will dump to stderr)
//                     panic("Could not open logfile %s", file);
//                 }
//                 logFdOut = fd;
//                 logFdErr = fd;
//             } else {
//                 fd = nullptr;
//                 logFdOut = stdout;
//                 logFdErr = stderr;
//             }
//         }

//         ~Logger() {
//             fclose(fd);
//         }

//         template<typename... Args>
//         void log_panic(const char* fmt, Args... args) {
//             __panic(logFdErr, logHeader.c_str(), fmt, args...);
//         }

//         template<typename... Args>
//         void log_warn(const char* fmt, Args... args) {
//             logPrintLock.lock();
//             __warn(logFdErr, logHeader.c_str(), fmt, args...);
//             logPrintLock.unlock();
//         }

//         template<typename... Args>
//         void log_info(const char* fmt, Args... args) {
//             logPrintLock.lock();
//             __info(logFdErr, logHeader.c_str(), fmt, args...);
//             logPrintLock.unlock();
//         }

//     private:
//         FILE* fd;
//         FILE* logFdErr;
//         FILE* logFdOut;
//         const std::string logHeader;
//         std::mutex logPrintLock;
// };


/* Assertion */

#ifndef NASSERT

#ifndef assert
#define assert(expr) \
if (!(expr)) { \
    fprintf(stderr, "%sFailed assertion on %s:%d '%s'\n", "", __FILE__, __LINE__, #expr); \
    fflush(stderr); \
    exit(PANIC_EXIT_CODE); \
};
#endif

#define assert_msg(cond, ...) \
if (!(cond)) { \
    fprintf(stderr, "%sFailed assertion on %s:%d: ", "", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    fflush(stderr); \
    exit(PANIC_EXIT_CODE); \
};

#else // NASSERT

// Avoid unused warnings, never emit any code
// see http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
#define assert(cond) do { (void)sizeof(cond); } while (0);
#define assert_msg(cond, ...) do { (void)sizeof(cond); } while (0);

#endif // NASSERT


#endif // UTILS_LOG_H_

