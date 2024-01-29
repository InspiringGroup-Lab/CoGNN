#ifndef UTILS_EXCEPTION_H_
#define UTILS_EXCEPTION_H_
/**
 * Generic types of exception.
 *
 * Two generic types: range exception and resource exception.
 *
 * Resource exception has multiple derived types.
 *
 * All exception types are derived from std::exception.
 */
#include <exception>
#include <string>

/**
 * Resource exception.
 */
class Exception : public std::exception {
public:
    const char* what() const throw() { return str_.c_str(); }
    virtual ~Exception() {}
protected:
    std::string str_;
protected:
    explicit Exception(const std::string& str) : str_(str) {}
};

/**
 * Range exception.
 */
class RangeException : public Exception {
public:
    explicit RangeException(const std::string& str) : Exception(str) {}
};

/**
 * Derived types of resource exception.
 */

class KeyInUseException : public Exception {
public:
    explicit KeyInUseException(const std::string& str) : Exception(str) {}
};

class MemoryException : public Exception {
public:
    explicit MemoryException(const std::string& str) : Exception(str) {}
};

class FileException : public Exception {
public:
    explicit FileException(const std::string& str) : Exception(str) {}
};

class PermissionException : public Exception {
public:
    explicit PermissionException(const std::string& str) : Exception(str) {}
};

class NullPointerException : public Exception {
public:
    explicit NullPointerException(const std::string& str) : Exception(str) {}
};

class UnknownTypeException : public Exception {
public:
    explicit UnknownTypeException(const std::string& str) : Exception(str) {}
};

class InvalidArgumentException : public Exception {
public:
    explicit InvalidArgumentException(const std::string& str) : Exception(str) {}
};

class TimeoutException : public Exception {
public:
    explicit TimeoutException(const std::string& str) : Exception(str) {}
};

class InternalException : public Exception {
public:
    explicit InternalException(const std::string& str) : Exception(str) {}
};

class MessageException : public Exception {
public:
    explicit MessageException(const std::string& str) : Exception(str) {}
};

class QueueException : public Exception {
public:
    explicit QueueException(const std::string& str) : Exception(str) {}
};

class ResultNotReadyException : public Exception {
public:
    explicit ResultNotReadyException(const std::string& str) : Exception(str) {}
};


#endif // UTILS_EXCEPTION_H_
