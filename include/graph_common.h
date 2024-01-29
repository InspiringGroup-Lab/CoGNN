#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <string>
#include "utils/exception.h"
#include "utils/log.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/access.hpp>

using std::string;

template<typename T>
using Ptr = std::shared_ptr<T>;

template<bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;

template<typename UnitType, typename RepType>
struct IndexType {
public:
    IndexType(const UnitType& idx) : idx_(idx) {}
    IndexType() : idx_() {}
    operator UnitType() const { return idx_; }
    typedef UnitType Type;
private:
    UnitType idx_;

    friend class boost::serialization::access;

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & idx_;
    } 
};

template<typename UnitType, typename RepType>
struct CountType {
public:
    CountType(const UnitType& cnt) : cnt_(cnt) {}
    UnitType cnt() const {
        return cnt_;
    }
    typedef UnitType Type;

    inline bool operator==(const CountType& other) const {
        return cnt_ == other.cnt_;
    }
    inline bool operator<(const CountType& other) const {
        return cnt_ < other.cnt_;
    }
    inline bool operator!=(const CountType& other) const {
        return !(*this == other);
    }
    inline bool operator<=(const CountType& other) const {
        return !(other < *this);
    }
    inline bool operator>(const CountType& other) const {
        return other < *this;
    }
    inline bool operator>=(const CountType& other) const {
        return !(*this < other);
    }

    CountType& operator+=(const CountType& other) {
        this->cnt_ += other.cnt_;
        return *this;
    }
    CountType& operator-=(const CountType& other) {
        this->cnt_ -= other.cnt_;
        return *this;
    }
    CountType& operator%=(const CountType& other) {
        this->cnt_ %= other.cnt_;
        return *this;
    }
    friend CountType operator+(CountType lhs, const CountType& rhs) {
        lhs += rhs;
        return lhs;
    }
    friend CountType operator-(CountType lhs, const CountType& rhs) {
        lhs -= rhs;
        return lhs;
    }
    friend CountType operator%(CountType lhs, const CountType& rhs) {
        lhs %= rhs;
        return lhs;
    }
    template<typename T>
    friend T operator*(T lhs, const CountType& rhs) {
        lhs *= rhs.cnt_;
        return lhs;
    }
    template<typename T>
    friend T operator/(T lhs, const CountType& rhs) {
        lhs /= rhs.cnt_;
        return lhs;
    }

    /**
     * Prefix increment.
     */
    CountType& operator++() {
        this->cnt_ += 1;
        return *this;
    }
    /**
     * Prefix decrement.
     */
    CountType& operator--() {
        this->cnt_ -= 1;
        return *this;
    }
    /**
     * Postfix increment.
     */
    CountType operator++(int) {
        CountType tmp(*this);
        operator++();
        return tmp;
    }
    /**
     * Postfix decrement.
     */
    CountType operator--(int) {
        CountType tmp(*this);
        operator--();
        return tmp;
    }

private:
    UnitType cnt_;

    friend class boost::serialization::access;

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & cnt_;
    }
};

#endif // COMMON_H_
