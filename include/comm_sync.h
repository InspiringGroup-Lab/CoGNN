#ifndef COMM_SYNC_H_
#define COMM_SYNC_H_

#include "utils/stream.h"
#include "utils/threads.h"
#include "utils/exception.h"

#include <thread>
#include <vector>
#include <memory>

#include "Common/Defines.h"
#include "Network/IOService.h"

#include "Network/Endpoint.h"
#include "Network/Channel.h"

// #include "Common/ByteStream.h"

#include "task.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

#include <sstream>
#include <iostream>

namespace GraphGASLite {

template<typename KType, typename VType>
class CommSync {
public:
    typedef KType KeyType;
    typedef VType ValType;

    /**
     * The key-value pair type used to communicate between threads.
     */
    class KeyValue {
    public:
        KeyValue(const KeyType& key, const ValType& val)
            : key_(key), val_(val)
        {
            // Nothing else to do.
        }

        KeyValue() : key_(), val_() { }

        const KeyType& key() const { return key_; }

        const ValType& val() const { return val_; }

    private:
        KeyType key_;
        ValType val_;
    };

    enum RecvStatusType {
        // Nothing is received.
        RECV_NONE,
        // Some data is received, but not finished.
        RECV_CONTINUED,
        // All data is received.
        RECV_FINISHED,
    };

    typedef Stream<KeyValue> KeyValueStream;

    // Reserve at most 4k or 256 key-value pairs for each prod-cons pair.
    static constexpr size_t reservedStreamSize = 4096/sizeof(KeyValue) < 256 ?
        4096/sizeof(KeyValue) : 256;

public:
    explicit CommSync(const uint32_t threadCount, const KeyValue& endTag);

    ~CommSync();

    /**
     * Number of threads.
     */
    uint32_t threadCount() const { return threadCount_; }

    /**
     * Thread register.
     */
    void threadIdIs(const uint32_t threadId);

    /**
     * Synchronization barrier.
     */
    void barrier(const uint32_t threadId);

    /**
     * Synchronization barrier, also do an AND reduction.
     */
    bool barrierAND(const uint32_t threadId, bool input);

    /**
     * Send a key-value pair from \c prodId to \c consId.
     */
    void keyValNew(const uint32_t prodId, const uint32_t consId,
            const KeyType& key, const ValType& val);

    /**
     * Send end-of-message tag from \c prodId to \c consId.
     */
    void endTagNew(const uint32_t prodId, const uint32_t consId);

    /**
     * Delete all key-value pairs associated with \c prodId at producer side
     * after communication is done.
     */
    void keyValProdDelAll(const uint32_t prodId);

    /**
     * Delete all key-value pairs associated with \c consId at consumer side
     * after communication is done.
     */
    void keyValConsDelAll(const uint32_t consId);

    /**
     * Set remote networking channels.
     */
    void setChannels(std::vector<std::vector<osuCrypto::Channel*>>* chnls) {
        channels = chnls;
    }

    /**
     * Asynchronously remote send keyValNew to a consumer.
     */
    void remoteKeyValNew(const uint32_t prodId, const uint32_t consId,
        const KeyType& key, const ValType& val) {
        // serialize obj into an std::string
        std::string serial_str;
        boost::iostreams::back_insert_device<std::string> inserter(serial_str);
        boost::iostreams::stream<boost::iostreams::back_insert_device<std::string>> s(inserter);
        boost::archive::binary_oarchive oa(s);

        ValType val_cpy = val;
        ValType share = ValType::splitRandomShareFromUpdate(prodId, consId, val_cpy);

        oa << prodId << consId << key << val_cpy << share;

        s.flush();

        (*channels)[prodId][consId]->asyncSend(std::move(serial_str));
    }

    /**
     * Asynchronously remote send keyValNew ends.
     */
    void endRemoteKeyValNew(const uint32_t prodId, const uint32_t consId) {
        std::string end_str = "FIN_SEND";
        (*channels)[prodId][consId]->asyncSend(std::move(end_str));
    }

    /**
     * Synchrnously remote get keyValNew from a producer.
     */
    int getKeyValNew(const uint32_t prodId, const uint32_t consId) {
        std::string serial_str;
        (*channels)[consId][prodId]->recv(serial_str);
        if (serial_str == "FIN_SEND") return -1;

        // wrap buffer inside a stream and deserialize serial_str into obj
        boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
        boost::archive::binary_iarchive ia(s);

        KeyType key;
        ValType val;
        ValType share;
        uint32_t pId;
        uint32_t cId;
        
        ia >> pId >> cId >> key >> val >> share;

        if (pId != prodId || cId != consId)
            throw MessageException("Unexpected prodId and consId in Recv.\n");

        ValType::mergeUpdateShare(consId, val, share);

        streamLists_[prodId][consId].put(KeyValue(key, val));
        return 0;
    }

    /**
     * Asynchronously remote get keyValNew ends.
     */
    void endGetKeyValNew(const uint32_t prodId, const uint32_t consId) {
        std::string end_str = "FIN_RECV";
        (*channels)[consId][prodId]->asyncSend(std::move(end_str));
    }

    /**
     * Ensure all KeyValNew have been received.
     */
    void EnsureEndGetKeyValNew(const uint32_t prodId, const uint32_t consId) {
        std::string serial_str;
        (*channels)[prodId][consId]->recv(serial_str);
        if (serial_str != "FIN_RECV") 
            throw MessageException("FIN_SEND not received\n");
    }

    void sendPosVec(const PosVec& posv, const uint32_t prodId, const uint32_t consId) {
        std::string cmd_str = "POS_VEC";

        std::string serial_str;
        boost::iostreams::back_insert_device<std::string> inserter(serial_str);
        boost::iostreams::stream<boost::iostreams::back_insert_device<std::string>> s(inserter);
        boost::archive::binary_oarchive oa(s);  

        oa << cmd_str << posv;

        s.flush();

        (*channels)[prodId][consId]->asyncSend(std::move(serial_str)); 
    }

    void recvPosVec(PosVec& posv, const uint32_t prodId, const uint32_t consId) {
        std::string serial_str;
        (*channels)[consId][prodId]->recv(serial_str);

        // wrap buffer inside a stream and deserialize serial_str into obj
        boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
        boost::archive::binary_iarchive ia(s);

        std::string cmd_str;
        ia >> cmd_str >> posv;

        if (cmd_str != "POS_VEC") {
            printf("Did not receive expected POS_VEC!\n");
            exit(-1);
        }
    }

    void sendShareVecVec(const ShareVecVec& svv, const uint32_t prodId, const uint32_t consId) {
        std::string cmd_str = "SHARE_VEC_VEC";

        std::string serial_str;
        boost::iostreams::back_insert_device<std::string> inserter(serial_str);
        boost::iostreams::stream<boost::iostreams::back_insert_device<std::string>> s(inserter);
        boost::archive::binary_oarchive oa(s);  

        oa << cmd_str << svv;

        s.flush();

        (*channels)[prodId][consId]->asyncSend(std::move(serial_str)); 
    }

    void recvShareVecVec(ShareVecVec& svv, const uint32_t prodId, const uint32_t consId) {
        std::string serial_str;
        (*channels)[consId][prodId]->recv(serial_str);

        // wrap buffer inside a stream and deserialize serial_str into obj
        boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
        boost::archive::binary_iarchive ia(s);

        std::string cmd_str;
        svv.clear();
        ia >> cmd_str >> svv;

        if (cmd_str != "SHARE_VEC_VEC") {
            printf("Did not receive expected SHARE_VEC_VEC!\n");
            exit(-1);
        }
    }

    /**
     * Receive all key-value pairs available now, and partition them into subpartitions.
     *
     * @param consId            Thread index of the consumer.
     * @param partitionCount    Number of subpartitions.
     * @param partitionFunc     Partition function. For a key \c k, partition index
     *                          is calculated as <tt>partitionFunc(k) % partitionCount</tt>.
     *
     * @return      A pair consisting of the subpartitions, and the receiving status.
     */
    std::pair<std::vector<KeyValueStream>, RecvStatusType> keyValPartitions(
            const uint32_t consId, const size_t partitionCount,
            std::function<size_t(const KeyType&)> partitionFunc);

    /**
     * Receive all key-value pairs available now, and partition them based on tiles.
     *
     * @param consId            Thread index of the consumer.
     *
     * @return      key-value pairs partitioned on tiles.
     */
    std::vector<KeyValueStream> keyValTiles(const uint32_t consId);

private:
    // Remote networking channel
    std::vector<std::vector<osuCrypto::Channel*>>* channels;

    const uint32_t threadCount_;

    /* Synchronization. */

    // Barrier.
    bar_t bar_;

    // Used for barrierAND.
    bool barANDCurReduction_;
    bool barANDLastResult_;

    /* Communication. */

    /**
     * End-of-message tag for communication.
     */
    const KeyValue endTag_;

    /**
     * Each producer thread is associated with multiple streams, each of which
     * is for a consumer thread.
     *
     * Indexed by [prodId][consId].
     */
    std::vector<std::vector<KeyValueStream>> streamLists_;

};

template<typename KType, typename VType>
constexpr size_t CommSync<KType, VType>::reservedStreamSize;



template<typename KType, typename VType>
CommSync<KType, VType>::
CommSync(const uint32_t threadCount, const KeyValue& endTag)
    : threadCount_(threadCount),
      bar_(threadCount), barANDCurReduction_(true), barANDLastResult_(false),
      endTag_(endTag)
{
    // Initialize communication streams.
    streamLists_.resize(threadCount_);
    for (auto& sl : streamLists_) {
        sl.resize(threadCount_);
        for (auto& s : sl) {
            s.reset(reservedStreamSize);
        }
    }
}

template<typename KType, typename VType>
CommSync<KType, VType>::
~CommSync() {
    // Nothing to do.
}

template<typename KType, typename VType>
void CommSync<KType, VType>::
threadIdIs(const uint32_t) {
    // Nothing to do.
}

template<typename KType, typename VType>
void CommSync<KType, VType>::
barrier(const uint32_t) {
    bar_.wait();
}

template<typename KType, typename VType>
bool CommSync<KType, VType>::
barrierAND(const uint32_t, bool input) {
    barANDCurReduction_ &= input;
    auto scb = [this](){
        barANDLastResult_ = barANDCurReduction_;
        barANDCurReduction_ = true;
    };
    bar_.wait(scb);
    return barANDLastResult_;
}

template<typename KType, typename VType>
void CommSync<KType, VType>::
keyValNew(const uint32_t prodId, const uint32_t consId,
        const KeyType& key, const ValType& val) {
    streamLists_[prodId][consId].put(KeyValue(key, val));
}

template<typename KType, typename VType>
void CommSync<KType, VType>::
endTagNew(const uint32_t, const uint32_t) {
    // Nothing to do.
}

template<typename KType, typename VType>
void CommSync<KType, VType>::
keyValProdDelAll(const uint32_t prodId) {
    for (auto& s : streamLists_[prodId]) {
        s.reset(std::max<size_t>(s.size(), reservedStreamSize));
    }
}

template<typename KType, typename VType>
void CommSync<KType, VType>::
keyValConsDelAll(const uint32_t consId) {
    for (auto& sl : streamLists_) {
        sl[consId].reset(std::max<size_t>(sl[consId].size(), reservedStreamSize));
    }
}

template<typename KType, typename VType>
std::pair<std::vector<typename CommSync<KType, VType>::KeyValueStream>,
    typename CommSync<KType, VType>::RecvStatusType> CommSync<KType, VType>::
keyValPartitions(const uint32_t consId, const size_t partitionCount,
        std::function<size_t(const KeyType&)> partitionFunc) {

    std::vector<KeyValueStream> prtns(partitionCount);

    // Take barrier to ensure all threads have finished sending all data.
    barrier(consId);

    // Local stream.
    if (partitionCount == 1) {
        prtns[0].swap(streamLists_[consId][consId]);
    } else {
        for (const auto& kv : streamLists_[consId][consId]) {
            auto pid = partitionFunc(kv.key()) % partitionCount;
            prtns[pid].put(kv);
        }
    }

    // Remote streams.
    for (uint32_t prodId = 0; prodId < threadCount_; prodId++) {
        // Skip local stream.
        if (prodId == consId) continue;

        for (const auto& kv : streamLists_[prodId][consId]) {
            auto pid = partitionFunc(kv.key()) % partitionCount;
            prtns[pid].put(kv);
        }
    }

    return std::make_pair(std::move(prtns), RECV_FINISHED);
}

template<typename KType, typename VType>
std::vector<typename CommSync<KType, VType>::KeyValueStream> CommSync<KType, VType>::
keyValTiles(const uint32_t consId) {

    std::vector<KeyValueStream> prtns(threadCount_);

    // Take barrier to ensure all threads have finished sending all data.
    // barrier(consId);

    // Remote streams.
    for (uint32_t prodId = 0; prodId < threadCount_; prodId++) {
        for (const auto& kv : streamLists_[prodId][consId]) {
            prtns[prodId].put(kv);
        }
    }

    return std::move(prtns);
}

} // namespace GraphGASLite

#endif // COMM_SYNC_H_
