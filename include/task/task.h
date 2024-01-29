#ifndef TASK_H_
#define TASK_H_

#include <cstdint>
#include <queue>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

enum TASK_TYPE {
    ADD_UINT,
    ADD_ULONG,
    ADD_DOUBLE,
    ADD_PAIR_DOUBLE_UINT,
    ADD_MIXED_PAIR_DOUBLE_UINT,
    ADD_UINT_WITH_REPLACE_PARENT,
    UINT_REPLACE_PARENT,
    MIN_UINT_WITH_PARENT,
    DIV_DOUBLE,
    DEC_UINT,
    DEC_ULONG,
    DEC_DOUBLE,
    DEC_PAIR_UINT_ULONG,
    SWAP_CIPHER_ENTRY,
    GCN_VECTOR_SCALE, // Vector times the weight of an edge, used in Scatter
    GCN_VECTOR_ADDITION,
    GCN_FORWARD_NN,
    GCN_FORWARD_NN_PREDICTION,
    GCN_BACKWARD_NN_INIT,
    GCN_BACKWARD_NN, 
};

// When finished, the task result is written to the first operand.

#ifdef SGXBACKEND
#define DATA_SIZE 16
#define MAC_SIZE 16
#define NAC_SIZE 16

struct CipherEntry{
	uint8_t data[DATA_SIZE];	// Ciphertext content
	uint8_t nac[NAC_SIZE];		// This field store nonce + counter 
	uint8_t mac[MAC_SIZE];		// This field stores MAC of data entry fields

    // CipherEntry& operator=(const CipherEntry & ce) {
    //     memcpy(data, ce.data, DATA_SIZE);
    //     memcpy(nac, ce.nac, NAC_SIZE);
    //     memcpy(mac, ce.mac, MAC_SIZE);
    //     return *this;
    // }
};
typedef struct CipherEntry CipherEntry;
#endif

// #ifdef SSHEBACKEND
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdlib>
#include <atomic>
#include <mutex>

class CryptoUtil;

#ifdef USE_FHE
#define MAX_CIPHER_TEXT_SIZE 400000
#define MAX_PUBLIC_KEY_SIZE 40000000
#else
#define MAX_CIPHER_TEXT_SIZE 15000
#define MAX_PUBLIC_KEY_SIZE 1800
#endif


// Define a singleton class to store the GNN parameters
class GNNParam {
    // Declare the private constructor
    GNNParam() {}
    // Declare the private copy constructor
    GNNParam(const GNNParam&) {}
    // Declare the private assignment operator
    GNNParam& operator=(const GNNParam&) {}
    
    // Declare the public parameters
public:
    int num_layers; // The number of layers
    int num_labels; // The number of labels (classes)
    int input_dim; // The input dimension
    int hidden_dim; // The hidden dimension
    int num_samples; // The global number of samples
    int num_edges;
    double learning_rate; // The learning rate
    double train_ratio;
    double val_ratio;
    double test_ratio;

    // Define a public static method to get the singleton instance
    static GNNParam& getGNNParam() {
        static GNNParam instance;
        // Return the instance
        return instance;
    }
    // Define a public method to read the configuration from a file
    void readConfig(const std::string& file_name) {
        // Open the file in read mode
        std::ifstream fin(file_name);
        // Check if the file is opened successfully
        if (fin.is_open()) {
            // Read the parameters from the file
            // The file format is as follows:
            // num_layers: <value>
            // num_labels: <value>
            // input_dim: <value>
            // hidden_dim: <value>
            // num_samples: <value>
            // learning_rate: <value>
            // train_ratio: <value>
            // val_ratio: <value>
            // test_ratio: <value>
            // Each line has a parameter name followed by a colon and a value
            // The values are separated by whitespace
            std::string param; // A string to store the parameter name
            char colon; // A char to store the colon
            while (fin >> param >> colon) {
                // Check if the colon is valid
                if (colon != ':') {
                    // Print an error message
                    std::cerr << "Invalid format: expected a colon after " << param << std::endl;
                    // Break the loop
                    break;
                }
                // Check the parameter name and assign the value accordingly
                if (param == "num_layers") {
                    fin >> num_layers;
                } else if (param == "num_labels") {
                    fin >> num_labels;
                } else if (param == "input_dim") {
                    fin >> input_dim;
                } else if (param == "hidden_dim") {
                    fin >> hidden_dim;
                } else if (param == "num_samples") {
                    fin >> num_samples;
                } else if (param == "num_edges") {
                    fin >> num_edges;
                } else if (param == "learning_rate") {
                    fin >> learning_rate;
                } else if (param == "train_ratio") {
                    fin >> train_ratio;
                } else if (param == "val_ratio") {
                    fin >> val_ratio;
                } else if (param == "test_ratio") {
                    fin >> test_ratio;
                } else {
                    // Print an error message
                    std::cerr << "Unknown parameter: " << param << std::endl;
                    // Break the loop
                    break;
                }
            }
            // Close the file
            fin.close();
        }
        else {
            // Print an error message
            std::cerr << "Failed to open the file: " << file_name << std::endl;
        }
    }
};

struct CipherEntry {
	uint8_t ct[MAX_CIPHER_TEXT_SIZE]; // HE ciphertext
	size_t tid; // tile index of the public key owner
    bool isShare;
    uint64_t share[2];
    int plainNum;

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & ct;
        ar & tid; 
        ar & isShare;
        ar & share;
        ar & plainNum;
    }
};
typedef struct CipherEntry CipherEntry;

struct CipherEntryVec {
	std::vector<CipherEntry> ceVecs;

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & ceVecs;
    }
};
typedef struct CipherEntryVec CipherEntryVec;

struct PackedCipherEntry {
	uint8_t ct[MAX_CIPHER_TEXT_SIZE]; // HE ciphertext
	size_t tid; // tile index of the public key owner
    bool isShare;
    std::vector<uint64_t> share;
    int plainNum;
    std::vector<int32_t> taskId;
    std::vector<int32_t> operandId;
    std::vector<int32_t> dstTid;

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & ct;
        ar & tid; 
        ar & isShare;
        ar & share;
        ar & plainNum;
        ar & taskId;
        ar & operandId;
        ar & dstTid;
    }
};
typedef struct PackedCipherEntry PackedCipherEntry;

// struct ShareVec {
// 	std::vector<uint64_t> shares;

//     template <typename Archive> 
//     void serialize(Archive &ar, const unsigned int version) 
//     { 
//         ar & shares;
//     }
// };
// typedef struct ShareVec ShareVec;
typedef std::vector<uint64_t> ShareVec;

typedef std::vector<std::vector<double>> DoubleTensor;
typedef std::vector<std::vector<uint64_t>> ShareTensor;
typedef std::vector<ShareTensor> ShareTensorVec;

ShareTensor transpose(const ShareTensor& st);

ShareTensor toShareTensor(const ShareVec& sv);

ShareVec toShareVec(const ShareTensor& st);

ShareVec toShareVec(int hotIndex, int vecSize);

struct PosVec {
    std::vector<uint64_t> pos;

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & pos;
    }
};
typedef struct PosVec PosVec;

// struct ShareVecVec {
// 	std::vector<ShareVec> shareVecs;

//     template <typename Archive> 
//     void serialize(Archive &ar, const unsigned int version) 
//     { 
//         ar & shareVecs;
//     }
// };
// typedef struct ShareVecVec ShareVecVec;
typedef std::vector<ShareVec> ShareVecVec;

struct PublicKeyEntry {
    uint8_t pk[MAX_PUBLIC_KEY_SIZE];

    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & pk;
    }
};
typedef struct PublicKeyEntry PublicKeyEntry;

// #endif

class TaskPayload {
public:
    bool isOperandEncrypted[2];
    virtual CipherEntry splitShareFromEncryptedOperand(int operandId0 = -1) = 0;
    virtual void writeShareToOperand(uint64_t share, int operandId0 = -1, int operandId1 = -1) = 0;
    virtual ShareVec getOperandShare(int operandId0 = -1) = 0;
    virtual int getEncTid(int operandId0 = -1) = 0;
    virtual CipherEntry encryptShare(int operandId0 = -1, const uint32_t tid = -1) = 0;
    virtual void mergeEncryptedShare(CipherEntry& ce, int operandId0 = -1) = 0;
    virtual void writeEncryptedOperand(CipherEntry& ce, int operandId0 = -1) = 0;
    virtual CipherEntry* getCipherEntryPtr(int operandId0 = -1) = 0;
    virtual void unifyOperand() = 0;
    virtual void copyOperand(TaskPayload& srcTP, int srcOperandId0 = 1, int dstOperandId0 = 0) = 0;
    virtual void setCipherEntryPlainNum(int operandId0 = -1, int num = 0) = 0;
    static CryptoUtil& cryptoUtil;
    int operandNum;
    int plainNumPerCE;
    bool operandMask;
    bool useMask;
    
    TaskPayload () {
        isOperandEncrypted[0] = false;
        isOperandEncrypted[1] = false;
        operandMask = true;
        useMask = false;
    }
};

template<typename OperandType, typename IndexType>
class MinWithParent : public TaskPayload {
public:
    union {
        struct {
            OperandType plain_val;
            IndexType plain_parent;
        };
        CipherEntry enc;
    } operands[2];

    MinWithParent() {
        operandNum = 2;
        plainNumPerCE = 2;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        std::vector<uint64_t> shares;
        cryptoUtil.splitRandomShareFromCipherEntry(operands[operandId0].enc, 2, shares);
        // operands[operandId0].enc.share.resize(2);
        for (int i=0; i<2; ++i) operands[operandId0].enc.share[i] = shares[i];
        // printf("split random share MWP %lld, %lld\n", shares[0], shares[1]);
        operands[operandId0].enc.isShare = true;
        if (operands[operandId0].enc.plainNum != 2) {
            printf("error MWP plain num %d\n", operands[operandId0].enc.plainNum);
            exit(-1);
        }
        return operands[operandId0].enc;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        std::vector<uint64_t> shares = {operands[operandId0].enc.share[0], operands[operandId0].enc.share[1]};
        CipherEntry ce = cryptoUtil.encryptToCipherEntry(shares, tid);
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        std::vector<uint64_t> shares = {operands[operandId0].enc.share[0], operands[operandId0].enc.share[1]};
        cryptoUtil.mergeShareIntoCipherEntry(ce, shares, ce.tid);
        // std::copy((uint8_t*)ce.ct, (uint8_t*)(ce.ct) + MAX_CIPHER_TEXT_SIZE, (uint8_t*)(operands[operandId0].enc.ct));
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
        // operands[operandId0].enc.tid = ce.tid;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operands[operandId0].enc.share.resize(2);
        operands[operandId0].enc.share[operandId1] = share;
        operands[operandId0].enc.isShare = true;
        isOperandEncrypted[operandId0] = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operands[operandId0].enc.share[0]);
        sv.push_back(operands[operandId0].enc.share[1]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operands[operandId0].enc.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operands[operandId0].enc;
    }

    void unifyOperand() {
        operands[1] = operands[0];
        isOperandEncrypted[1] = isOperandEncrypted[0];
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        operands[dstOperandId0] = static_cast<MinWithParent&>(srcTP).operands[srcOperandId0];
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operands[operandId0].enc.plainNum = num;
    }
};

template<typename OperandType>
class Add : public TaskPayload {
public:
    union {
        OperandType plain;
        CipherEntry enc;
    } operands[2];

    Add() {
        operandNum = 2;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        uint64_t share = cryptoUtil.splitRandomShareFromCipherEntry(operands[operandId0].enc);
        // operands[operandId0].enc.share.resize(1);
        operands[operandId0].enc.share[0] = share;
        operands[operandId0].enc.isShare = true;
        return operands[operandId0].enc;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce = cryptoUtil.encryptToCipherEntry(operands[operandId0].enc.share[0], tid);
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        cryptoUtil.mergeShareIntoCipherEntry(ce, operands[operandId0].enc.share[0], ce.tid);
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operands[operandId0].enc.share.resize(1);
        operands[operandId0].enc.share[0] = share;
        operands[operandId0].enc.isShare = true;
        isOperandEncrypted[operandId0] = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operands[operandId0].enc.share[0]);
        sv.push_back(operands[operandId0].enc.share[1]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operands[operandId0].enc.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operands[operandId0].enc;
    }

    void unifyOperand() {
        operands[1] = operands[0];
        isOperandEncrypted[1] = isOperandEncrypted[0];
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        operands[dstOperandId0] = static_cast<Add&>(srcTP).operands[srcOperandId0];
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operands[operandId0].enc.plainNum = num;
    }
};

template<typename OperandType, typename IndexType>
class AddWithReplaceParent : public MinWithParent<OperandType, IndexType> {
};

template<typename OperandType, typename IndexType>
class ReplaceParent : public MinWithParent<OperandType, IndexType> {
};

template<typename OperandType1, typename OperandType2>
class AddPair : public TaskPayload {
public:
    union {
        struct {
            OperandType1 plain_a;
            OperandType2 plain_b;
        };
        CipherEntry enc;
    } operands[2];

    AddPair() {
        operandNum = 2;
        plainNumPerCE = 2;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        std::vector<uint64_t> shares;
        cryptoUtil.splitRandomShareFromCipherEntry(operands[operandId0].enc, 2, shares);
        // operands[operandId0].enc.share.resize(2);
        for (int i=0; i<2; ++i) operands[operandId0].enc.share[i] = shares[i];
        operands[operandId0].enc.isShare = true;
        return operands[operandId0].enc;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        std::vector<uint64_t> shares = {operands[operandId0].enc.share[0], operands[operandId0].enc.share[1]};
        CipherEntry ce = cryptoUtil.encryptToCipherEntry(shares, tid);
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        std::vector<uint64_t> shares = {operands[operandId0].enc.share[0], operands[operandId0].enc.share[1]};
        cryptoUtil.mergeShareIntoCipherEntry(ce, shares, ce.tid);
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operands[operandId0].enc.share.resize(2);
        operands[operandId0].enc.share[operandId1] = share;
        operands[operandId0].enc.isShare = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operands[operandId0].enc.share[0]);
        sv.push_back(operands[operandId0].enc.share[1]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operands[operandId0].enc.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operands[operandId0].enc;
    }

    void unifyOperand() {
        operands[1] = operands[0];
        isOperandEncrypted[1] = isOperandEncrypted[0];
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        operands[dstOperandId0] = static_cast<AddPair&>(srcTP).operands[srcOperandId0];
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operands[operandId0].enc.plainNum = num;
    }
};

// The second member in the pair is plain
template<typename OperandType1, typename OperandType2>
class AddMixedPair : public TaskPayload {
public:
    struct {
        union {
            OperandType1 plain_a;
            CipherEntry enc_a;
        };
        OperandType2 plain_b;
    } operands[2];

    AddMixedPair() {
        operandNum = 2;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        uint64_t share = cryptoUtil.splitRandomShareFromCipherEntry(operands[operandId0].enc_a);
        // operands[operandId0].enc_a.share.resize(1);
        operands[operandId0].enc_a.share[0] = share;
        operands[operandId0].enc_a.isShare = true;
        return operands[operandId0].enc_a;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce = cryptoUtil.encryptToCipherEntry(operands[operandId0].enc_a.share[0], tid);
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        cryptoUtil.mergeShareIntoCipherEntry(ce, operands[operandId0].enc_a.share[0], ce.tid);
        operands[operandId0].enc_a = ce;
        operands[operandId0].enc_a.isShare = false;
        isOperandEncrypted[operandId0] = true;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operands[operandId0].enc_a.share.resize(1);
        operands[operandId0].enc_a.share[0] = share;
        operands[operandId0].enc_a.isShare = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operands[operandId0].enc_a.share[0]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operands[operandId0].enc_a.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operands[operandId0].enc_a = ce;
        operands[operandId0].enc_a.isShare = false;
        isOperandEncrypted[operandId0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operands[operandId0].enc_a;
    }

    void unifyOperand() {
        operands[1] = operands[0];
        isOperandEncrypted[1] = isOperandEncrypted[0];
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        operands[dstOperandId0] = static_cast<AddMixedPair&>(srcTP).operands[srcOperandId0];
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operands[operandId0].enc_a.plainNum = num;
    }
};

template<typename OperandType>
class Div : public TaskPayload {
public:
    union {
        OperandType plain;
        CipherEntry enc;
    } operands[2];

    Div() {
        operandNum = 2;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        uint64_t share = cryptoUtil.splitRandomShareFromCipherEntry(operands[operandId0].enc);
        // operands[operandId0].enc.share.resize(1);
        operands[operandId0].enc.share[0] = share;
        operands[operandId0].enc.isShare = true;
        return operands[operandId0].enc;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce = cryptoUtil.encryptToCipherEntry(operands[operandId0].enc.share[0], tid);
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        cryptoUtil.mergeShareIntoCipherEntry(ce, operands[operandId0].enc.share[0], ce.tid);
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operands[operandId0].enc.share.resize(1);
        operands[operandId0].enc.share[0] = share;
        operands[operandId0].enc.isShare = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operands[operandId0].enc.share[0]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operands[operandId0].enc.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operands[operandId0].enc = ce;
        operands[operandId0].enc.isShare = false;
        isOperandEncrypted[operandId0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operands[operandId0].enc;
    }

    void unifyOperand() {
        operands[1] = operands[0];
        isOperandEncrypted[1] = isOperandEncrypted[0];
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        operands[dstOperandId0] = static_cast<Div&>(srcTP).operands[srcOperandId0];
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operands[operandId0].enc.plainNum = num;
    }
};

class SwapCipherEntry : public TaskPayload {
public:
    union {
        CipherEntry enc;
    } operands[2];

    SwapCipherEntry() {
        operandNum = 2;
        plainNumPerCE = 2;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0);

    CipherEntry encryptShare(int operandId0, const uint32_t tid);

    void mergeEncryptedShare(CipherEntry& ce, int operandId0);

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1);

    ShareVec getOperandShare(int operandId0);

    int getEncTid(int operandId0);

    void writeEncryptedOperand(CipherEntry& ce, int operandId0);

    CipherEntry* getCipherEntryPtr(int operandId0);

    void unifyOperand();

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0);

    void setCipherEntryPlainNum(int operandId0, int num);
};

template<typename OperandType>
class DecryptEntry : public TaskPayload {
public:
    union {
        OperandType plain;
        CipherEntry enc;
    } operand;

    DecryptEntry() {
        operandNum = 1;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        uint64_t share = cryptoUtil.splitRandomShareFromCipherEntry(operand.enc);
        // operand.enc.share.resize(1);
        operand.enc.share[0] = share;
        operand.enc.isShare = true;
        return operand.enc;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce = operand.enc;
        ce.tid = tid;
        // No encryption here
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        operand.plain = cryptoUtil.template mergeShareAs<OperandType>(ce.share[0], operand.enc.share[0]);
        isOperandEncrypted[0] = false;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operand.enc.share.resize(1);
        operand.enc.share[0] = share;
        operand.enc.isShare = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operand.enc.share[0]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operand.enc.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operand.enc = ce;
        operand.enc.isShare = false;
        isOperandEncrypted[0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operand.enc;
    }

    void unifyOperand() {
        // Nothing to do
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        // Nothing to do
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operand.enc.plainNum = num;
    }
};

template<typename OperandType1, typename OperandType2>
class DecryptEntryPair : public TaskPayload {
public:
    union {
        struct {
            OperandType1 plain_a;
            OperandType2 plain_b;
        };
        CipherEntry enc;
    } operand;

    DecryptEntryPair() {
        operandNum = 1;
        plainNumPerCE = 2;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        std::vector<uint64_t> shares;
        cryptoUtil.splitRandomShareFromCipherEntry(operand.enc, 2, shares);
        // operand.enc.share.resize(2);
        for (int i=0; i<2; ++i) operand.enc.share[i] = shares[i];
        operand.enc.isShare = true;
        return operand.enc;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce = operand.enc;
        ce.tid = tid;
        // No encryption here
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        std::vector<uint64_t> shareVec0 = {ce.share[0], ce.share[1]};
        std::vector<uint64_t> shareVec1 = {operand.enc.share[0], operand.enc.share[1]};
        cryptoUtil.mergeShareVec(shareVec0, shareVec1);
        operand.plain_a = cryptoUtil.template decodeFixedPointAs<OperandType1>(shareVec0[0]);
        operand.plain_b = cryptoUtil.template decodeFixedPointAs<OperandType1>(shareVec0[1]);
        isOperandEncrypted[operandId0] = false;
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        // operand.enc.share.resize(2);
        operand.enc.share[operandId1] = share;
        operand.enc.isShare = true;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv.push_back(operand.enc.share[0]);
        sv.push_back(operand.enc.share[1]);
        return sv;
    }

    int getEncTid(int operandId0) {
        return operand.enc.tid;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
        operand.enc = ce;
        operand.enc.isShare = false;
        isOperandEncrypted[0] = true;
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return &operand.enc;
    }

    void unifyOperand() {
        // Nothing to do
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        // Nothing to do
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
        operand.enc.plainNum = num;
    }
};

class GCNVectorScale : public TaskPayload {
public:

    ShareVec embeddingVec;
    uint64_t scaler0;
    uint64_t scaler1;

    GCNVectorScale() {
        operandNum = 3;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        CipherEntry ce;
        return ce;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce;
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        return;
    }

    void writeShareToOperand(const ShareVec& embeddingVecShare, uint64_t s0, uint64_t s1) {
        embeddingVec = embeddingVecShare;
        scaler0 = s0;
        scaler1 = s1;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        return sv;
    }

    int getEncTid(int operandId0) {
        return -1;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return NULL;
    }

    void unifyOperand() {
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
    }
};

class GCNVectorAddition : public TaskPayload {
public:
    ShareVec operands[2];

    GCNVectorAddition() {
        operandNum = 2;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        CipherEntry ce;
        return ce;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce;
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        return;
    }

    void writeShareToOperand(const ShareVec& share, int operandId0) {
        operands[operandId0] = share;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        sv = operands[operandId0];
        return sv;
    }

    int getEncTid(int operandId0) {
        return -1;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return NULL;
    }

    void unifyOperand() {
        operands[1] = operands[0];
        isOperandEncrypted[1] = isOperandEncrypted[0];
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
        operands[dstOperandId0] = static_cast<GCNVectorAddition&>(srcTP).operands[srcOperandId0];
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
    }
};

class GCNForwardNN : public TaskPayload {
public:

    ShareVec ah;
    ShareTensor weight;

    ShareVec z;
    ShareVec new_h;

    GCNForwardNN() {
        operandNum = 2;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        CipherEntry ce;
        return ce;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce;
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        return;
    }

    void writeShareToOperand(const ShareVec& ahShare, const ShareTensor& weightShare) {
        ah = ahShare;
        weight = weightShare;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        return sv;
    }

    int getEncTid(int operandId0) {
        return -1;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return NULL;
    }

    void unifyOperand() {
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
    }
};

class GCNForwardNNPrediction : public TaskPayload {
public:

    ShareVec ah;
    ShareTensor weight;
    ShareVec y;

    ShareVec z;
    ShareVec p;
    ShareVec p_minus_y;

    GCNForwardNNPrediction() {
        operandNum = 2;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        CipherEntry ce;
        return ce;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce;
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        return;
    }

    void writeShareToOperand(const ShareVec& ahShare, const ShareTensor& weightShare, const ShareVec yShare) {
        ah = ahShare;
        weight = weightShare;
        y = yShare;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        return sv;
    }

    int getEncTid(int operandId0) {
        return -1;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return NULL;
    }

    void unifyOperand() {
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
    }
};

class GCNBackwardNNInit : public TaskPayload {
public:

    ShareVec p_minus_y;
    ShareVec ah_t;
    ShareTensor weight_t;

    ShareTensor d;
    ShareVec g;
    
    GCNBackwardNNInit() {
        operandNum = 3;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        CipherEntry ce;
        return ce;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce;
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        return;
    }

    void writeShareToOperand(const ShareVec& pMinusYShare, const ShareVec& ahTShare, const ShareTensor& weightTShare) {
        p_minus_y = pMinusYShare;
        ah_t = ahTShare;
        weight_t = weightTShare;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        return sv;
    }

    int getEncTid(int operandId0) {
        return -1;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return NULL;
    }

    void unifyOperand() {
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
    }
};

class GCNBackwardNN : public TaskPayload {
public:

    ShareVec a_t_g;
    ShareVec ah_t;
    ShareVec z;
    ShareTensor weight_t;

    ShareTensor d;
    ShareVec g;
    
    GCNBackwardNN() {
        operandNum = 4;
        plainNumPerCE = 1;
    }

    CipherEntry splitShareFromEncryptedOperand(int operandId0) {
        CipherEntry ce;
        return ce;
    }

    CipherEntry encryptShare(int operandId0, const uint32_t tid) {
        CipherEntry ce;
        return ce;
    }

    void mergeEncryptedShare(CipherEntry& ce, int operandId0) {
        return;
    }

    void writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
        return;
    }

    void writeShareToOperand(const ShareVec& aTGShare, const ShareVec& ahTShare, const ShareVec& zShare, const ShareTensor& weightTShare) {
        a_t_g = aTGShare;
        ah_t = ahTShare;
        weight_t = weightTShare;
        z = zShare;
        return;
    }

    ShareVec getOperandShare(int operandId0) {
        ShareVec sv;
        return sv;
    }

    int getEncTid(int operandId0) {
        return -1;
    }

    void writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    }

    CipherEntry* getCipherEntryPtr(int operandId0) {
        return NULL;
    }

    void unifyOperand() {
    }

    void copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
    }

    void setCipherEntryPlainNum(int operandId0, int num) {
    }
};

struct Task {
    TASK_TYPE type;
    uint64_t vertexIndex; // Task destination vertex index
    bool finished;
    void* buf; // Task payload (content)

    uint64_t srcIndex = (uint64_t)-1; // Optional, task source vertex index

    uint64_t dstTid = 0; // Optional, task destination/cooperation tile index
    // For inter-graph edge, the task destination tile index is the owner of the destination vertex.
    // For intra-graph edge, the task destination is the encryptor of the source vertex data.
    uint64_t srcTid = 0; // Optional, task source tile index

    std::atomic<uint8_t> readyOperandNum{0}; // Optional

    bool isFinal = false;

    std::mutex mtx;

    bool isDummy = false;

    Task(const Task& t) {
        type = t.type;
        vertexIndex = t.vertexIndex;
        finished = t.finished;
        buf = t.buf;
        srcIndex = t.srcIndex;
        dstTid = t.dstTid;
        srcTid = t.srcTid;
        readyOperandNum = t.readyOperandNum.load();
        isFinal = t.isFinal;
        isDummy = t.isDummy;
    }

    Task& operator=(const Task& t) {
        type = t.type;
        vertexIndex = t.vertexIndex;
        finished = t.finished;
        buf = t.buf;
        srcIndex = t.srcIndex;
        dstTid = t.dstTid;
        srcTid = t.srcTid;
        readyOperandNum = t.readyOperandNum.load();
        isFinal = t.isFinal;
        isDummy = t.isDummy;
        return *this;        
    }

    Task() {
        type = (TASK_TYPE)0;
        vertexIndex = 0;
        finished = false;
        buf = NULL;
        srcIndex = 0;
        dstTid = 0;
        srcTid = 0;
        readyOperandNum = 0;
        isFinal = false;
        isDummy = false;
    }

    Task(TASK_TYPE a, uint64_t b, bool c, void* d, uint64_t e, uint64_t f, uint64_t g):
        type(a), vertexIndex(b), finished(c), buf(d), srcIndex(e), dstTid(f), srcTid(g), readyOperandNum(0), isFinal(false), isDummy(false) {
    }

    Task(TASK_TYPE a, uint64_t b, bool c, void* d, uint64_t e, uint64_t f, uint64_t g, uint64_t h, bool i):
        type(a), vertexIndex(b), finished(c), buf(d), srcIndex(e), dstTid(f), srcTid(g), readyOperandNum(h), isFinal(i), isDummy(false) {
    }

    Task(TASK_TYPE a, uint64_t b, bool c, void* d, uint64_t e, uint64_t f, uint64_t g, uint64_t h, bool i, bool isdm):
        type(a), vertexIndex(b), finished(c), buf(d), srcIndex(e), dstTid(f), srcTid(g), readyOperandNum(h), isFinal(i), isDummy(isdm) {
    }

#ifdef SSHEBACKEND
    template <typename Archive> 
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & type;
        ar & vertexIndex; 
        ar & finished;
        ar & srcIndex;
        ar & dstTid;
        ar & srcTid;
        // ar & readyOperandNum;
        ar & isFinal;
    }
#endif

    void delete_task_content_buf() {
        switch (type) {
            case ADD_PAIR_DOUBLE_UINT:
                delete (AddPair<double, uint32_t>*)buf;
                break;
            case ADD_MIXED_PAIR_DOUBLE_UINT:
                delete (AddMixedPair<double, uint32_t>*)buf;
                break;
            case MIN_UINT_WITH_PARENT:
                delete (MinWithParent<uint32_t, uint64_t>*)buf;
                break;
            case ADD_DOUBLE:
                delete (Add<double>*)buf;
                break;
            case ADD_ULONG:
                delete (Add<uint64_t>*)buf;
                break;
            case ADD_UINT:
                delete (Add<uint32_t>*)buf;  
                break; 
            case ADD_UINT_WITH_REPLACE_PARENT:
                delete (AddWithReplaceParent<uint32_t, uint64_t>*)buf;  
                break;
            case UINT_REPLACE_PARENT:
                delete (ReplaceParent<uint32_t, uint64_t>*)buf;  
                break;     
            case DIV_DOUBLE:
                delete (Div<double>*)buf;
                break;
            case DEC_UINT:
                delete (DecryptEntry<uint32_t>*)buf;
                break;
            case DEC_ULONG:
                delete (DecryptEntry<uint64_t>*)buf;
                break;
            case DEC_DOUBLE:
                delete (DecryptEntry<double>*)buf;
                break;
            case DEC_PAIR_UINT_ULONG:
                delete (DecryptEntryPair<uint32_t, uint64_t>*)buf;
                break;
            case SWAP_CIPHER_ENTRY:
                delete (SwapCipherEntry*)buf;
                break;
            case GCN_VECTOR_SCALE:
                delete (GCNVectorScale*)buf;
                break;
            case GCN_VECTOR_ADDITION:
                delete (GCNVectorAddition*)buf;
                break;
            case GCN_FORWARD_NN:
                delete (GCNForwardNN*)buf;
                break;
            case GCN_FORWARD_NN_PREDICTION:
                delete (GCNForwardNNPrediction*)buf;
                break;
            case GCN_BACKWARD_NN_INIT:
                delete (GCNBackwardNNInit*)buf;
                break;
            case GCN_BACKWARD_NN:
                delete (GCNBackwardNN*)buf;
                break;
            default:
                break;
        }
    }

    ~Task() {}
};

typedef struct Task Task;

#endif