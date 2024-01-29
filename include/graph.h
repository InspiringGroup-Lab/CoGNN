#ifndef GRAPH_H_
#define GRAPH_H_

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <queue>
#include "graph_common.h"
#include "task.h"

namespace GraphGASLite {

class VertexIdxRepType;
typedef IndexType<uint64_t, VertexIdxRepType> VertexIdx;

class EdgeIdxRepType;
typedef IndexType<uint64_t, EdgeIdxRepType> EdgeIdx;

class TileIdxRepType;
typedef IndexType<uint64_t, TileIdxRepType> TileIdx;

class DegreeRepType;
typedef CountType<uint32_t, DegreeRepType> DegreeCount;

template<typename VertexDataType, typename UpdateDataType, typename EdgeWeightType>
class GraphTile;

template<typename VertexDataType, typename UpdateDataType, typename EdgeWeightType>
class Vertex {
public:
    typedef VertexDataType DataType;
    typedef UpdateDataType UpdateType;

    VertexIdx vid() const { return vid_; }

    DegreeCount inDeg() const { return inDeg_; }
    DegreeCount outDeg() const { return outDeg_; }
    void inDegInc(const DegreeCount& d = 1u) {
        inDeg_ += d;
    }
    void outDegInc(const DegreeCount& d = 1u) {
        outDeg_ += d;
    }

    VertexDataType& data() { return data_; }
    const VertexDataType& data() const { return data_; }

    bool hasUpdate() const { return hasUpdate_; }
    UpdateType accUpdate() const { return accUpdate_; }
    std::queue<UpdateType>& updateQueue() { return updateQueue_; }

    /**
     * Append a new update to the queue of mirror vertex, waiting for entering task queue.
     */
    void appendUpdateToQueue(UpdateType update) {
        updateQueue_.push(update);
        hasUpdate_ = true;
        updateNum_ += 1;
    }

    /**
     * Set accUpdate value.
     */
    void setAccUpdate(const UpdateType& update) {
        accUpdate_ = update;
    }

    /**
     * Update queue is empty or not
     */
    bool isUpdateQueueEmpty() {
        // return updateQueue_.empty();
        return (updateNum_ == 0);
    }

    /**
     * Generate computation tasks using accUpdate and updateQueue_
     */
    void getTasks(std::queue<Task>& taskq) {
        if (updateQueue_.size() <= 0) {
            // printf("updateQueue_.size() = %d\n", updateQueue_.size());
            // printf("updateNum_ = %d\n", updateNum_);
            throw QueueException("Empty update queue!\n");
        } else if (updateQueue_.size() == 1) {
            auto task = UpdateType::genTask(accUpdate_, updateQueue_.front(), (uint64_t)(vid_), (uint64_t)tid_, (uint64_t)tid_, true);
            updateQueue_.pop();
            taskq.push(task);
            updateNum_ -= 1;
        } else {
            // Divide and Conquer for queue size >= 2
            while (updateQueue_.size() >= 2) {
                UpdateType update0 = updateQueue_.front();
                updateQueue_.pop();
                auto task = UpdateType::genTask(update0, updateQueue_.front(), (uint64_t)(vid_), (uint64_t)tid_, (uint64_t)tid_, false);
                updateQueue_.pop();
                taskq.push(task);
                updateNum_ -= 1;
            }
        }

        return;
    }

    Task getTask() {
        if (updateQueue_.size() <= 0) {
            // printf("updateQueue_.size() = %d\n", updateQueue_.size());
            // printf("updateNum_ = %d\n", updateNum_);
            throw QueueException("Empty update queue!\n");
        }
        auto task = UpdateType::genTask(accUpdate_, updateQueue_.front(), (uint64_t)(vid_), (uint64_t)tid_, (uint64_t)tid_, true);
        updateQueue_.pop();
        updateNum_ -= 1;
        
        return task;
    }

    /**
     * Write task result to mirror vertex accUpdate_
     */
    void writeTaskResult(const struct Task& task) {
        if (task.isFinal) {
            setAccUpdate(UpdateType::getTaskResult(task));
        } else {
            updateQueue_.push(UpdateType::getTaskResult(task));
        }
        return;
    }

    /**
     * Delete all updates, i.e., reset accUpdate.
     */
    void updateDelAll() {
        accUpdate_ = UpdateType();
        while (!updateQueue_.empty()) {
            updateQueue_.pop();
        }
        hasUpdate_ = false;
        updateNum_ = 0;
    }

    void pushToTaskq(Task task) {
        taskq_.push_back(task);
    }

    std::vector<Task>& getTaskq() {
        return taskq_;
    }

    std::vector<std::vector<Task>>& getTaskvs() {
        return taskvs_;
    }

    std::vector<Task>& getPlainTaskv() {
        return plain_taskv_;
    }

    std::vector<uint64_t>& getSrcVertexv() {
        return srcvv_;
    }

    void pushToSrcVertexv(uint64_t vid) {
        srcvv_.push_back(vid);
    }

    std::vector<EdgeWeightType>& getIncomingEdgev() {
        return incomingEdgev_;
    }

    void pushToIncomingEdgev(EdgeWeightType ew) {
        incomingEdgev_.push_back(ew);
    }

    std::vector<bool>& getIsSrcDummyv() {
        return isSrcDummyv_;
    }

    void pushToIsSrcDummyv(bool flag) {
        isSrcDummyv_.push_back(flag);
    }

    void setReorderedIndex(uint64_t index) {
        reorderedIndex_ = index;
    }

    uint64_t getReorderedIndex() {
        return reorderedIndex_;
    }

    void setIsBorderVertex(bool val) {
        isBorderVertex_ = val;
    }

    bool isBorderVertex() {
        return isBorderVertex_;
    }

private:
    const VertexIdx vid_;
    const TileIdx tid_;
    const TileIdx remoteMirrorVertexPos;
    DegreeCount inDeg_;
    DegreeCount outDeg_;
    VertexDataType data_;
    UpdateType accUpdate_;
    std::queue<UpdateType> updateQueue_;
    bool hasUpdate_;
    bool isBorderVertex_;
    int updateNum_;
    std::vector<Task> taskq_;
    std::vector<std::vector<Task>> taskvs_;
    std::vector<Task> plain_taskv_;
    std::vector<uint64_t> srcvv_; // source vertex index vector
    std::vector<bool> isSrcDummyv_;
    std::vector<EdgeWeightType> incomingEdgev_;
    uint64_t reorderedIndex_;

private:
    template<typename VDT, typename UDT, typename EWT>
    friend class GraphTile;

    template<typename... Args>
    explicit Vertex(const VertexIdx& vid, const TileIdx& tid, Args&&... args)
        : vid_(vid), tid_(tid), inDeg_(0), outDeg_(0), hasUpdate_(false), isBorderVertex_(false), updateNum_(0), accUpdate_(UpdateType()),
          data_(vid, std::forward<Args>(args)...)
    {
        // Nothing else to do.
    }

    Vertex(const Vertex&) = delete;
    Vertex& operator=(const Vertex&) = delete;
    Vertex(Vertex&&) = delete;
    Vertex& operator=(Vertex&&) = delete;
    bool operator==(const Vertex&) const = delete;
};

template<typename UpdateDataType, typename EdgeWeightType>
class MirrorVertex {
public:
    typedef UpdateDataType UpdateType;

    VertexIdx vid() const { return vid_; }

    TileIdx masterTileId() const { return masterTileId_; }

    bool hasUpdate() const { return hasUpdate_; }

    UpdateType& accUpdate() { return accUpdate_; }

    std::queue<UpdateType>& updateQueue() { return updateQueue_; }

    /**
     * Add a new update, i.e., merge into accUpdate.
     */
    void updateNew(const UpdateType& update) {
        accUpdate_ += update;
        hasUpdate_ = true;
    }

    /**
     * Append a new update to the queue of mirror vertex, waiting for entering task queue.
     */
    void appendUpdateToQueue(UpdateType update) {
        updateQueue_.push(update);
        hasUpdate_ = true;
        updateNum_ += 1;
    }

    /**
     * Set accUpdate value.
     */
    void setAccUpdate(const UpdateType& update) {
        accUpdate_ = update;
    }

    /**
     * Update queue is empty or not
     */
    bool isUpdateQueueEmpty() {
        // return updateQueue_.empty();
        return (updateNum_ == 0);
    }

    /**
     * Generate computation tasks using accUpdate and updateQueue_
     */
    void getTasks(std::queue<Task>& taskq) {
        if (updateQueue_.size() <= 0) {
            // printf("updateQueue_.size() = %d\n", updateQueue_.size());
            // printf("updateNum_ = %d\n", updateNum_);
            throw QueueException("Empty update queue!\n");
        } else if (updateQueue_.size() == 1) {
            auto task = UpdateType::genTask(accUpdate_, updateQueue_.front(), (uint64_t)(vid_), (uint64_t)curTileId_, (uint64_t)masterTileId_, true);
            updateQueue_.pop();
            taskq.push(task);
            updateNum_ -= 1;
        } else {
            // Divide and Conquer for queue size >= 2
            while (updateQueue_.size() >= 2) {
                UpdateType update0 = updateQueue_.front();
                updateQueue_.pop();
                auto task = UpdateType::genTask(update0, updateQueue_.front(), (uint64_t)(vid_), (uint64_t)curTileId_, (uint64_t)masterTileId_, false);
                updateQueue_.pop();
                taskq.push(task);
                updateNum_ -= 1;
            }
        }
        
        return;
    }

    Task getTask() {
        if (updateQueue_.size() <= 0) {
            throw QueueException("Empty update queue!\n");
        }
        auto task = UpdateType::genTask(accUpdate_, updateQueue_.front(), (uint64_t)(vid_), (uint64_t)curTileId_, (uint64_t)masterTileId_, true);
        updateQueue_.pop();
        updateNum_ -= 1;
        
        return task;
    }

    /**
     * Write task result to mirror vertex accUpdate_
     */
    void writeTaskResult(const struct Task& task) {
        if (task.isFinal) {
            setAccUpdate(UpdateType::getTaskResult(task));
        } else {
            updateQueue_.push(UpdateType::getTaskResult(task)); // Do not use appendUpdateToQueue here.
        }
        return;
    }

    /**
     * Delete all updates, i.e., reset accUpdate.
     */
    void updateDelAll() {
        accUpdate_ = UpdateType();
        while (!updateQueue_.empty()) {
            updateQueue_.pop();
        }
        hasUpdate_ = false;
        updateNum_ = 0;
    }

    /**
     * When sync vertex degree with other tiles, clear the mirror vertex
     * acc degree after propagating to master vertex.
     */
    DegreeCount accDeg() const { return accDeg_; }
    void accDegDel() {
        accDeg_ = 0;
    }

    void pushToTaskq(Task task) {
        taskq_.push_back(task);
    }

    std::vector<Task>& getTaskq() {
        return taskq_;
    }

    std::vector<std::vector<Task>>& getTaskvs() {
        return taskvs_;
    }

    std::vector<Task>& getPlainTaskv() {
        return plain_taskv_;
    }

    std::vector<uint64_t>& getSrcVertexv() {
        return srcvv_;
    }

    void pushToSrcVertexv(uint64_t vid) {
        srcvv_.push_back(vid);
    }

    std::vector<EdgeWeightType>& getIncomingEdgev() {
        return incomingEdgev_;
    }

    void pushToIncomingEdgev(EdgeWeightType ew) {
        incomingEdgev_.push_back(ew);
    }

    std::vector<bool>& getIsSrcDummyv() {
        return isSrcDummyv_;
    }

    void pushToIsSrcDummyv(bool flag) {
        isSrcDummyv_.push_back(flag);
    }

private:
    const VertexIdx vid_;
    const TileIdx masterTileId_;
    const TileIdx curTileId_;

    bool hasUpdate_;
    int updateNum_;

    // union {
    UpdateType accUpdate_;
    // Only used at vertex degree initialization. Must be cleared before real processing.
    DegreeCount accDeg_;
    // };

    std::queue<UpdateType> updateQueue_;
    std::vector<Task> taskq_;
    std::vector<std::vector<Task>> taskvs_;
    std::vector<Task> plain_taskv_;
    std::vector<uint64_t> srcvv_; // source vertex index vector
    std::vector<bool> isSrcDummyv_;
    std::vector<EdgeWeightType> incomingEdgev_;

private:
    template<typename VDT, typename UDT, typename EWT>
    friend class GraphTile;

    MirrorVertex(const VertexIdx& vid, const TileIdx& masterTileId, const TileIdx& curTileId)
        : vid_(vid), masterTileId_(masterTileId), curTileId_(curTileId), accDeg_(0), hasUpdate_(false), updateNum_(0), accUpdate_(UpdateType())
          // accUpdate_ initialized after accDeg_ is used.
    {
        // Nothing else to do.
    }

    void accDegInc(const DegreeCount& d = 1u) {
        accDeg_ += d;
    }

    MirrorVertex(const MirrorVertex&) = delete;
    MirrorVertex& operator=(const MirrorVertex&) = delete;
    MirrorVertex(MirrorVertex&&) = delete;
    MirrorVertex& operator=(MirrorVertex&&) = delete;
    bool operator==(const MirrorVertex&) const = delete;
};

template<typename EdgeWeightType = uint32_t>
class Edge {
public:
    typedef EdgeWeightType WeightType;

    VertexIdx srcId() const { return srcId_; }
    VertexIdx dstId() const { return dstId_; }

    EdgeWeightType weight() const { return weight_; }
    void weightIs(const EdgeWeightType& weight) {
        weight_ = weight;
    }
    EdgeWeightType& weight() { return weight_; }

private:
    VertexIdx srcId_;
    VertexIdx dstId_;
    EdgeWeightType weight_;

private:
    template<typename VDT, typename UDT, typename EWT>
    friend class GraphTile;

    Edge(const VertexIdx& srcId, const VertexIdx& dstId, const EdgeWeightType& weight)
        : srcId_(srcId), dstId_(dstId), weight_(weight)
    {
        // Nothing else to do.
    }

    /**
     * ``Less-than'' function used to sort edges, first source index, then dest index.
     *
     * Avoid overloading ==, <, etc..
     */
    static bool lessFunc(const Edge& e1, const Edge& e2) {
        if (e1.srcId_ == e2.srcId_) return e1.dstId_ < e2.dstId_;
        return e1.srcId_ < e2.srcId_;
    }

    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;
    bool operator==(const Edge&) const = delete;

public:
    /**
     * Move-assignable and move-constructible, to allow sorting.
     */
    Edge(Edge&&) = default;
    Edge& operator=(Edge&&) = default;
};


template<typename VertexDataType, typename UpdateDataType, typename EdgeWeightType = uint32_t>
class GraphTile {
public:
    typedef UpdateDataType UpdateType;

    typedef Vertex<VertexDataType, UpdateDataType, EdgeWeightType> VertexType;
    typedef Edge<EdgeWeightType> EdgeType;
    typedef MirrorVertex<UpdateDataType, EdgeWeightType> MirrorVertexType;

    typedef std::unordered_map< VertexIdx, Ptr<VertexType>, std::hash<VertexIdx::Type> > VertexMap;
    typedef std::vector< EdgeType > EdgeList;
    typedef std::unordered_map< VertexIdx, Ptr<MirrorVertexType>, std::hash<VertexIdx::Type> > MirrorVertexMap;

    typedef typename VertexMap::iterator VertexIter;
    typedef typename VertexMap::const_iterator VertexConstIter;
    typedef typename EdgeList::iterator EdgeIter;
    typedef typename EdgeList::const_iterator EdgeConstIter;
    typedef typename MirrorVertexMap::iterator MirrorVertexIter;
    typedef typename MirrorVertexMap::const_iterator MirrorVertexConstIter;

public:
    explicit GraphTile(const TileIdx& tid)
        : tid_(tid), vertices_(), edges_(), edgeSorted_(false), finalized_(false),
          vidLastVisited_(-1), vLastVisited_(nullptr), mvidLastVisited_(-1), mvLastVisited_(nullptr)
    {
        // Nothing else to do.
    }

    TileIdx tid() const { return tid_; }

    /* Vertices. */

    template<typename... Args>
    void vertexNew(const VertexIdx& vid, Args&&... args) {
        checkNotFinalized(__func__);
        auto vertex = Ptr<VertexType>(new VertexType(vid, tid_, std::forward<Args>(args)...));
        if (vertices_.insert( typename VertexMap::value_type(vid, vertex) ).second == false) {
            throw KeyInUseException(std::to_string(vid));
        }
    }

    // Ptr<VertexType> vertex(const VertexIdx& vid) {
    //     if (!finalized_ || vid != vidLastVisited_) {
    //         auto it = vertices_.find(vid);
    //         if (it != vertices_.end()) {
    //             vLastVisited_ = it->second;
    //         } else {
    //             vLastVisited_ = nullptr;
    //         }
    //         vidLastVisited_ = vid;
    //     }
    //     return vLastVisited_;
    // }

    Ptr<VertexType> vertex(const VertexIdx& vid) {
        Ptr<VertexType> cur_vertex;
        auto it = vertices_.find(vid);
        if (it != vertices_.end()) {
            cur_vertex = it->second;
        } else {
            cur_vertex = nullptr;
        }
        return cur_vertex;
    }

    size_t vertexCount() const { return vertices_.size(); }

    bool hasVertex(const VertexIdx& vid) const {
        return vertices_.find(vid) != vertices_.end();
    }

    inline VertexConstIter vertexIter() const {
        return vertices_.cbegin();
    }
    inline VertexConstIter vertexIterEnd() const {
        return vertices_.cend();
    }

    inline VertexIter vertexIter() {
        return vertices_.begin();
    }
    inline VertexIter vertexIterEnd() {
        return vertices_.end();
    }

    /* Mirror vertices. */

    Ptr<MirrorVertexType> mirrorVertex(const VertexIdx& vid) {
        if (!finalized_ || vid != mvidLastVisited_) {
            auto it = mirrorVertices_.find(vid);
            if (it != mirrorVertices_.end()) {
                mvLastVisited_ = it->second;
            } else {
                mvLastVisited_ = nullptr;
            }
            mvidLastVisited_ = vid;
        }
        return mvLastVisited_;
    }

    inline MirrorVertexConstIter mirrorVertexIter() const {
        return mirrorVertices_.cbegin();
    }
    inline MirrorVertexConstIter mirrorVertexIterEnd() const {
        return mirrorVertices_.cend();
    }

    inline MirrorVertexIter mirrorVertexIter() {
        return mirrorVertices_.begin();
    }
    inline MirrorVertexIter mirrorVertexIterEnd() {
        return mirrorVertices_.end();
    }

    /* Edges. */

    void edgeNew(const VertexIdx& srcId, const VertexIdx& dstId, const TileIdx& dstTileId, const EdgeWeightType& weight) {
        checkNotFinalized(__func__);
        // Source vertex must be in this tile.
        if (vertices_.count(srcId) == 0) {
            throw RangeException(std::to_string(srcId));
        }
        // Destination vertex can be in different tile.
        if (dstTileId == tid_ && vertices_.count(dstId) == 0) {
            throw RangeException(std::to_string(dstId));
        }
        if (dstTileId != tid_ && mirrorVertices_.count(dstId) == 0) {
            // Create mirror vertex if destination vertex is in different tile.
            auto mirrorVertex = Ptr<MirrorVertexType>(new MirrorVertexType(dstId, dstTileId, tid_));
            mirrorVertices_.insert( typename MirrorVertexMap::value_type(dstId, mirrorVertex) );
        }
        // Repeating edges with the same srcId and dstId are accepted.
        // Use move constructor.
        edges_.push_back(EdgeType(srcId, dstId, weight));
        edgeSorted_ &= EdgeType::lessFunc(edges_[edges_.size()-2], edges_[edges_.size()-1]);
        // Increment degree.
        vertex(srcId)->outDegInc();
        if (dstTileId != tid_) {
            mirrorVertex(dstId)->accDegInc();
        } else {
            vertex(dstId)->inDegInc();
        }
    }

    bool edgeSorted() const { return edgeSorted_; }
    void edgeSortedIs(bool sorted) {
        if (!edgeSorted_ && sorted) {
            std::sort(edges_.begin(), edges_.end(), EdgeType::lessFunc);
            edgeSorted_ = true;
        }
    }

    inline EdgeConstIter edgeIter() const {
        return edges_.cbegin();
    }
    inline EdgeConstIter edgeIterEnd() const {
        return edges_.cend();
    }

    inline EdgeIter edgeIter() {
        return edges_.begin();
    }
    inline EdgeIter edgeIterEnd() {
        return edges_.end();
    }

    size_t edgeCount() const { return edges_.size(); }

    bool finalized() const { return finalized_; }
    void finalizedIs(const bool finalized) {
        if (!finalized_ && finalized) {
            // Finalize the graph tile.

            // Sort edge list.
            edgeSortedIs(true);

            // Check mirror vertex acc degree has been propagated to other tiles and cleared.
            for (auto mvIter = mirrorVertexIter(); mvIter != mirrorVertexIterEnd(); ++mvIter) {
                const auto mv = mvIter->second;
                if (mv->accDeg() != 0) {
                    throw PermissionException("Cannot finalize graph tile " + std::to_string(tid_)
                            + " due to uncleared mirror vertex " + std::to_string(mv->vid())
                            + " acc degree.");
                }
                // Reset application update.
                mv->updateDelAll();
            }

        }
        finalized_ = finalized;
    }

private:
    const TileIdx tid_;

    VertexMap vertices_;
    EdgeList edges_;

    MirrorVertexMap mirrorVertices_;

    bool edgeSorted_;

    /**
     * Finalize graph tile to prevent further changes on graph structure.
     * Any mutator to add/delete vertex/edge is not allowed after finalizing,
     * unless explicitly de-finalize. However, mutators that do not change
     * graph topology are allowed, e.g, edgeSortedIs().
     */
    bool finalized_;

    // Cached vertex and mirror vertex.
    VertexIdx vidLastVisited_;
    Ptr<VertexType> vLastVisited_;
    VertexIdx mvidLastVisited_;
    Ptr<MirrorVertexType> mvLastVisited_;

private:
    void checkNotFinalized(const string& funcName) {
        if (finalized_) {
            throw PermissionException(funcName + ": Graph tile has already been finalized.");
        }
    }

    GraphTile(const GraphTile&) = delete;
    GraphTile& operator=(const GraphTile&) = delete;
    GraphTile(GraphTile&&) = delete;
    GraphTile& operator=(GraphTile&&) = delete;
    bool operator==(const GraphTile&) const = delete;

};

} // namespace GraphGASLite

#endif // GRAPH_H_
