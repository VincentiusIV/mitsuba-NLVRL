#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/timer.h>
#include <functional>

NAMESPACE_BEGIN(mitsuba)

#define SIZE_T_FMT "%Iu"

template <typename DataType, typename IndexType>
void permute_inplace(DataType *data, std::vector<IndexType> &perm) {
    for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] != i) {
            /* The start of a new cycle has been found. Save
               the value at this position, since it will be
               overwritten */
            IndexType j     = (IndexType) i;
            DataType curval = data[i];

            do {
                /* Shuffle backwards */
                IndexType k = perm[j];
                data[j]     = data[k];

                /* Also fix the permutations on the way */
                perm[j] = j;
                j       = k;

                /* Until the end of the cycle has been found */
            } while (perm[j] != i);

            /* Fix the final position with the saved value */
            data[j] = curval;
            perm[j] = j;
        }
    }
}

/**
 * \brief Simple kd-tree node for use with \ref PointKDTree.
 *
 * This class is an example of how one might write a space-efficient kd-tree
 * node that is compatible with the \ref PointKDTree class. The implementation
 * supports associating a custom data record with each node and works up to
 * 16 dimensions.
 *
 * \tparam _PointType Underlying point data type (e.g. \ref TPoint3<float>)
 * \tparam _DataRecord Custom storage that should be associated with each
 *  tree node
 *
 * \ingroup libcore
 * \sa PointKDTree
 * \sa LeftBalancedKDNode
 */
template <typename _PointType, typename _DataRecord> struct SimpleKDNode {
    typedef _PointType PointType;
    typedef _DataRecord DataRecord;
    typedef uint32_t IndexType;
    using Scalar = scalar_t<PointType>;

    enum { ELeafFlag = 0x10, EAxisMask = 0x0F };

    static const bool leftBalancedLayout = false;

    PointType position;
    IndexType right;
    DataRecord data;
    uint8_t flags;

    /// Initialize a KD-tree node
    inline SimpleKDNode() : position((Scalar) 0), right(0), data(), flags(0) {}
    /// Initialize a KD-tree node with the given data record
    inline SimpleKDNode(const DataRecord &data)
        : position((Scalar) 0), right(0), data(data), flags(0) {}

    /// Given the current node's index, return the index of the right child
    inline IndexType getRightIndex(IndexType self) const { return right; }
    /// Given the current node's index, set the right child index
    inline void setRightIndex(IndexType self, IndexType value) {
        right = value;
    }

    /// Given the current node's index, return the index of the left child
    inline IndexType getLeftIndex(IndexType self) const { return self + 1; }
    /// Given the current node's index, set the left child index
    inline void setLeftIndex(IndexType self, IndexType value) {
#if defined(MTS_DEBUG)
        if (value != self + 1)
            Log(LogLevel::Error, "SimpleKDNode::setLeftIndex(): Internal error!");
#endif
    }

    /// Check whether this is a leaf node
    inline bool isLeaf() const { return flags & (uint8_t) ELeafFlag; }
    /// Specify whether this is a leaf node
    inline void setLeaf(bool value) {
        if (value)
            flags |= (uint8_t) ELeafFlag;
        else
            flags &= (uint8_t) ~ELeafFlag;
    }

    /// Return the split axis associated with this node
    inline uint16_t getAxis() const { return flags & (uint8_t) EAxisMask; }
    /// Set the split flags associated with this node
    inline void setAxis(uint8_t axis) {
        flags = (flags & (uint8_t) ~EAxisMask) | axis;
    }

    /// Return the position associated with this node
    inline const PointType &getPosition() const { return position; }
    /// Set the position associated with this node
    inline void setPosition(const PointType &value) { position = value; }

    /// Return the data record associated with this node
    inline DataRecord &getData() { return data; }
    /// Return the data record associated with this node (const version)
    inline const DataRecord &getData() const { return data; }
    /// Set the data record associated with this node
    inline void setData(const DataRecord &val) { data = val; }
};

/**
 * \brief Generic multi-dimensional kd-tree data structure for point data
 *
 * This class organizes point data in a hierarchical manner so various
 * types of queries can be performed efficiently. It supports several
 * heuristics for building ``good'' trees, and it is oblivious to the
 * data layout of the nodes themselves.
 *
 * Note that this class is meant for point data only --- for things that
 * have some kind of spatial extent, the classes \ref GenericKDTree and
 * \ref ShapeKDTree will be more appropriate.
 *
 * \tparam _NodeType Underlying node data structure. See \ref SimpleKDNode as
 * an example for the required public interface
 *
 * \ingroup libcore
 * \see SimpleKDNode
 */
template <typename Float, typename Spectrum, typename _NodeType> class PointKDTree {
public:
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    typedef _NodeType NodeType;
    typedef typename Point3f PointType;
    typedef typename NodeType::IndexType IndexType;
    typedef typename PointType::Scalar Scalar;
    typedef typename Vector3f VectorType;
    typedef BoundingBox<PointType> AABBType;

    /// Supported tree construction heuristics
    enum EHeuristic {
        /// Create a balanced tree by splitting along the median
        EBalanced = 0,

        /// Create a left-balanced tree
        ELeftBalanced,

        /**
         * \brief Use the sliding midpoint tree construction rule. This
         * ensures that cells do not become overly elongated.
         */
        ESlidingMidpoint,

        /**
         * \brief Choose the split plane by optimizing a cost heuristic
         * based on the ratio of voxel volumes.
         *
         * Note that Mitsuba's implementation of this heuristic is not
         * particularly optimized --- the tree construction construction
         * runs time O(n (log n)^2) instead of O(n log n).
         */
        EVoxelVolume
    };

    /// Result data type for k-nn queries
    struct SearchResult {
        float distSquared;
        IndexType index;

        inline SearchResult() {}

        inline SearchResult(const SearchResult &searchResult) : SearchResult(searchResult.distSquared, searchResult.index) {}

        inline SearchResult(float distSquared, IndexType index)
            : distSquared(distSquared), index(index) {}

        std::string toString() const {
            std::ostringstream oss;
            oss << "SearchResult[distance=" << std::sqrt(distSquared)
                << ", index=" << index << "]";
            return oss.str();
        }

        inline bool operator==(const SearchResult &r) const {
            return distSquared == r.distSquared && index == r.index;
        }

    };

    struct SearchResultComparator {
    public:
        inline bool operator()(const SearchResult &a,
                               const SearchResult &b) const {
            return a.distSquared < b.distSquared;
        }
    };

public:
    size_t getSize() {
        size_t total = 0;
        total += sizeof(m_nodes) + sizeof(NodeType) * m_nodes.size();
        total += sizeof(m_aabb);
        total += sizeof(m_depth);
        return total;
    }
    /**
     * \brief Create an empty KD-tree that can hold the specified
     * number of points
     */
    inline PointKDTree(size_t nodes         = 0,
                       EHeuristic heuristic = ESlidingMidpoint)
        : m_nodes(nodes), m_heuristic(heuristic), m_depth(0) {}

    // =============================================================
    //! @{ \name \c stl::vector-like interface
    // =============================================================
    /// Clear the kd-tree array
    inline void clear() {
        m_nodes.clear();
        m_aabb.reset();
    }
    /// Resize the kd-tree array
    inline void resize(size_t size) { m_nodes.resize(size); }
    /// Reserve a certain amount of memory for the kd-tree array
    inline void reserve(size_t size) { m_nodes.reserve(size); }
    /// Return the size of the kd-tree
    inline size_t size() const { return m_nodes.size(); }
    /// Return the capacity of the kd-tree
    inline size_t capacity() const { return m_nodes.capacity(); }
    /// Append a kd-tree node to the node array
    inline void push_back(const NodeType &node) {
        m_nodes.push_back(node);
        m_aabb.expand(node.getPosition());
    }
    /// Return one of the KD-tree nodes by index
    inline NodeType &operator[](size_t idx) { return m_nodes[idx]; }
    /// Return one of the KD-tree nodes by index (const version)
    inline const NodeType &operator[](size_t idx) const { return m_nodes[idx]; }
    //! @}
    // =============================================================

    /// Set the AABB of the underlying point data
    inline void setAABB(const AABBType &aabb) { m_aabb = aabb; }
    /// Return the AABB of the underlying point data
    inline const AABBType &getAABB() const { return m_aabb; }
    /// Return the depth of the constructed KD-tree
    inline size_t getDepth() const { return m_depth; }
    /// Set the depth of the constructed KD-tree (be careful with this)
    inline void setDepth(size_t depth) { m_depth = depth; }

    /// Construct the KD-tree hierarchy
    void build(bool recomputeAABB = false) {
        if (m_nodes.size() == 0) {
            Log(LogLevel::Error, "build(): kd-tree is empty!");
            return;
        }


        if (recomputeAABB) {
            m_aabb.reset();
            for (size_t i = 0; i < m_nodes.size(); ++i)
                m_aabb.expand(m_nodes[i].getPosition());
        }

        /* Instead of shuffling around the node data itself, only modify
           an indirection table initially. Once the tree construction
           is done, this table will contain a indirection that can then
           be applied to the data in one pass */
        std::vector<IndexType> indirection(m_nodes.size());
        for (size_t i = 0; i < m_nodes.size(); ++i)
            indirection[i] = (IndexType) i;

        m_depth = 0;
        if (NodeType::leftBalancedLayout) {
            std::vector<IndexType> permutation(m_nodes.size());
            buildLB(0, 1, indirection.begin(), indirection.begin(),
                    indirection.end(), permutation);

            permute_inplace(&m_nodes[0], permutation);
        } else {
            build(1, indirection.begin(), indirection.begin(),
                  indirection.end());

            permute_inplace(&m_nodes[0], indirection);
        }

    }

    /**
     * \brief Run a k-nearest-neighbor search query
     *
     * \param p Search position
     * \param sqrSearchRadius
     *      Specifies the squared maximum search radius. This parameter can be
     * used to restrict the k-nn query to a subset of the data -- it that is not
     *      desired, simply set it to positive infinity. After the query
     *      finishes, the parameter value will correspond to the (potentially
     * lower) maximum query radius that was necessary to ensure that the number
     * of results did not exceed \c k. \param k Maximum number of search results
     * \param results Target array for search results. Must
     *      contain storage for at least \c k+1 entries!
     *      (one extra entry is needed for shuffling data around)
     * \return The number of search results (equal to \c k or less)
     */
    size_t nnSearch(const PointType &p, float &_sqrSearchRadius, size_t k,
                    SearchResult *results, size_t &M) const {
        if (m_nodes.size() == 0)
            return 0;

        IndexType *stack = new IndexType[m_depth + 1];
        IndexType index = 0, stackPos = 1;
        float sqrSearchRadius = _sqrSearchRadius;
        size_t resultCount    = 0;
        bool isHeap           = false;
        stack[0]              = 0;

        while (stackPos > 0) {
            const NodeType &node = m_nodes[index];
            IndexType nextIndex;

            /* Recurse on inner nodes */
            if (!node.isLeaf()) {
                float distToPlane =
                    p[node.getAxis()] - node.getPosition()[node.getAxis()];

                bool searchBoth = distToPlane * distToPlane <= sqrSearchRadius;

                if (distToPlane > 0) {
                    /* The search query is located on the right side of the
                       split. Search this side first. */
                    if (hasRightChild(index)) {
                        if (searchBoth)
                            stack[stackPos++] = node.getLeftIndex(index);
                        nextIndex = node.getRightIndex(index);
                    } else if (searchBoth) {
                        nextIndex = node.getLeftIndex(index);
                    } else {
                        nextIndex = stack[--stackPos];
                    }
                } else {
                    /* The search query is located on the left side of the
                       split. Search this side first. */
                    if (searchBoth && hasRightChild(index))
                        stack[stackPos++] = node.getRightIndex(index);

                    nextIndex = node.getLeftIndex(index);
                }
            } else {
                nextIndex = stack[--stackPos];
            }

            /* Check if the current point is within the query's search radius */
            PointType between = (node.getPosition() - p);
            float pointDistSquared = squared_norm(between);

            if (pointDistSquared < sqrSearchRadius) {
                ++M; 
                /* Switch to a max-heap when the available search
                   result space is exhausted */
                if (resultCount < k) {
                    /* There is still room, just add the point to
                       the search result list */
                    results[resultCount++] =
                        SearchResult(pointDistSquared, index);
                } else {
                    if (!isHeap) {
                        /* Establish the max-heap property */
                        std::make_heap(results, results + resultCount, SearchResultComparator());
                        isHeap = true;
                    }
                    SearchResult *end = results + resultCount + 1;

                    /* Add the new point, remove the one that is farthest away
                     */
                    results[resultCount] = SearchResult(pointDistSquared, index);
                    std::push_heap(results, end, SearchResultComparator());
                    std::pop_heap(results, end, SearchResultComparator());

                    /* Reduce the search radius accordingly */
                    sqrSearchRadius = results[0].distSquared;
                }
            }
            index = nextIndex;
        }
        delete stack;
        _sqrSearchRadius = sqrSearchRadius;
        return resultCount;
    }

    template <typename Functor> 
    size_t beamQuery(const PointType &p, Vector3f dir, Float farT, float searchRadius, Functor &functor) const {
        if (m_nodes.size() == 0)
            return 0;

        IndexType *stack = (IndexType *) alloca((m_depth + 1) * sizeof(IndexType));
        IndexType index = 0, stackPos = 1, found = 0;
        float distSquared = searchRadius * searchRadius;
        stack[0]          = 0;

        Vector3f invDir = 1.0f / dir;

        while (stackPos > 0) {
            const NodeType &node = m_nodes[index];
            IndexType nextIndex;

            Vector3f mins = (node->minBounds - pos) * invDir;
            Vector3f maxs = (node->maxBounds - pos) * invDir;
            float minT = max(invDir[0] > 0.0f ? mins[0] : maxs[0], 
                invDir[1] > 0.0f ? mins[1] : maxs[1], 
                invDir[2] > 0.0f ? mins[2] : maxs[2]);
            float maxT = min(invDir[0] > 0.0f ? maxs[0] : mins[0], 
                invDir[1] > 0.0f ? maxs[1] : mins[1], 
                invDir[2] > 0.0f ? maxs[2] : mins[2]);

            /* Recurse on inner nodes */
            if (!node.isLeaf()) {
                float distToPlane = p[node.getAxis()] - node.getPosition()[node.getAxis()];

                bool searchBoth = distToPlane * distToPlane <= distSquared;

                if (distToPlane > 0) {
                    /* The search query is located on the right side of the
                       split. Search this side first. */
                    if (hasRightChild(index)) {
                        if (searchBoth)
                            stack[stackPos++] = node.getLeftIndex(index);
                        nextIndex = node.getRightIndex(index);
                    } else if (searchBoth) {
                        nextIndex = node.getLeftIndex(index);
                    } else {
                        nextIndex = stack[--stackPos];
                    }
                } else {
                    /* The search query is located on the left side of the
                       split. Search this side first. */
                    if (searchBoth && hasRightChild(index))
                        stack[stackPos++] = node.getRightIndex(index);

                    nextIndex = node.getLeftIndex(index);
                }
            } else {
                nextIndex = stack[--stackPos];
            }

            /* Check if the current point is within the query's search radius */
            const float pointDistSquared = squared_norm(node.getPosition() - p);

            if (pointDistSquared < distSquared) {
                ++found;
                functor(node);
            }

            index = nextIndex;
        }
        return (size_t) found;
    }

    /**
     * \brief Run a k-nearest-neighbor search query and record statistics
     *
     * \param p Search position
     * \param sqrSearchRadius
     *      Specifies the squared maximum search radius. This parameter can be
     * used to restrict the k-nn query to a subset of the data -- it that is not
     *      desired, simply set it to positive infinity. After the query
     *      finishes, the parameter value will correspond to the (potentially
     * lower) maximum query radius that was necessary to ensure that the number
     * of results did not exceed \c k. \param k Maximum number of search results
     * \param results
     *      Target array for search results. Must contain
     *      storage for at least \c k+1 entries! (one
     *      extra entry is needed for shuffling data around)
     * \return The number of used traversal steps
     */
    size_t nnSearchCollectStatistics(const PointType &p, float &sqrSearchRadius,
                                     size_t k, std::vector<SearchResult> &results,
                                     size_t &traversalSteps) const {
        traversalSteps = 0;

        if (m_nodes.size() == 0)
            return 0;

        IndexType *stack =
            (IndexType *) alloca((m_depth + 1) * sizeof(IndexType));
        IndexType index = 0, stackPos = 1;
        size_t resultCount = 0;
        bool isHeap        = false;
        stack[0]           = 0;

        while (stackPos > 0) {
            const NodeType &node = m_nodes[index];
            ++traversalSteps;
            IndexType nextIndex;

            /* Recurse on inner nodes */
            if (!node.isLeaf()) {
                float distToPlane =
                    p[node.getAxis()] - node.getPosition()[node.getAxis()];

                bool searchBoth = distToPlane * distToPlane <= sqrSearchRadius;

                if (distToPlane > 0) {
                    /* The search query is located on the right side of the
                       split. Search this side first. */
                    if (hasRightChild(index)) {
                        if (searchBoth)
                            stack[stackPos++] = node.getLeftIndex(index);
                        nextIndex = node.getRightIndex(index);
                    } else if (searchBoth) {
                        nextIndex = node.getLeftIndex(index);
                    } else {
                        nextIndex = stack[--stackPos];
                    }
                } else {
                    /* The search query is located on the left side of the
                       split. Search this side first. */
                    if (searchBoth && hasRightChild(index))
                        stack[stackPos++] = node.getRightIndex(index);

                    nextIndex = node.getLeftIndex(index);
                }
            } else {
                nextIndex = stack[--stackPos];
            }

            /* Check if the current point is within the query's search radius */
            const float pointDistSquared =
                (node.getPosition() - p).lengthSquared();

            if (resultCount < k) {
                    /* There is still room, just add the point to
                       the search result list */
                    results[resultCount++] =
                        SearchResult(pointDistSquared, index);
                } else {
                    if (!isHeap) {
                        /* Establish the max-heap property */
                        std::make_heap(results, results + resultCount,
                                       SearchResultComparator());
                        isHeap = true;
                    }
                    SearchResult *end = results + resultCount + 1;

                    /* Add the new point, remove the one that is farthest away
                     */
                    results[resultCount] =
                        SearchResult(pointDistSquared, index);
                    std::push_heap(results, end, SearchResultComparator());
                    std::pop_heap(results, end, SearchResultComparator());


                    /* Reduce the search radius accordingly */
                    sqrSearchRadius = results[0].distSquared;
                }
            index = nextIndex;
        }
        return resultCount;
    }

    /**
     * \brief Run a k-nearest-neighbor search query without any
     * search radius threshold
     *
     * \param p Search position
     * \param k Maximum number of search results
     * \param results
     *      Target array for search results. Must contain
     *      storage for at least \c k+1 entries! (one
     *      extra entry is needed for shuffling data around)
     * \return The number of used traversal steps
     */

    inline size_t nnSearch(const PointType &p, size_t k,
                           SearchResult *results) const {
        float searchRadiusSqr = std::numeric_limits<float>::infinity();
        return nnSearch(p, searchRadiusSqr, k, results);
    }

    /**
     * \brief Execute a search query and run the specified functor on them,
     * which potentially modifies the nodes themselves
     *
     * The functor must have an operator() implementation, which accepts
     * a \a NodeType as its argument.
     *
     * \param p Search position
     * \param functor Functor to be called on each search result
     * \param searchRadius Search radius
     * \return The number of functor invocations
     */
    template <typename Functor>
    size_t executeModifier(const PointType &p, float searchRadius,
                           Functor &functor) {
        if (m_nodes.size() == 0)
            return 0;

        IndexType *stack =
            (IndexType *) alloca((m_depth + 1) * sizeof(IndexType));
        size_t index = 0, stackPos = 1, found = 0;
        float distSquared = searchRadius * searchRadius;
        stack[0]          = 0;

        while (stackPos > 0) {
            NodeType &node = m_nodes[index];
            IndexType nextIndex;

            /* Recurse on inner nodes */
            if (!node.isLeaf()) {
                float distToPlane =
                    p[node.getAxis()] - node.getPosition()[node.getAxis()];

                bool searchBoth = distToPlane * distToPlane <= distSquared;

                if (distToPlane > 0) {
                    /* The search query is located on the right side of the
                       split. Search this side first. */
                    if (hasRightChild(index)) {
                        if (searchBoth)
                            stack[stackPos++] = node.getLeftIndex(index);
                        nextIndex = node.getRightIndex(index);
                    } else if (searchBoth) {
                        nextIndex = node.getLeftIndex(index);
                    } else {
                        nextIndex = stack[--stackPos];
                    }
                } else {
                    /* The search query is located on the left side of the
                       split. Search this side first. */
                    if (searchBoth && hasRightChild(index))
                        stack[stackPos++] = node.getRightIndex(index);

                    nextIndex = node.getLeftIndex(index);
                }
            } else {
                nextIndex = stack[--stackPos];
            }

            /* Check if the current point is within the query's search radius */
            const float pointDistSquared =
                (node.getPosition() - p).lengthSquared();

            if (pointDistSquared < distSquared) {
                functor(node);
                ++found;
            }

            index = nextIndex;
        }
        return found;
    }

    /**
     * \brief Execute a search query and run the specified functor on them
     *
     * The functor must have an operator() implementation, which accepts
     * a constant reference to a \a NodeType as its argument.
     *
     * \param p Search position
     * \param functor Functor to be called on each search result
     * \param searchRadius  Search radius
     * \return The number of functor invocations
     */
    template <typename Functor>
    size_t executeQuery(const PointType &p, float searchRadius,
                        Functor &functor) const {
        if (m_nodes.size() == 0)
            return 0;

        IndexType *stack =
            (IndexType *) alloca((m_depth + 1) * sizeof(IndexType));
        IndexType index = 0, stackPos = 1, found = 0;
        float distSquared = searchRadius * searchRadius;
        stack[0]          = 0;

        while (stackPos > 0) {
            const NodeType &node = m_nodes[index];
            IndexType nextIndex;

            /* Recurse on inner nodes */
            if (!node.isLeaf()) {
                float distToPlane =
                    p[node.getAxis()] - node.getPosition()[node.getAxis()];

                bool searchBoth = distToPlane * distToPlane <= distSquared;

                if (distToPlane > 0) {
                    /* The search query is located on the right side of the
                       split. Search this side first. */
                    if (hasRightChild(index)) {
                        if (searchBoth)
                            stack[stackPos++] = node.getLeftIndex(index);
                        nextIndex = node.getRightIndex(index);
                    } else if (searchBoth) {
                        nextIndex = node.getLeftIndex(index);
                    } else {
                        nextIndex = stack[--stackPos];                        
                    }
                } else {
                    /* The search query is located on the left side of the
                       split. Search this side first. */
                    if (searchBoth && hasRightChild(index))
                        stack[stackPos++] = node.getRightIndex(index);

                    nextIndex = node.getLeftIndex(index);
                }
            } else {
                nextIndex = stack[--stackPos];
            }

            /* Check if the current point is within the query's search radius */
            const float pointDistSquared = squared_norm(node.getPosition() - p);

            if (pointDistSquared < distSquared) {
                ++found;
                functor(node);
            }

            index = nextIndex;
        }
        return (size_t) found;
    }

    /**
     * \brief Run a search query
     *
     * \param p Search position
     * \param results Index list of search results
     * \param searchRadius  Search radius
     * \return The number of functor invocations
     */
    size_t search(const PointType &p, float searchRadius,
                  std::vector<IndexType> &results) const {
        if (m_nodes.size() == 0)
            return 0;

        IndexType *stack =
            (IndexType *) alloca((m_depth + 1) * sizeof(IndexType));
        IndexType index = 0, stackPos = 1, found = 0;
        float distSquared = searchRadius * searchRadius;
        stack[0]          = 0;

        while (stackPos > 0) {
            const NodeType &node = m_nodes[index];
            IndexType nextIndex;

            /* Recurse on inner nodes */
            if (!node.isLeaf()) {
                float distToPlane =
                    p[node.getAxis()] - node.getPosition()[node.getAxis()];

                bool searchBoth = distToPlane * distToPlane <= distSquared;

                if (distToPlane > 0) {
                    /* The search query is located on the right side of the
                       split. Search this side first. */
                    if (hasRightChild(index)) {
                        if (searchBoth)
                            stack[stackPos++] = node.getLeftIndex(index);
                        nextIndex = node.getRightIndex(index);
                    } else if (searchBoth) {
                        nextIndex = node.getLeftIndex(index);
                    } else {
                        nextIndex = stack[--stackPos];
                    }
                } else {
                    /* The search query is located on the left side of the
                       split. Search this side first. */
                    if (searchBoth && hasRightChild(index))
                        stack[stackPos++] = node.getRightIndex(index);

                    nextIndex = node.getLeftIndex(index);
                }
            } else {
                nextIndex = stack[--stackPos];
            }

            /* Check if the current point is within the query's search radius */
            const float pointDistSquared =
                (node.getPosition() - p).lengthSquared();

            if (pointDistSquared < distSquared) {
                ++found;
                results.push_back(index);
            }

            index = nextIndex;
        }
        return (size_t) found;
    }

    /**
     * \brief Return whether or not the inner node of the
     * specified index has a right child node.
     *
     * This function is available for convenience and abstracts away some
     * details about the underlying node representation.
     */
    inline bool hasRightChild(IndexType index) const {
        if (NodeType::leftBalancedLayout) {
            return 2 * index + 2 < m_nodes.size();
        } else {
            return m_nodes[index].getRightIndex(index) != 0;
        }
    }



protected:

    struct CoordinateOrdering {
    public:
        inline CoordinateOrdering(const std::vector<NodeType> &nodes, int axis)
            : m_nodes(nodes), m_axis(axis) {}
        inline bool operator()(const IndexType &i1, const IndexType &i2) const {
            return m_nodes[i1].getPosition()[m_axis] <
                   m_nodes[i2].getPosition()[m_axis];
        }

    private:
        const std::vector<NodeType> &m_nodes;
        int m_axis;
    };

    struct LessThanOrEqual {
    public:
        inline LessThanOrEqual(const std::vector<NodeType> &nodes, int axis,
                               Scalar value)
            : m_nodes(nodes), m_axis(axis), m_value(value) {}
        inline bool operator()(const IndexType &i) const {
            return m_nodes[i].getPosition()[m_axis] <= m_value;
        }

    private:
        const std::vector<NodeType> &m_nodes;
        int m_axis;
        Scalar m_value;
    };

    /**
     * Given a number of entries, this method calculates the number of nodes
     * nodes on the left subtree of a left-balanced tree. There are two main
     * cases here:
     *
     * 1) It is possible to completely fill the left subtree
     * 2) It doesn't work - the last level contains too few nodes, e.g :
     *         O
     *        / \
     *       O   O
     *      /
     *     O
     *
     * The function assumes that "count" > 1.
     */
    inline IndexType leftSubtreeSize(IndexType count) const {
        /* Layer 0 contains one node */
        IndexType p = 1;

        /* Traverse downwards until the first incompletely
           filled tree level is encountered */
        while (2 * p <= count)
            p *= 2;

        /* Calculate the number of filled slots in the last level */
        IndexType remaining = count - p + 1;

        if (2 * remaining < p) {
            /* Case 2: The last level contains too few nodes. Remove
               overestimate from the left subtree node count and add
               the remaining nodes */
            p = (p >> 1) + remaining;
        }

        return p - 1;
    }

    /// Left-balanced tree construction routine
    void buildLB(IndexType idx, size_t depth,
                 typename std::vector<IndexType>::iterator base,
                 typename std::vector<IndexType>::iterator rangeStart,
                 typename std::vector<IndexType>::iterator rangeEnd,
                 typename std::vector<IndexType> &permutation) {
        m_depth = std::max(depth, m_depth);

        IndexType count = (IndexType) (rangeEnd - rangeStart);
        Assert(count > 0);

        if (count == 1) {
            /* Create a leaf node */
            m_nodes[*rangeStart].setLeaf(true);
            permutation[idx] = *rangeStart;
            return;
        }

        typename std::vector<IndexType>::iterator split =
            rangeStart + leftSubtreeSize(count);
        int axis = m_aabb.major_axis();
        std::nth_element(rangeStart, split, rangeEnd,
                         CoordinateOrdering(m_nodes, axis));

        NodeType &splitNode = m_nodes[*split];
        splitNode.setAxis(axis);
        splitNode.setLeaf(false);
        permutation[idx] = *split;

        /* Recursively build the children */
        Scalar temp      = m_aabb.max[axis],
               splitPos  = splitNode.getPosition()[axis];
        m_aabb.max[axis] = splitPos;
        buildLB(2 * idx + 1, depth + 1, base, rangeStart, split, permutation);
        m_aabb.max[axis] = temp;

        if (split + 1 != rangeEnd) {
            temp             = m_aabb.min[axis];
            m_aabb.min[axis] = splitPos;
            buildLB(2 * idx + 2, depth + 1, base, split + 1, rangeEnd,
                    permutation);
            m_aabb.min[axis] = temp;
        }
    }

    /// Default tree construction routine
    void build(size_t depth, typename std::vector<IndexType>::iterator base,
               typename std::vector<IndexType>::iterator rangeStart,
               typename std::vector<IndexType>::iterator rangeEnd) {
        m_depth = std::max(depth, m_depth);

        IndexType count = (IndexType) (rangeEnd - rangeStart);
        Assert(count > 0);

        if (count == 1) {
            /* Create a leaf node */
            m_nodes[*rangeStart].setLeaf(true);
            return;
        }

        int axis = 0;
        typename std::vector<IndexType>::iterator split;

        switch (m_heuristic) {
            case EBalanced: {
                split = rangeStart + count / 2;
                axis  = m_aabb.major_axis();
                std::nth_element(rangeStart, split, rangeEnd,
                                 CoordinateOrdering(m_nodes, axis));
            }; break;

            case ELeftBalanced: {
                split = rangeStart + leftSubtreeSize(count);
                axis  = m_aabb.major_axis();
                std::nth_element(rangeStart, split, rangeEnd,
                                 CoordinateOrdering(m_nodes, axis));
            }; break;

            case ESlidingMidpoint: {
                /* Sliding midpoint rule: find a split that is close to the
                 * spatial median */
                axis = m_aabb.major_axis();

                Scalar midpoint =
                    (Scalar) 0.5f * (m_aabb.max[axis] + m_aabb.min[axis]);

                size_t nLT =
                    std::count_if(rangeStart, rangeEnd,
                                  LessThanOrEqual(m_nodes, axis, midpoint));

                /* Re-adjust the split to pass through a nearby point */
                split = rangeStart + nLT;

                if (split == rangeStart)
                    ++split;
                else if (split == rangeEnd)
                    --split;

                std::nth_element(rangeStart, split, rangeEnd,
                                 CoordinateOrdering(m_nodes, axis));
            }; break;

            case EVoxelVolume: {
                float bestCost = std::numeric_limits<float>::infinity();

                for (int dim = 0; dim < 3; ++dim) {
                    std::sort(rangeStart, rangeEnd,
                              CoordinateOrdering(m_nodes, dim));

                    size_t numLeft = 1, numRight = count - 2;
                    AABBType leftAABB(m_aabb), rightAABB(m_aabb);
                    float invVolume = 1.0f / m_aabb.volume();
                    for (typename std::vector<IndexType>::iterator it =
                             rangeStart + 1;
                         it != rangeEnd; ++it) {
                        ++numLeft;
                        --numRight;
                        float pos         = m_nodes[*it].getPosition()[dim];
                        leftAABB.max[dim] = rightAABB.min[dim] = pos;

                        float cost = (numLeft * leftAABB.volume() +
                                      numRight * rightAABB.volume()) *
                                     invVolume;
                        if (cost < bestCost) {
                            bestCost = cost;
                            axis     = dim;
                            split    = it;
                        }
                    }
                }
                std::nth_element(rangeStart, split, rangeEnd,
                                 CoordinateOrdering(m_nodes, axis));
            }; break;
        }

        NodeType &splitNode = m_nodes[*split];
        splitNode.setAxis(axis);
        splitNode.setLeaf(false);

        if (split + 1 != rangeEnd)
            splitNode.setRightIndex((IndexType) (rangeStart - base),
                                    (IndexType) (split + 1 - base));
        else
            splitNode.setRightIndex((IndexType) (rangeStart - base), 0);

        splitNode.setLeftIndex((IndexType) (rangeStart - base),
                               (IndexType) (rangeStart + 1 - base));
        std::iter_swap(rangeStart, split);

        /* Recursively build the children */
        Scalar temp      = m_aabb.max[axis],
               splitPos  = splitNode.getPosition()[axis];
        m_aabb.max[axis] = splitPos;
        build(depth + 1, base, rangeStart + 1, split + 1);
        m_aabb.max[axis] = temp;

        if (split + 1 != rangeEnd) {
            temp             = m_aabb.min[axis];
            m_aabb.min[axis] = splitPos;
            build(depth + 1, base, split + 1, rangeEnd);
            m_aabb.min[axis] = temp;
        }
    }


protected:
    std::vector<NodeType> m_nodes;
    AABBType m_aabb;
    EHeuristic m_heuristic;
    size_t m_depth;
};

NAMESPACE_END(mitsuba)
