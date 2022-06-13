#pragma once

#include <enoki/fwd.h>
#include <enoki/stl.h>
#include <mitsuba/core/bbox.h>

#include "kd_tree.h"
#include "vrl_struct.h"

#include <queue>

NAMESPACE_BEGIN(mitsuba)

#define VERBOSE_UPPERBOUND_PROBLEM 0
#define COMPUTE_DISTANCE_PRECISE 1
#define DEBUG_UPPER_BOUND 0
#define DEBUG_VRL_LC 1

// StatsCounter VRLLightcutNbEval("VRL", "Lightcut nb eval: ", EAverage);
// StatsCounter VRLBoundProblem("VRL", "Lightcut bound issue", EPercentage);
// StatsCounter VRLBetterDistance("VRL", "Better distance computation", EPercentage);
// StatsCounter VRLUpperBoundDistance("VRL", "Average upperbound distance", EAverage);
// StatsCounter VRLDebugUpperBoundDistance("VRL", "Debug: Average upperbound distance", EAverage);
// StatsCounter VRLDebugPDFBound("VRL", "Debug: PDF bound ratio", EAverage);

template <typename Float, typename Spectrum> class VRLLightCut {
public:
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    typedef VRL<Float, Spectrum> VRL;

    // Node structure
    struct Node {
        Node *parent      = nullptr;
        Node *children[2] = { nullptr, nullptr };
        bool isLeaf       = true;
        // Node information for omnidirectional
        bool isSame[2] = { false, false };

        BoundingBox<Point3f> aabb; //< AABB of the cluster

        VRL represent; //< representative
        Float maxVRLLenght;

        // For better computation of the upper bound
        std::vector<Node *> nodes;

        // Used for efficient clustering
        KdNode<Float, Spectrum, Node> *kdnode = nullptr;
        bool valid                            = true;

        Node(const VRL &_vrl) : represent(_vrl) {
            aabb.expand(represent.origin);
            aabb.expand(represent.origin + represent.direction * represent.length);

            // Max length
            maxVRLLenght = represent.length;
            nodes.push_back(this); // Add itself
        }

        ~Node() {
            if (children[0])
                delete children[0];
            if (children[1])
                delete children[1];
        }

        struct DistanceResult {
            Float distance;
            bool n1_aabb;
        };

        DistanceResult dist(const Node *n2) const {
            const Node *n1 = this;
            Float total    = hmax(n1->represent.flux + n2->represent.flux);
            auto aabb      = n1->aabb;
            aabb.expand(n2->aabb);
            auto diag = squared_norm(aabb.extents());

            return DistanceResult{ diag * total, true };
        };

        Node(Node *n1, Node *n2, Float v, bool n1_merge) {
            if (n1_merge) {
                aabb = n1->aabb;
                aabb.expand(n2->aabb);
            } else {
                aabb = n2->aabb;
                aabb.expand(n1->aabb);
            }
            isLeaf       = false;
            maxVRLLenght = std::max(n1->maxVRLLenght, n2->maxVRLLenght);

            // Update the node representation
            children[0] = n1;
            children[1] = n2;
            n1->parent  = this;
            n2->parent  = this;

            // Keep the nodes list to a certain point
            // TODO: Not memory efficient
            for (auto n : n1->nodes) {
                nodes.push_back(n);
            }
            for (auto n : n2->nodes) {
                nodes.push_back(n);
            }

            // Select randomly the representative child
            Float leftWM  = (hmax(n1->represent.flux) * n1->represent.length) / 1000;
            Float rightWM = (hmax(n2->represent.flux) * n2->represent.length) / 1000;
            Float totalWM = leftWM + rightWM;
            Float ratio   = leftWM / totalWM;
            int index     = v < ratio ? 0 : 1;

            // Update the VRL, scale it flux and mark which child is representative
            represent = children[index]->represent; // Make a copy of the VRL
            ratio     = children[1 - index]->represent.length / children[index]->represent.length;
            represent.flux += children[1 - index]->represent.flux * ratio;
            isSame[index]     = true;
            isSame[1 - index] = false;
        }
    };

public:
    VRLLightCut(const Scene *scene, const std::vector<VRL> &vrls, Sampler *sampler, int thresholdBetterDist, Float thresholdError)
        : m_thresholdBetterDist(thresholdBetterDist), m_errorRatio(thresholdError) {
        std::vector<Node *> nodes;
        nodes.reserve(vrls.size());
        for (const VRL &vrl : vrls) {
            nodes.push_back(new Node(vrl));
        }

        if (true/*vrls.size() < 1024*/) {
            m_root = buildLightTree(nodes, sampler, true, true);
        } else {
            m_root = buildTreeKDAlternate(nodes, sampler);
        }

            // This strategy is too slow.
         //m_root = buildTreeStable(scene, nodes, sampler);
    }

    /// Release all memory
    virtual ~VRLLightCut() {
        // TODO: Need to recursively delete all the nodes
        if (m_root != 0)
            delete m_root;
    }

    /// Compute the beam radiance estimate for the given ray segment and medium
    struct LCQuery {
        const Ray3f ray;
        Sampler *sampler;
        size_t nb_evaluation;
    };

    Spectrum query(const Scene *scene, Sampler *sampler, LCQuery &query, size_t &nb_BBIntersection, bool useUniformSampling, bool useDirectIllum, UInt32 channel) const {
        if (m_root == nullptr)
            return Spectrum(0.f);

        // Priority queue
        struct CutCluster {
            const Node *node;
            Float bound;
            Spectrum estimate;
        };
        auto compare_cut = [](const CutCluster &a, const CutCluster &b) { return a.bound < b.bound; };
        typedef std::priority_queue<CutCluster, std::vector<CutCluster>, decltype(compare_cut)> LightCutPriorityQueue;
        LightCutPriorityQueue refiningCut(compare_cut);

        // TODO: This might be not necessary if we only use the query to accumulate the values...
        // TODO: However, it might be too complex to do it
        // Get the root node and put it inside the queue
        refiningCut.emplace(CutCluster{ m_root, getClusterUpperBound(scene, sampler, *m_root, query.ray, channel, nb_BBIntersection),
                                        m_root->represent.getContrib(scene, useUniformSampling, useDirectIllum, query.ray, query.ray.maxt, query.sampler, channel) });
        Spectrum Li = refiningCut.top().estimate;

#if DEBUG_VRL_LC
        /*std::ostringstream stome;
        stome << "estimate: " << Li;
        Log(LogLevel::Info, stome.str().c_str());*/
#endif

        // Starting to dive inside the light tree
        while (!refiningCut.empty()) {
            // + Get node with largest error
            CutCluster current_element = refiningCut.top();
            refiningCut.pop();
            if (current_element.bound < hmax((m_errorRatio * Li))) {
                break; // Finished as all the element will be smaller
            }

            // TODO: Accumulate the childs contribution
            // TODO: If their contribution is higher, then the algorithm does not work
            Li -= current_element.estimate;
            const Node *current_cluster = current_element.node;

            // Debugging upperbound
#if DEBUG_UPPER_BOUND
            {
                Float debug_upperbound = getClusterUpperBound(Node(current_cluster->vrl), query.ray);
                // Note that the float is clamped to int
                if (debug_upperbound < 100 && current_element.estimate.max() > 0.01) {
                    if (debug_upperbound < current_element.estimate.max()) {
                        Float diff = current_element.estimate.max() - debug_upperbound;
                        //                    Log(LogLevel::Info, "Diff: %f | Upper: %f | Real: %f", diff / current_element.estimate.max(), debug_upperbound, current_element.estimate.max());
                    }
                    VRLDebugUpperBoundDistance += debug_upperbound / current_element.estimate.max();
                    VRLDebugUpperBoundDistance.incrementBase();
                }
            }
#endif

            Spectrum childrenEstimates(0.f);
            for (int i = 0; i < 2; i++) {
                if (current_cluster->children[i]) {
                    const Node *child_cluster = current_cluster->children[i];
                    Spectrum estimate         = child_cluster->represent.flux;
                    if (current_cluster->isSame[i]) {
                        estimate /= current_cluster->represent.flux;
                        estimate *= current_element.estimate;
                    } else {
                        query.nb_evaluation += 1;
                        estimate = child_cluster->represent.getContrib(scene, useUniformSampling, useDirectIllum, query.ray, query.ray.maxt, query.sampler, channel);
                    }
                    if (std::isnan(estimate[0]) || std::isnan(estimate[1]) || std::isnan(estimate[2])) {
                        Log(LogLevel::Warn, "Invalid sample!");
                        estimate = Spectrum(0.f);
                    }
                    childrenEstimates += estimate;
                    Li += estimate;
                    if (!child_cluster->isLeaf) {
                        CutCluster new_cut = { 
                            child_cluster, 
                            getClusterUpperBound(scene, sampler, *child_cluster, query.ray, channel, nb_BBIntersection), 
                            std::move(estimate) 
                        };
                        refiningCut.emplace(std::move(new_cut));
                    }
                }
            }

            // We should never or rarely go to this condition
            if (current_element.bound < 100) {
                // Make it up to 100
                // VRLUpperBoundDistance += (current_element.bound / hmax(childrenEstimates)); // Get clamped...
                // VRLUpperBoundDistance.incrementBase();
            }

            // To be able to check the bound
            if ((current_element.bound - hmax(childrenEstimates)) < -math::Epsilon<Float>) {
                /*VRLBoundProblem += 1;*/
#if VERBOSE_UPPERBOUND_PROBLEM
                Float percentage = current_element.bound / childrenEstimates.max();
                Log(LogLevel::Warn, "The deduced error bound is not an upper bound: percentage: %f | bound = %f | children est = %f", percentage, current_element.bound, childrenEstimates.max());
#endif
            }
            /*VRLBoundProblem.incrementBase();*/
        }

        /*VRLLightcutNbEval += query.nb_evaluation;
        VRLLightcutNbEval.incrementBase();*/

#if DEBUG_VRL_LC
        if (std::isnan(Li[0]) || std::isnan(Li[1]) || std::isnan(Li[2])) {
            std::ostringstream stome;
            stome << "NaN Li: " << Li;
            Log(LogLevel::Info, stome.str().c_str());
        }

#endif

        return Li;
    }

    std::string getObj(const int level) const {
        int nb_box = 0;
        return getObjRec(m_root, level, nb_box);
    };

private:
    std::string getObjRec(Node *n, int level, int &nb_box) const {
        if (level == 0) {
            std::stringstream ss;
            ss << "o AABB" << nb_box << "\n" << n->aabb.getObj(nb_box * 8);
            nb_box += 1;
            return ss.str();
        } else {
            std::string t;
            if (n->children[0]) {
                t += getObjRec(n->children[0], level - 1, nb_box);
            }
            if (n->children[1]) {
                t += getObjRec(n->children[1], level - 1, nb_box);
            }
            return t;
        }
    };

    // Method to get the upper bound for a given cluster
    Float getClusterUpperBound(const Scene *scene, Sampler *sampler, const Node &cluster, const Ray3f &r, UInt32 channel, size_t &BBIntersection) const {
        const Medium *medium = cluster.represent.getMedium();

        // Find closest distance between camera ray and the cluster
        // TODO: Need to reimplement this case when we have a small number of VRL (for performance reasons)
        Float min_length       = 0.0;
        Point3f min_aabb_point = Point3f(0.0);
        if (cluster.nodes.size() > m_thresholdBetterDist) {
            min_length = cluster.aabb.getMinDistanceSqr(r);
            min_length = safe_sqrt(min_length);

            // increment number of bounding volume intersection counter
            // nb_BBIntersection++;

        } else {
#if COMPUTE_DISTANCE_PRECISE
            /*VRLBetterDistance += 1;*/
            min_length = std::numeric_limits<Float>::max();
            for (Node *n : cluster.nodes) {
                min_length = min(min_length, sqrDistanceVRL(r, n->represent));
            }            
            min_length = safe_sqrt(min_length);
#else
            DCPQuery query;
            min_length = query.operator()(r, cluster.aabb);
            min_length = safe_sqrt(min_length);
#endif
        };

        // VRLBetterDistance.incrementBase();

        // Do not need to continue
        if (min_length < 0.0001) {
            return std::numeric_limits<Float>::max();
        }
        Mask active = true;

        // TODO: This part of the code should be adapted/checked in case of heterogenous participating media rendering
        Vector3f dir = r.d;
        Spectrum transmittance(1.0);
        Spectrum material = Spectrum(1.f);
        if (medium->is_homogeneous()) {
            // TODO: For example, the direction does not matter when homogenous media is used.
            Ray3f rayOtoPonCluster(r.o, r.d, 0);
            rayOtoPonCluster.mint = 0;
            rayOtoPonCluster.maxt = min_length;

            transmittance = medium->evalMediumTransmittance(rayOtoPonCluster, sampler, active);

            MediumInteraction3f ray_mi;
            ray_mi.p                         = r.o;
            auto [sigma_s, sigma_n, sigma_t] = medium->get_scattering_coefficients(ray_mi, active);
            material *= sigma_s;
            material *= sigma_s;

        } else {
            Log(LogLevel::Error, "Heterogeneous Lightcut not implemented");
        }

        // TODO: This is a very loose upper bound for the moment
        //  it is certainly possible to found a more thigh bound for the phase function
        //  using special arthimetics
        const PhaseFunction *pf = medium->phase_function();
        if (has_flag(pf->flags(active), PhaseFunctionFlags::Isotropic)) {
            material *= 1.0 / (16 * M_PI * M_PI);
        } else {
            material *= 2.0 / (16 * M_PI * M_PI);
        }

        // --- Bound for Kulla when a VRL point is taken uniformly
        // Where we assume that the camera ray is infinite! Which is usually not the case
        Float invRayPDF = cluster.maxVRLLenght;
        Float invVRLPDF = 2.0 * atan(r.maxt * 0.5 / min_length) / min_length;
        Float invPdf    = invRayPDF * invVRLPDF;

        // Sanity check
        Spectrum tmpLi = invPdf * cluster.represent.flux * material * transmittance;
        if (tmpLi[0] != tmpLi[0] || tmpLi[1] != tmpLi[1] || tmpLi[2] != tmpLi[2] || tmpLi[0] < 0.f || tmpLi[1] < 0.f || tmpLi[2] < 0.f || std::isnan(tmpLi[0]) || std::isnan(tmpLi[1]) ||
            std::isnan(tmpLi[2])) {
            Log(LogLevel::Info, "ERROR!");
        }

        return hmax(tmpLi);
    }

    // Method to build the light tree
    Node *buildLightTree(std::vector<Node *> &nodes, Sampler *sampler, bool verbose = false, bool multithread = false) const {
        struct QueueElement {
            // This node information
            size_t id_node1;
            Node *node1;
            size_t id_node2;
            Node *node2;
            Float distance;
            bool n1_merge;
            bool operator>(const QueueElement &rhs) const { return distance > rhs.distance; }
            bool operator<(const QueueElement &rhs) const { return distance < rhs.distance; }
        };
        using Queue = std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>>;
        Queue pairs;

        const bool display = true;

        // Naive function for computing the min distance
        struct MinResult {
            size_t id;
            Float dist;
            bool n1_merge;
        };
        auto min_pair = [&](Node *n, size_t i) -> MinResult {
            size_t o          = 0;
            auto min_distance = Node::DistanceResult{ std::numeric_limits<Float>::max(), false };

            for (auto j = 0; j < nodes.size(); j++) {
                if (i == j)
                    continue;
                if (nodes[j] == nullptr)
                    continue;
                auto dist = n->dist(nodes[j]);
                if (min_distance.distance > dist.distance) {
                    o            = j;
                    min_distance = dist;
                }
            }
            return MinResult{ o, min_distance.distance, min_distance.n1_aabb };
        };

        // Initialize the pair by naive approach
        if (display) {
            Log(LogLevel::Info, "Initializing the pair ....");
        }

        if (multithread) {
            std::mutex mutexPair;
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
            for (size_t i = 0; i < nodes.size(); i++) {
                auto b = min_pair(nodes[i], i);
                {
                    std::lock_guard<std::mutex> guard(mutexPair);
                    pairs.emplace(QueueElement{ i, nodes[i], b.id, nodes[b.id], b.dist, b.n1_merge });
                }
            }
        } else {
            for (size_t i = 0; i < nodes.size(); i++) {
                auto b = min_pair(nodes[i], i);
                pairs.emplace(QueueElement{ i, nodes[i], b.id, nodes[b.id], b.dist, b.n1_merge });
            }
        }

        if (display) {
            Log(LogLevel::Info, "Construction bottom up (naive) [Queue size: %i]", pairs.size());
        }
        size_t percentage_10 = pairs.size() * 0.1;
        size_t nbNodes       = nodes.size();
        while (nbNodes != 1) {
            QueueElement pair = pairs.top();
            pairs.pop();
            if (pair.node1->parent == nullptr && pair.node2->parent == nullptr) {
                // We can merge this nodes (init parents as well)
                Node *merge_node = new Node(pair.node1, pair.node2, sampler->next_1d(), pair.n1_merge);
                // Empty the vector and replace with the merge_node
                nodes[pair.id_node1] = merge_node;
                nodes[pair.id_node2] = nullptr;
                // Create a new pair
                auto b = min_pair(merge_node, pair.id_node1);
                pairs.emplace(QueueElement{ pair.id_node1, merge_node, b.id, nodes[b.id], b.dist, b.n1_merge });
                // Reduce the number of nodes that we need to treat
                nbNodes -= 1;
                if (display) {
                    if (nbNodes % percentage_10 == 0) {
                        Log(LogLevel::Info, "Number of nodes that need to be merged: %i", nbNodes);
                    }
                }
            } else {
                if (pair.node1->parent == nullptr) {
                    auto b = min_pair(pair.node1, pair.id_node1);
                    pairs.emplace(QueueElement{ pair.id_node1, pair.node1, b.id, nodes[b.id], b.dist, b.n1_merge });
                }
                if (pair.node2->parent == nullptr) {
                    auto b = min_pair(pair.node2, pair.id_node2);
                    pairs.emplace(QueueElement{ pair.id_node2, pair.node2, b.id, nodes[b.id], b.dist, b.n1_merge });
                }
            }
        }

        Node *root = nullptr;
        for (auto i = 0; i < nodes.size(); i++) {
            if (nodes[i] != nullptr) {
                // Found the root
                if (display) {
                    Log(LogLevel::Info, "Found root node");
                }
                root = nodes[i];
            }
        }
        return root;
    }

    Node *buildTreeKDAlternate(std::vector<Node *> &clusterList, Sampler *sampler) const {
        auto compare_min_x = [](const Node *n1, const Node *n2) -> bool {
            return std::min(n1->represent.origin[0], n2->represent.getEndPoint()[0]) < std::min(n2->represent.origin[0], n2->represent.getEndPoint()[0]);
        };
        auto compare_max_x = [](const Node *n1, const Node *n2) -> bool {
            return std::max(n1->represent.origin[0], n2->represent.getEndPoint()[0]) < std::max(n2->represent.origin[0], n2->represent.getEndPoint()[0]);
        };
        auto compare_min_y = [](const Node *n1, const Node *n2) -> bool {
            return std::min(n1->represent.origin[1], n2->represent.getEndPoint()[1]) < std::min(n2->represent.origin[1], n2->represent.getEndPoint()[1]);
        };
        auto compare_max_y = [](const Node *n1, const Node *n2) -> bool {
            return std::max(n1->represent.origin[1], n2->represent.getEndPoint()[1]) < std::max(n2->represent.origin[1], n2->represent.getEndPoint()[1]);
        };
        auto compare_min_z = [](const Node *n1, const Node *n2) -> bool {
            return std::min(n1->represent.origin[2], n2->represent.getEndPoint()[2]) < std::min(n2->represent.origin[2], n2->represent.getEndPoint()[2]);
        };
        auto compare_max_z = [](const Node *n1, const Node *n2) -> bool {
            return std::max(n1->represent.origin[2], n2->represent.getEndPoint()[2]) < std::max(n2->represent.origin[2], n2->represent.getEndPoint()[2]);
        };

        //// build the light tree
        std::vector<std::vector<Node *>> new_queue;
        std::vector<std::vector<Node *>> queue;
        new_queue.push_back(clusterList);
        Log(LogLevel::Info, "Splitting the nodes using 6D KDTree heuristic", clusterList.size());

        std::vector<std::vector<Node *>> finalParition;
        for (int depth = 0; !new_queue.empty(); depth++) {
            queue = new_queue;
            new_queue.clear();
            while (!queue.empty()) {
                auto current = queue.back();
                queue.pop_back();

                // Alternate inside the node based on the parent node...
                int axis = depth % 6;
                if (axis == 0) {
                    sort(current.begin(), current.end(), compare_max_x);
                } else if (axis == 1) {
                    sort(current.begin(), current.end(), compare_max_y);
                } else if (axis == 2) {
                    sort(current.begin(), current.end(), compare_max_z);
                } else if (axis == 3) {
                    sort(current.begin(), current.end(), compare_min_x);
                } else if (axis == 4) {
                    sort(current.begin(), current.end(), compare_min_y);
                } else if (axis == 5) {
                    sort(current.begin(), current.end(), compare_min_z);
                }

                // create two new deque's from either side
                int median = current.size() / 2;
                std::vector<Node *> lc(current.begin(), current.begin() + median);
                std::vector<Node *> rc(current.begin() + median, current.end());
                if (lc.size() > 1024) {
                    new_queue.emplace_back(std::move(lc));
                } else {
                    finalParition.emplace_back(std::move(lc));
                }
                if (rc.size() > 1024) {
                    new_queue.emplace_back(std::move(rc));
                } else {
                    finalParition.emplace_back(std::move(rc));
                }
            }
        }
        Log(LogLevel::Info, "Number of mini-tree: %i", finalParition.size());

        // Construct the nodes using better approach
        std::vector<Node *> mergedNodes;
        std::mutex mutexMiniTree;
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < finalParition.size(); i++) {
            Sampler *new_sampler = nullptr;
            {
                std::lock_guard<std::mutex> guard(mutexMiniTree);
                new_sampler = sampler->clone();
            }
            Node *new_node = buildLightTree(finalParition[i], new_sampler);
            {
                std::lock_guard<std::mutex> guard(mutexMiniTree);
                mergedNodes.push_back(new_node);
            }
        }
        return buildLightTree(mergedNodes, sampler);
    }
    Node *buildTreeStable(const Scene *scene, std::vector<Node *> &clusterList, Sampler *sampler) const {
        // TODO: The vector of nodes is bigger. Might be due to some pushs

        // construct kdtree
        size_t numNodes           = 2 * clusterList.size() - 1;
        size_t numCurrentClusters = clusterList.size();
        clusterList.resize(numNodes); // Allocate more

        Log(LogLevel::Info, "Build the bootstrap tree...");
        KdTree<Node> kdtree(scene->getBSphere().radius);
        kdtree.constructTree(clusterList, numCurrentClusters);

        Node *na = clusterList[0];
        Node *nb = kdtree.queryNN(na);

        Log(LogLevel::Info, "Construct the light tree...");
        if (!nb)
            throw std::runtime_error("local agglomerative clustering error");
        size_t nbMerged = 0;
        while (numCurrentClusters < numNodes) {
            Node *nc = kdtree.queryNN(nb);
            if (!nc)
                throw std::runtime_error("local agglomerative clustering error");
            if (na == nc) {
                na->valid   = false;
                nb->valid   = false;
                Node *naOld = na;
                na          = new Node(na, nb, sampler->next1D(), true);
                kdtree.invalidateNodesAndUpdate(naOld, nb, na);
                clusterList[numCurrentClusters++] = na;
                nb                                = NULL;
                nb                                = kdtree.queryNN(na);
                nbMerged += 1;
                if (nbMerged % (numNodes / 20) == 0) {
                    Float p = ((numCurrentClusters / (Float) numNodes) - 0.5) * 2.0;
                    Log(LogLevel::Info, "[%f] Building...", p * 100);
                }
            } else {
                na = nb;
                nb = nc;
            }
        }
        Log(LogLevel::Info, "Light tree built!");

        return clusterList[numNodes - 1];
    }

    inline Float sqrDistanceVRL(const Ray3f &r, const VRL &beam) const {
        Vector3f u = beam.direction * beam.length;
        Vector3f v = r.d * r.maxt;
        Vector3f w(beam.origin - r.o);

        Float a = dot(u, u);
        Float b = dot(u, v);
        Float c = dot(v, v);
        Float d = dot(u, w);
        Float e = dot(v, w);

        Float D = a * c - b * b;

        Float sc, sN, sD = D;
        Float tc, tN, tD = D;

        if (D < 0.0001) {
            sN = 0.0;
            sD = 1.0;
            tN = e;
            tD = c;
        } else {
            sN = (b * e - c * d);
            tN = (a * e - b * d);
            if (sN < 0.0) { // sc < 0 => the s=0 edge is visible
                sN = 0.0;
                tN = e;
                tD = c;
            } else if (sN > sD) { // sc > 1  => the s=1 edge is visible
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < 0.0) { // tc < 0 => the t=0 edge is visible
            tN = 0.0;
            // recompute sc for this edge
            if (-d < 0.0)
                sN = 0.0;
            else if (-d > a)
                sN = sD;
            else {
                sN = -d;
                sD = a;
            }
        } else if (tN > tD) { // tc > 1  => the t=1 edge is visible
            tN = tD;
            // recompute sc for this edge
            if ((-d + b) < 0.0)
                sN = 0;
            else if ((-d + b) > a)
                sN = sD;
            else {
                sN = (-d + b);
                sD = a;
            }
        }

        // finally do the division to get sc and tc
        sc = (std::abs(sN) < 0.0001 ? 0.0 : sN / sD);
        tc = (std::abs(tN) < 0.0001 ? 0.0 : tN / tD);

        // get the difference of the two closest points
        Vector3f dP = w + (sc * u) - (tc * v); // =  S1(sc) - S2(tc)
        return squared_norm(dP);
    }

protected:
    Node *m_root              = nullptr;
    Float m_errorRatio        = 0.1;
    int m_thresholdBetterDist = 8;
};

NAMESPACE_END(mitsuba)