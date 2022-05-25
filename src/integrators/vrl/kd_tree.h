#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)
// This code come from illuminationCut's paper.

// Templated based on the node type
template <typename Float, typename Spectrum, class Node> class KdNode{
public:
    MTS_IMPORT_TYPES()
    MTS_IMPORT_OBJECT_TYPES()

    KdNode():depth(0),cluster(NULL),intensity(0.0),bbox(),left(NULL),right(NULL),valid(false),parent(NULL){}
    ~KdNode(){
        if (left)
            delete left;
        if (right)
            delete right;
    }

    int depth;
    Node* cluster;
    Spectrum intensity;
    BoundingBox<Point3f> bbox;

    KdNode<Float, Spectrum, Node>* left;
    KdNode<Float, Spectrum, Node> *right;
    bool valid;
    KdNode<Float, Spectrum, Node> *parent;
};

template <typename Float, typename Spectrum, class Node>
class KdTree{
public:
    typedef KdNode<Float, Spectrum, Node> KdNode;

    KdNode* root;
    float sceneradiusSqr;

    KdTree(float sr):root(NULL),sceneradiusSqr(sr){

    }

    ~KdTree(){
        if (root)
            delete root;
    }

    void constructTree(std::vector<Node *> &leafs, size_t numLeafs) {
        std::vector<Node *> points;
        for (size_t i = 0; i < numLeafs; ++i)
            points.push_back(leafs[i]);

        this->root = constructSubtree(points, 0);
        this->setParents(this->root);
        return;
    }

    KdNode *constructSubtree(std::vector<Node *> &points, int level) {
        if (points.size() <= 1) {
            if (points.size() == 0)
                return NULL;
            KdNode *node = new KdNode<Node>;
            node->valid  = true;
            node->depth  = level;

            node->cluster         = points[0];
            node->intensity       = node->cluster->represent.flux;
            node->bbox            = node->cluster->aabb;
            node->cluster->kdnode = node;
            return node;
        }
        KdNode *node = new KdNode<Node>;
        node->valid  = true;
        node->depth  = level;

        size_t medianPlace = points.size() / 2;
        std::sort(points.begin(), points.end(), levComp<Node>(level));

        // FIXME there is no need to create a new vector, should just pass the iterators
        std::vector<Node *> left(points.begin(), points.begin() + medianPlace);
        std::vector<Node *> right(points.begin() + medianPlace + 1, points.end());

        node->cluster         = points[medianPlace];
        node->intensity       = node->cluster->represent.flux;
        node->bbox            = node->cluster->aabb;
        node->cluster->kdnode = node;
        node->left            = constructSubtree(left, level + 1);
        node->right           = constructSubtree(right, level + 1);

        return node;
    }

    Node *queryNN(Node *node) {
        // get the corresponding kdnode
        KdNode *queryNode = node->kdnode;
        // we have an upper bound on the diagonal alpha<sqrt(distmin/I)
        // initial guess root, and check other if equality
        KdNode *currentbest = queryNode;
        int explored        = 0;

        // modified values for search start
        float currentdist = std::numeric_limits<float>::infinity();
        float upperbound  = std::numeric_limits<float>::infinity();
        // check children
        this->queryChild(queryNode, true, upperbound, currentdist, currentbest, queryNode, explored);
        this->queryChild(queryNode, false, upperbound, currentdist, currentbest, queryNode, explored);
        // check upwards
        this->queryParent(queryNode->parent, queryNode, upperbound, currentdist, currentbest, queryNode, explored);
        // return the best one;
        if (currentbest == queryNode) {
            return NULL;
        }
        return currentbest->cluster;
    }
    void invalidateNodesAndUpdate(Node *a, Node *b, Node *newNode) {
        // invalidate the lower one, update intensity, assign new cluster
        if (a->kdnode->depth > b->kdnode->depth) {
            // invalidate kdnode
            a->kdnode->valid = false;
            b->kdnode->intensity += a->kdnode->intensity;
            this->deleteSubtree(a->kdnode);

            newNode->kdnode    = b->kdnode;
            b->kdnode->cluster = newNode;
        } else {
            b->kdnode->valid = false;
            a->kdnode->intensity += b->kdnode->intensity;
            this->deleteSubtree(b->kdnode);

            newNode->kdnode    = a->kdnode;
            a->kdnode->cluster = newNode;
        }
        return;
    }
    bool deleteInvalidLeaf(KdNode *leaf) {
        if (!leaf)
            return true;

        bool deletedLeaf = false;
        // if invalid leaf
        if (!leaf->valid && !leaf->right && !leaf->left) {
            bool left = (leaf->parent->left == leaf);
            if (left)
                leaf->parent->left = NULL;
            else
                leaf->parent->right = NULL;

            deletedLeaf = true;
            delete leaf;
        }
        return deletedLeaf;
    }

    void deleteSubtree(KdNode *node) {
        if (node == this->root)
            return;

        KdNode *parent     = node->parent;
        KdNode *otherChild = (parent->left == node) ? parent->right : parent->left;
        if (this->deleteInvalidLeaf(node))
            if (this->deleteInvalidLeaf(otherChild))
                this->deleteSubtree(parent);
    }

    void setParents(KdNode *node) {
        if (node->left) {
            node->left->parent = node;
            this->setParents(node->left);
        }
        if (node->right) {
            node->right->parent = node;
            this->setParents(node->right);
        }
    }

    void queryParent(KdNode *parent, KdNode *child, float &upperbound, float &currentdist, KdNode *&currentbest, KdNode *queryNode, int &explored) {
        // this the root (hopefully)
        if (!parent)
            return;
        // recurse into other child
        bool otherchild = (child != parent->left);
        this->queryChild(parent, otherchild, upperbound, currentdist, currentbest, queryNode, explored);
        // recurse upwards
        this->queryParent(parent->parent, parent, upperbound, currentdist, currentbest, queryNode, explored);
        // end
        return;
    }
    void queryChild(KdNode *parent, bool left, float &upperbound, float &currentdist, KdNode *&currentbest, KdNode *queryNode, int &explored) {

        if (!parent)
            return;

        explored++;
        // update mindist with parent
        if (parent->valid && parent != queryNode) {
            // FIXME: Cosine : this->sceneradiusSqr
            float tmpDist = queryNode->cluster->dist(parent->cluster).distance;
            if (currentdist > tmpDist) {
                currentdist = tmpDist;
                currentbest = parent;
                upperbound  = sqrt(currentdist / hmax(queryNode->intensity));
            }
        }

        float levelDistParent = levComp<Node>::dist(parent, queryNode, parent->depth); // negative means it is on the left
        if (left) {
            if (levelDistParent <= 0) {
                this->queryChild(parent->left, true, upperbound, currentdist, currentbest, queryNode, explored);
                this->queryChild(parent->left, false, upperbound, currentdist, currentbest, queryNode, explored);
            } else {
                if (levelDistParent > upperbound) {
                    return;
                } else {
                    // recurse
                    this->queryChild(parent->left, true, upperbound, currentdist, currentbest, queryNode, explored);
                    this->queryChild(parent->left, false, upperbound, currentdist, currentbest, queryNode, explored);
                }
            }
        }
        if (!left) {
            if (levelDistParent >= 0) {
                // recurse
                this->queryChild(parent->right, true, upperbound, currentdist, currentbest, queryNode, explored);
                this->queryChild(parent->right, false, upperbound, currentdist, currentbest, queryNode, explored);
            } else {
                if (-levelDistParent > upperbound) {
                    return;
                } else {
                    // recurse
                    this->queryChild(parent->right, true, upperbound, currentdist, currentbest, queryNode, explored);
                    this->queryChild(parent->right, false, upperbound, currentdist, currentbest, queryNode, explored);
                }
            }
        }
        return;
    }
};


///////////////////////////////////////////// IMPLEMENTATION
template <typename Float, typename Spectrum, class Node> 
struct levComp {
    typedef KdNode<Float, Spectrum, Node> KdNode;

    int level;
    static float dist(KdNode* split, KdNode* current, int level){
        switch ( level % 3 ){
            case 0:
                return (current->bbox.min.x - split->bbox.min.x);
            case 1:
                return (current->bbox.min.y - split->bbox.min.y);
            case 2:
                return (current->bbox.min.z - split->bbox.min.z);
        }

        throw std::runtime_error("compare error in kdtree construction");
        return false;
    }
    levComp(int level):level(level){}
    bool operator() (Node* i,Node* j) {
        switch ( level % 3 ){
            case 0:
                return (i->aabb.min.x < j->aabb.min.x);
            case 1:
                return (i->aabb.min.y < j->aabb.min.y);
            case 2:
                return (i->aabb.min.z < j->aabb.min.z);
        }

        throw std::runtime_error("compare error in kdtree construction");
        return false;
    }
};


NAMESPACE_END(mitsuba)
