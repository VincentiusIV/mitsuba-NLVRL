#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/timer.h>


NAMESPACE_BEGIN(mitsuba)

const int k = 3;

struct PointNode {
    float point[k];
    short flag;
    PointNode *left;
    PointNode *right;
};

template<typename TNode>
class PointKDTree {
    static_assert(std::is_base_of<PointNode, TNode>::value, "TNode must derive from PointNode");
public:

    TNode *insertRec(TNode *root, TNode *newNode, unsigned depth) { 
        if (root == NULL) {
            root = newNode;
        } else {
            unsigned cd = depth % k;
            if (newNode->point[cd] < (root->point[cd]))
                root->left = insertRec(root->left, newNode, depth + 1);
            else
                root->right = insertRec(root->right, newNode, depth + 1);
            return root;
        }
    }

    TNode *insert(float point[]) { 
        return insertRec(root, point, 0);
    }

private:
    TNode *root;
};

NAMESPACE_END(mitsuba)