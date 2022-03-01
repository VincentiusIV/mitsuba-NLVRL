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

    TNode *newNode(int point[]) {
        TNode *temp = new TNode();
        for (int i = 0; i < k; i++)
            temp->point[i] = point[i];
        temp->left = temp->right = NULL;
        return temp;
    }

    TNode *insertRec(TNode *root, float point[], unsigned depth) { 
        if (root == NULL)
            return newNode(point);
        unsigned cd = depth % k;
        if (point[cd] < (root->point[cd]))
            root->left = insertRec(root->left, point, depth + 1);
        else
            root->right = insertRec(root->right, point, depth + 1);
        return root;
    }

    TNode *insert(float point[]) { 
        return insertRec(root, point, 0);
    }

private:
    TNode *root;
};

NAMESPACE_END(mitsuba)