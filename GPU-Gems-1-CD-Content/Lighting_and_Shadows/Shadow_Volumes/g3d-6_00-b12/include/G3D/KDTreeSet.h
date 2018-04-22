/**
  @file KDTreeSet.h
  
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2004-01-11
  @edited  2004-01-11

  Copyright 2000-2004, Morgan McGuire.
  All rights reserved.
  */

#ifndef G3D_KDTREESET_H
#define G3D_KDTREESET_H

#include "G3D/Array.h"
#include "G3D/Table.h"
#include "G3D/Vector3.h"
#include "G3D/AABox.h"
#include "G3D/Sphere.h"
#include "G3D/Box.h"
#include "G3D/Triangle.h"
#include "G3D/Ray.h"
#include <algorithm>

inline void getBounds(const G3D::Vector3& v, G3D::AABox& out) {
    out = G3D::AABox(v);
}


inline void getBounds(const G3D::AABox& a, G3D::AABox& out) {
    out = a;
}


inline void getBounds(const G3D::Sphere& s, G3D::AABox& out) {
    s.getBounds(out);
}


inline void getBounds(const G3D::Box& b, G3D::AABox& out) {
    b.getBounds(out);
}


inline void getBounds(const G3D::Triangle& t, G3D::AABox& out) {
    t.getBounds(out);
}

namespace G3D {

/**
 A G3D::Set that supports spatial queries.  Internally, objects
 are arranged into a k-d tree according to their axis-aligned
 bounds.  This increases the cost of insertion to O(log n) but
 allows fast overlap queries.

 <B>Moving Set Members</B>
 It is important that objects do not move without updating the
 KDTreeSet.  If the axis-aligned bounds of an object are about
 to change, KDTreeSet::remove it before they change and 
 KDTreeSet::insert it again afterward.  For objects
 where the hashCode and == operator are invariant of position,
 you can use the KDTreeSet::update method as a shortcut to
 insert/remove an object in one step after it has moved.
 
 <B>Template Parameters</B>
 <DT>The template parameter <I>T</I> must be one for which
 the following functions are overloaded:

  <BLOCKQUOTE>
  <P><CODE>void ::getBounds(const T&, G3D::AABox&);</CODE>
  <DT><CODE>bool operator==(const T&, const T&);</CODE>
  <DT><CODE>unsigned int ::hashCode(const T&);</CODE>
  <DT><CODE>T::T();</CODE> <I>(public constructor of no arguments)</I>
  </BLOCKQUOTE>

  When using large objects, consider making the template parameter
  a <I>pointer</I> to the object type because the T-values are
  copied many times during tree balancing.

 <B>Dimensions</B>
 Although designed as a 3D-data structure, you can use the KDTreeSet
 for data distributed along 2 or 1 axes by simply returning bounds
 that are always zero along one or more dimensions.

 <B>BETA: Public State</B>
  Most of the public state will become private in a future release;
  it is exposed in this release only to aid debugging.  If you look
  at the source code it should be obvious where the division lies.
 */
template<class T> class KDTreeSet {
//private:
public:

    /** Wrapper for a value that includes a cache of its bounds. */
    class Handle {
    public:
        T                   value;
        AABox               bounds;

        Handle() {}

        inline Handle(const T& v) : value(v) {
            getBounds(v, bounds);
        }
    };


    /**
     A sort predicate that returns true if the midpoint of the
     first argument is less than the midpoint of the second
     along the specified axis.

     Used by makeNode.
     */
    class CenterLT {
    public:
        Vector3::Axis           sortAxis;

        CenterLT(Vector3::Axis a) : sortAxis(a) {}

        inline bool operator()(const Handle& a, const Handle& b) {
            const AABox& A = a.bounds;
            const AABox& B = b.bounds;

            // Only compare distance along the sort axis.  It is faster
            // to sum the low and high than average them.
            return
                (A.low()[sortAxis] + A.high()[sortAxis]) <
                (B.low()[sortAxis] + B.high()[sortAxis]);
        }
    };


    class Node {
    public:

        Vector3::Axis       splitAxis;

        /** Location along the specified axis */
        double              splitLocation;

        Node*               child[2];

        /** Array of values at this node */
        Array<Handle>       valueArray;

        /** Creates node with NULL children */
        Node() {
            splitAxis     = Vector3::X_AXIS;
            splitLocation = 0;
            for (int i = 0; i < 2; ++i) {
                child[i] = NULL;
            }
        }

        /**
         Doesn't clone children.
         */
        Node(const Node& other) : valueArray(other.valueArray) {
            splitAxis       = other.splitAxis;
            splitLocation   = other.splitLocation;
            for (int i = 0; i < 2; ++i) {
                child[i] = NULL;
            }
        }

        /** Copies the specified subarray of pt into point, NULLs the children */
        Node(const Array<Handle>& pt, int beginIndex, int endIndex) {
            splitAxis     = Vector3::X_AXIS;
            splitLocation = 0;
            for (int i = 0; i < 2; ++i) {
                child[i] = NULL;
            }

            int n = endIndex - beginIndex + 1;

            valueArray.resize(n);
            for (int i = n - 1; i >= 0; --i) {
                valueArray[i] = pt[i + beginIndex];
            }
        }


        /** Deletes the children (but not the values) */
        ~Node() {
            for (int i = 0; i < 2; ++i) {
                delete child[0];
            }
        }


        /** Returns true if this node is a leaf (no children) */
        inline bool isLeaf() const {
            return (child[0] == NULL) && (child[1] == NULL);
        }


        /**
         Recursively appends all handles and children's handles
         to the array.
         */
        void getHandles(Array<Handle>& handleArray) const {
            handleArray.append(valueArray);
            for (int i = 0; i < 2; ++i) {
                if (child[i] != NULL) {
                    child[i]->getHandles(handleArray);
                }
            }
        }


        /** Returns the deepest node that completely contains bounds. */
        Node* findDeepestContainingNode(const AABox& bounds) {

            // See which side of the splitting plane the bounds are on
            if (bounds.high()[splitAxis] < splitLocation) {
                // Bounds are on the low side.  Recurse into the child
                // if it exists.
                if (child[0] != NULL) {
                    return child[0]->findDeepestContainingNode(bounds);
                }
            } else if (bounds.low()[splitAxis] > splitLocation) {
                // Bounds are on the high side, recurse into the child
                // if it exists.
                if (child[1] != NULL) {
                    return child[1]->findDeepestContainingNode(bounds);
                }
            }

            // There was no containing child, so this node is the
            // deepest containing node.
            return this;
        }

        /** Appends all members that intersect the box */
        void getIntersectingMembers(const AABox& box, Array<T>& members) const {
            // Test all values at this node
            for (int v = 0; v < valueArray.size(); ++v) {
                if (valueArray[v].bounds.intersects(box)) {
                    members.append(valueArray[v].value);
                }
            }

            // If the left child overlaps the box, recurse into it
            if (box.low()[splitAxis] < splitLocation) {
                child[0]->getIntersectingMembers(box, members);
            }

            // If the right child overlaps the box, recurse into it
            if (box.hi()[splitAxis] > splitLocation) {
                child[1]->getIntersectingMembers(box, members);
            }
        }
    };

    /** Returns the X, Y, and Z extents of the point sub array. */
    static Vector3 computeExtent(const Array<Handle>& point, int beginIndex, int endIndex) {
        Vector3 lo = Vector3::INF3;
        Vector3 hi = -lo;

        for (int p = beginIndex; p <= endIndex; ++p) {
            lo = lo.min(point[p].bounds.low());
            hi = hi.max(point[p].bounds.high());
        }

        return hi - lo;
    }

    /** Number of points to put in each leaf node when
        constructing a balanced tree. */
    enum {VALUES_PER_NODE = 3};

    /**
     Recursively subdivides the subarray.
     Begin and end indices are inclusive.
     */
    Node* makeNode(Array<Handle>& point, int beginIndex, int endIndex) {
        Node* node = NULL;

        if (endIndex - beginIndex + 1 <= VALUES_PER_NODE) {
            // Make a new leaf node
            node = new Node(point, beginIndex, endIndex);

            // Set the pointers in the memberTable
            for (int i = beginIndex; i <= endIndex; ++i) {
                memberTable.set(point[i].value, node);
            }

        } else {
            // Make a new internal node
            node = new Node();

            Vector3 extent = computeExtent(point, beginIndex, endIndex);

            Vector3::Axis splitAxis = extent.primaryAxis();

            // Compute the median along the axis

            // Sort only the subarray 
            std::sort(
                point.getCArray() + beginIndex,
                point.getCArray() + endIndex + 1,
                CenterLT(splitAxis));
            int midIndex = (beginIndex + endIndex) / 2;

            // Choose the split location between the two middle elements
            const Vector3 median = 
                (point[midIndex].bounds.high() +
                 point[min(midIndex + 1, point.size())].bounds.low()) * 0.5;

            node->splitAxis     = splitAxis;
            node->splitLocation = median[splitAxis];
            node->child[0]      = makeNode(point, beginIndex, midIndex);
            node->child[1]      = makeNode(point, midIndex + 1, endIndex);
        }

        return node;
    }

    /**
     Recursively clone the passed in node tree, setting
     pointers for members in the memberTable as appropriate.
     called by the assignment operator.
     */
    Node* cloneTree(Node* src) {
        Node* dst = new Node(*src);

        // Make back pointers
        for (int i = 0; i < dst->valueArray.size(); ++i) {
            memberTable.set(dst->valueArray[i].value, dst);
        }

        // Clone children
        for (int i = 0; i < 2; ++i) {
            if (src->child[i] != NULL) {
                dst->child[i] = cloneTree(src->child[i]);
            }
        }

        return dst;
    }

    /** Maps members to the node containing them */
    Table<T, Node*>         memberTable;

    Node*                   root;

public:

    /** To construct a balanced tree, insert the elements and then call
      KDTreeSet::balance(). */
    KDTreeSet() : root(NULL) {}


    KDTreeSet(const KDTreeSet& src) {
        *this = src;
    }


    KDTreeSet& operator=(const KDTreeSet& src) {
        // Clone tree takes care of filling out the memberTable.
        root = cloneTree(src.root);
    }


    ~KDTreeSet() {
        clear();
    }

    /**
     Throws out all elements of the set.
     */
    void clear() {
        memberTable.clear();
        delete root;
        root = NULL;
    }

    /**
     Inserts an object into the set if it is not
     already present.  O(log n) time.  Does not
     cause the tree to be balanced.
     */
    void insert(const T& value) {
        if (contains(value)) {
            // Already in the set
            return;
        }

        Handle h(value);

        if (root == NULL) {
            // This is the first node; create a root node
            root = new Node();
        }

        Node* node = root->findDeepestContainingNode(h.bounds);

        // Insert into the node
        node->valueArray.append(h);
        
        // Insert into the node table
        memberTable.set(value, node);
    }


    /**
     Returns true if this object is in the set, otherwise
     returns false.  O(1) time.
     */
    bool contains(const T& value) {
        return memberTable.containsKey(value);
    }


    /**
     Removes an object from the set in O(1) time.
     It is an error to remove members that are not already
     present.  May unbalance the tree.
    */
    void remove(const T& value) {
        debugAssertM(contains(value),
            "Tried to remove an element from a "
            "KDTreeSet that was not present");

        Array<Handle>& list = memberTable[value]->valueArray;

        // Find the element and remove it
        for (int i = list.length() - 1; i >= 0; --i) {
            if (list[i].value == value) {
                list.fastRemove(i);
                break;
            }
        }
        memberTable.remove(value);
    }


    /**
     If the element is in the set, it is removed.
     The element is then inserted.

     This is useful when the == and hashCode methods
     on <I>T</I> are independent of the bounds.  In
     that case, you may call update(v) to insert an
     element for the first time and call update(v)
     again every time it moves to keep the tree 
     up to date.
     */
    void update(const T& value) {
        if (contains(value)) {
            remove(value);
        }
        insert(value);
    }


    /**
     Rebalances the tree (slow).  Call when objects
     have moved substantially from their original positions
     (which unbalances the tree and causes the spatial
     queries to be slow).
     */
    void balance() {
        Array<Handle> handleArray;
        root->getHandles(handleArray);

        // Delete the old tree
        clear();

        root = makeNode(handleArray, 0, handleArray.size() - 1);
    }


    /**
     Appends all members whose bounds intersect the box.
     */
    // TODO: make this an iterator as well
    void getIntersectingMembers(const AABox& box, Array<T>& members) const {
        if (root == NULL) {
            return;
        }
        root->getIntersectingMembers(box, members);
    }


    /**
     Returns an array of all members of the set.
     */
    void getMembers(Array<T>& members) const {
        memberTable.getKeys(members);
    }


    /**
     C++ STL style iterator variable.  See begin().
     */
    class Iterator {
    private:
        friend class KDTreeSet<T>;

        // Note: this is a Table iterator, we are currently defining
        // Set iterator
        typename Table<T, Node*>::Iterator it;

        Iterator(const typename Table<T, Node*>::Iterator& it) : it(it) {}

    public:
        inline bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

        bool operator==(const Iterator& other) const {
            return it == other.it;
        }

        /**
         Pre increment.
         */
        Iterator& operator++() {
            ++it;
            return *this;
        }

        /**
         Post increment (slower than preincrement).
         */
        Iterator operator++(int) {
            Iterator old = *this;
            ++(*this);
            return old;
        }

        const T& operator*() const {
            return it->key;
        }

        T* operator->() const {
            return &(it->key);
        }

        operator T*() const {
            return &(it->key);
        }
    };


    /**
     C++ STL style iterator method.  Returns the first member.  
     Use preincrement (++entry) to get to the next element (iteration
     order is arbitrary).  
     Do not modify the set while iterating.
     */
    Iterator begin() const {
        return Iterator(memberTable.begin());
    }


    /**
     C++ STL style iterator method.  Returns one after the last iterator
     element.
     */
    const Iterator end() const {
        return Iterator(memberTable.end());
    }
};

}

#endif
