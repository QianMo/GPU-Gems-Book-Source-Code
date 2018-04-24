/*! \file DiscTree.h
 *  \author Jared Hoberock, Yuntao Jia
 *  \brief Defines the interface to a
 *         crappy hierarchy for the discs.
 */

#ifndef DISC_TREE_H
#define DISC_TREE_H

#include <gpcpu/Vector.h>
#include <vector>

class DiscTree
{
  public:
    typedef float3 Point;
    typedef float3 Normal;

    struct Disc
    {
      Point mPosition;
      Normal mNormal;
      float mArea;
      float mOcclusion;	  
	  int	mVertex;
    };

    typedef std::vector<Disc> Discs;

    typedef unsigned int NodeIndex;
    static const NodeIndex NULL_NODE = UINT_MAX;
    struct Node
    {
      Disc mDisc;
      NodeIndex mParent;
      NodeIndex mLeftChild;
      NodeIndex mRightChild;
      NodeIndex mNextNode;
	  float		mNextNodeContinue;
    };

    // non-recursive entry to build.
    void build(const Discs &discs);

    void sort(std::vector<unsigned int>::iterator &begin,
              std::vector<unsigned int>::iterator &end,
              const Discs &discs,
              const unsigned int axis);

    void findBounds(const std::vector<unsigned int>::iterator &begin,
                    const std::vector<unsigned int>::iterator &end,
                    const Discs &discs,
                    Point &minCorner, Point &maxCorner);

    unsigned int findPrincipalAxis(const Point &min,
                                   const Point &max) const;

    /*! This method computes the index of the next node in a
     *  depth first traversal of this tree, from node i.
     *  \param i The Node of interest.
     *  \return The index of the next Node from i, if it exists;
     *          UINT_MAX, otherwise.
     */
    NodeIndex computeNextIndex(const NodeIndex i);

    /*! This method computes the index of a Node's brother to the right,
     *  if it exists.
     *  \param i The index of the Node of interest.
     *  \return The index of Node i's brother to the right, if it exists;
     *          UINT_MAX, otherwise.
     */
    NodeIndex computeRightBrotherIndex(const NodeIndex i) const;

    // tree nodes are stored here
    // Note: leaf nodes are stored in a contiguous block at the
    // beginning of this array.  The order they are stored in
    // corresponds to the order of the original Discs array
    // passed to build()
    // In otherwords, after build(discs), it is guaranteed that:
    //
    // discs[i] == mNodes[i].mDisc for i < discs.size()
    std::vector<Node> mNodes;

    // the NodeIndex of the root of the tree is stored here
    NodeIndex mRootIndex;

    // recursive version of build().
    NodeIndex build(const NodeIndex parent,
                    std::vector<unsigned int>::iterator &begin,
                    std::vector<unsigned int>::iterator &end,
                    const Discs &discs);
};

#endif // DISC_TREE_H

