/*! \file DiscTree.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DiscTree class.
 */

#include "DiscTree.h"
#include <limits>
#include <algorithm>
#include <assert.h>

struct AxisSort
{
  unsigned int axis;
  inline bool operator()(const unsigned int &lhs,
                         const unsigned int &rhs) const
  {
    return (*mDiscs)[lhs].mPosition[axis]
      < (*mDiscs)[rhs].mPosition[axis];
  }
  const DiscTree::Discs *mDiscs;
};

void DiscTree::sort(std::vector<unsigned int>::iterator &begin,
                    std::vector<unsigned int>::iterator &end,
                    const Discs &discs,
                    const unsigned int axis)
{
  // sort points along this axis
  AxisSort sorter = {axis, &discs};
  std::sort(begin, end, sorter);
}

void DiscTree::findBounds(const std::vector<unsigned int>::iterator &begin,
                          const std::vector<unsigned int>::iterator &end,
                          const Discs &discs,
                          Point &minCorner, Point &maxCorner)
{
  float inf = std::numeric_limits<float>::infinity();
  minCorner = Point(inf,inf,inf);
  maxCorner = -minCorner;
  for(std::vector<unsigned int>::iterator p = begin;
      p != end;
      ++p)
  {
    for(int i = 0; i < 3; ++i)
    {
      if(discs[*p].mPosition[i] < minCorner[i])
      {
        minCorner[i] = discs[*p].mPosition[i];
      }

      if(discs[*p].mPosition[i] > maxCorner[i])
      {
        maxCorner[i] = discs[*p].mPosition[i];
      }
    }
  }	
}

void DiscTree::build(const Discs &discs)
{
  // we will sort an array of disc indices
  std::vector<unsigned int> discIndices(discs.size());
  for(unsigned i = 0; i < discIndices.size(); ++i)
  {
    discIndices[i] = i;
  }
  
  // initialize
  // We start out with at least this many leaf nodes
  // more will be added as we create interior nodes
  mNodes.resize(discs.size());

  // recurse
  mRootIndex = build(NULL_NODE, discIndices.begin(), discIndices.end(),
                     discs);

  // now, for each node, compute the index of the next
  // node in a depth-first traversal
  for(unsigned int i = 0;
      i != mNodes.size();
      ++i)
  {
    mNodes[i].mNextNode = computeNextIndex(i);
  } // end for n
}

DiscTree::NodeIndex DiscTree::build(const NodeIndex parent,
                                    std::vector<unsigned int>::iterator &begin,
                                    std::vector<unsigned int>::iterator &end,
                                    const Discs &discs)
{
  // base case
  if(begin + 1 == end)
  {
    // add a leaf node: these are stored
    // in order at the beginning of the array
    Node node = {discs[*begin],
                 parent,
                 NULL_NODE,
                 NULL_NODE,
                 NULL_NODE,
				 0.0f};				// initial the continue to 1, suppose the node has a real brother 
    mNodes[*begin] = node;

    NodeIndex result = static_cast<NodeIndex>(*begin);
    return result;
  }
  else if(begin == end)
  {
    std::cerr << "DiscTree::build(): empty base case." << std::endl;
    return NULL_NODE;
  }

  // find the bounds of the points
  Point min, max;
  findBounds(begin, end, discs, min, max);

  unsigned int axis = findPrincipalAxis(min, max);

  // add a new node
  NodeIndex nodeLocation = static_cast<NodeIndex>(mNodes.size());
  mNodes.push_back(Node());
  mNodes.back().mParent = parent;

  // sort along this axis
  sort(begin, end, discs, axis);

  unsigned int diff = static_cast<unsigned int>(end - begin);

  // find the element to split on
  std::vector<unsigned int>::iterator split = begin + (end - begin) / 2;

  // recurse
  assert(begin <= split);
  assert(split <= end);
  NodeIndex leftChild = build(nodeLocation, begin, split, discs);
  mNodes[nodeLocation].mLeftChild = leftChild;
  NodeIndex rightChild = build(nodeLocation, split, end, discs);
  mNodes[nodeLocation].mRightChild = rightChild;

  // set my position to the average of my children
  mNodes[nodeLocation].mDisc.mPosition = float3(0,0,0);
  unsigned int numChildren = 0;
  float weights[2] = {0,0};
  if(leftChild != NULL_NODE)
  {
    weights[0] = mNodes[leftChild].mDisc.mArea;	
  }

  if(rightChild != NULL_NODE)
  {
    weights[1] = mNodes[rightChild].mDisc.mArea;	
  }
  
  // sum weights
  float sum = weights[0] + weights[1];

  float invSum = 1.0f / sum;

  // blend position
  mNodes[nodeLocation].mDisc.mPosition =
    weights[0] * mNodes[leftChild].mDisc.mPosition +
    weights[1] * mNodes[rightChild].mDisc.mPosition;
  mNodes[nodeLocation].mDisc.mPosition *= invSum;

  // blend normal
  mNodes[nodeLocation].mDisc.mNormal =
    weights[0] * mNodes[leftChild].mDisc.mNormal +
    weights[1] * mNodes[rightChild].mDisc.mNormal;
  mNodes[nodeLocation].mDisc.mNormal *= invSum;
  mNodes[nodeLocation].mDisc.mNormal = mNodes[nodeLocation].mDisc.mNormal.normalize();

  // sum area
  mNodes[nodeLocation].mDisc.mArea = sum;
  
  // vertex
  mNodes[nodeLocation].mDisc.mVertex = -1;

  // occlusion
  mNodes[nodeLocation].mDisc.mOcclusion = 1.f;

  return nodeLocation;
}

unsigned int DiscTree::findPrincipalAxis(const Point &min,
                                         const Point &max) const
{
  // find the principal axis of the points
  unsigned int axis = 0;
  float maxLength = -1.0f;
  float temp;
  for(int i = 0; i < 3; ++i)
  {
    temp = max[i] - min[i];
    if(temp > maxLength)
    {
      maxLength = temp;
      axis = i;
    }
  }

  return axis;
}

DiscTree::NodeIndex DiscTree::computeNextIndex(const NodeIndex i)
{
  NodeIndex result = mRootIndex;
  mNodes[i].mNextNodeContinue = 1;

  // case 1
  // there is no next node to visit after the root
  if(i == mRootIndex)
  {
	mNodes[i].mNextNodeContinue = 0;					// set the continue to 0, that means it does not have a real brother 
	result = NULL_NODE;
  }
  else
  {
    // case 2
    // if i am my parent's left child, return my brother
    result = computeRightBrotherIndex(i);
    if(result == NULL_NODE)
    { 
      // case 3
      // return my father's next      
	  mNodes[i].mNextNodeContinue = 0;					// set the continue to 0, that means it does not have a real brother 
	  result = computeNextIndex(mNodes[i].mParent);
    }
  }

  return result;
}

DiscTree::NodeIndex
DiscTree::computeRightBrotherIndex(const NodeIndex i) const
{
  NodeIndex result = NULL_NODE;

  const Node &node = mNodes[i];
  if(i == mNodes[node.mParent].mLeftChild)
  {
    result = mNodes[node.mParent].mRightChild;
  }

  return result;
}

