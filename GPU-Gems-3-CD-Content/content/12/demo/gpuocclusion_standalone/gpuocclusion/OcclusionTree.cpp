/*! \file OcclusionTree.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of OcclusionTree class.
 */

#include "OcclusionTree.h"
#include <limits>
#include <algorithm>
#include <assert.h>

void OcclusionTree
  ::build(const std::vector<Point> &positions,
          const Triangles &triangles)
{
  mVertexPositions = positions;
  mTriangles = triangles;

  // we will sort an array of indices
  std::vector<unsigned int> triIndices(triangles.size());
  for(unsigned i = 0; i < triangles.size(); ++i)
  {
    triIndices[i] = i;
  }

  // initialize
  // We start out with at least this many leaf nodes
  // more will be added as we create interior nodes
  mNodes.resize(mTriangles.size());

  // recurse
  mRootIndex = build(NULL_NODE,
                     triIndices.begin(),
                     triIndices.end(),
                     triangles);

  // for each node, compute the index of the next
  // node in a depth-first traversal
  for(NodeIndex i = 0;
      i != mNodes.size();
      ++i)
  {
    mNodes[i].mNextNode = computeNextIndex(i);
  } // end for i
} // end OcclusionTree::build()

OcclusionTree::NodeIndex OcclusionTree
  ::build(const NodeIndex parent,
          std::vector<unsigned int>::iterator &begin,
          std::vector<unsigned int>::iterator &end,
          const Triangles &triangles)
{
  assert(begin <= end);

  // base case
  if(begin + 1 == end)
  {
    // add a leaf node: these are stored
    // in order at the beginning of the array
    Node node;
    node.mOcclusion = 1.0f;
    node.mParent = parent;
    node.mLeftChild = node.mRightChild = NULL_NODE;
    createApproximatingDisc(triangles[*begin], node.mDisc);

    if(node.mDisc.mArea == 0)
    {
      std::cerr << "OcclusionTree::build(): zero area triangle: " << *begin << std::endl;
    } // end if

    if(node.mDisc.mNormal[0] != node.mDisc.mNormal[0])
    {
      std::cerr << "OcclusionTree::build(): nan normal on triangle: " << *begin << std::endl;
      node.mDisc.mNormal = float3(0,0,0);
    } // end if

    // set the Node
    mNodes[*begin] = node;

    NodeIndex result = static_cast<NodeIndex>(*begin);
    return result;
  }
  else if(begin == end)
  {
    std::cerr << "OcclusionTree::build(): empty base case." << std::endl;
    return NULL_NODE;
  }

  // find the bounds of the points
  Point min, max;
  findBounds(begin, end, triangles, min, max);

  unsigned int axis = findPrincipalAxis(min, max);

  // add a new node
  NodeIndex nodeLocation = static_cast<NodeIndex>(mNodes.size());
  mNodes.push_back(Node());
  mNodes.back().mParent = parent;
  mNodes.back().mOcclusion = 1.0f;

  // sort along this axis
  sort(begin, end, mVertexPositions, triangles, axis);

  unsigned int diff = static_cast<unsigned int>(end - begin);

  // find the element to split on
  std::vector<unsigned int>::iterator split
    = begin + (end - begin) / 2;

  // recurse
  assert(begin <= split);
  assert(split <= end);
  assert(begin <= end);
  NodeIndex leftChild = build(nodeLocation, begin, split, triangles);
  mNodes[nodeLocation].mLeftChild = leftChild;
  NodeIndex rightChild = build(nodeLocation, split, end, triangles);
  mNodes[nodeLocation].mRightChild = rightChild;

  // create an approximating disc
  assert(leftChild != NULL_NODE);
  assert(rightChild != NULL_NODE);
  createApproximatingDisc(mNodes[leftChild],
                          mNodes[rightChild],
                          mNodes[nodeLocation].mDisc);

  return nodeLocation;
} // end OcclusionTree::build()

OcclusionTree::Node
  ::Node(void)
{
  ;
} // end Node::Node()

void OcclusionTree
  ::findBounds(const std::vector<unsigned int>::iterator &begin,
               const std::vector<unsigned int>::iterator &end,
               const Triangles &triangles,
               Point &minCorner, Point &maxCorner)
{
  float inf = std::numeric_limits<float>::infinity();
  Point inf3(inf,inf,inf);
  minCorner = inf3;
  maxCorner = -inf3;
      
  for(std::vector<unsigned int>::iterator t = begin;
      t != end;
      ++t)
  {
    const Triangle &tri = triangles[*t];
    Point centroid = mVertexPositions[tri[0]]
      + mVertexPositions[tri[1]]
      + mVertexPositions[tri[2]];
    centroid /= 3.0f;

    for(unsigned int i =0;
        i < 3;
        ++i)
    {
       const float &x = centroid[i];

       if(x < minCorner[i])
       {
         minCorner[i] = x;
       } // end if

       if(x > maxCorner[i])
       {
         maxCorner[i] = x;
       } // end if
    } // end for j
  } // end for t
} // end OcclusionTree::findBounds()

unsigned int OcclusionTree
  ::findPrincipalAxis(const Point &min,
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
} // end OcclusionTree::findPrincipalAxis()

struct SortTriangles
{
  unsigned int axis;
  inline bool operator()(const unsigned int &lhs,
                         const unsigned int &rhs) const
  {
    float lhsCentroid = 0, rhsCentroid = 0;
    for(int i = 0; i < 3; ++i)
    {
      lhsCentroid += mVertices[mTriangles[lhs][i]][axis];
      rhsCentroid += mVertices[mTriangles[rhs][i]][axis];
    } // end for i

    lhsCentroid /= 3.0f;
    rhsCentroid /= 3.0f;

    return lhsCentroid < rhsCentroid;
  }

  const std::vector<float3> &mVertices;
  const std::vector<OcclusionTree::Triangle> &mTriangles;
};

void OcclusionTree
  ::sort(std::vector<unsigned int>::iterator &begin,
         std::vector<unsigned int>::iterator &end,
         const std::vector<float3> &vertexPositions,
         const Triangles &triangles,
         const unsigned int axis)
{
  // sort points along this axis
  SortTriangles sorter = {axis, vertexPositions, triangles};
  std::sort(begin, end, sorter);
} // end OcclusionTree::sort()

void OcclusionTree
  ::createApproximatingDisc(const Triangle &tri,
                            Disc &disc) const
{
  computeTriangleAreaAndNormal(tri, disc.mArea, disc.mNormal);

  disc.mCentroid = mVertexPositions[tri[0]]
    + mVertexPositions[tri[1]]
    + mVertexPositions[tri[2]];
  disc.mCentroid /= 3.0f;
} // end OcclusionTree::createApproximatingDisc()

void OcclusionTree
  ::computeTriangleAreaAndNormal(const Triangle &tri,
                                 float &area,
                                 Normal &n) const
{
  const float3 &v0 = mVertexPositions[tri[0]];
  const float3 &v1 = mVertexPositions[tri[1]];
  const float3 &v2 = mVertexPositions[tri[2]];

  float3 e1 = v1 - v0;
  float3 e2 = v2 - v0;

  n = e1.cross(e2);
  float length = n.norm();
  n /= length;
  area = 0.5f * length;
} // end OcclusionTree::computeTriangleAreaAndNormal()

void OcclusionTree
  ::createApproximatingDisc(const Disc &d0,
                            const Disc &d1,
                            Disc &disc) const
{
  disc.mArea = d0.mArea + d1.mArea;
  float w0 = d0.mArea / disc.mArea;
  float w1 = d1.mArea / disc.mArea;

  disc.mNormal = w0 * d0.mNormal + w1 * d1.mNormal;
  disc.mNormal = disc.mNormal.normalize();

  disc.mCentroid = w0 * d0.mCentroid + w1 * d1.mCentroid;

  if(disc.mCentroid[0] != disc.mCentroid[0])
  {
    std::cerr << "OcclusionTree::createApproximatingDisc(): nan centroid." << std::endl;
    disc.mCentroid = (d0.mCentroid + d1.mCentroid) / 2.0f;
  } // end if

  if(disc.mNormal[0] != disc.mNormal[0])
  {
    std::cerr << "OcclusionTree::createApproximatingDisc(): nan normal." << std::endl;
    disc.mNormal = disc.mCentroid;
    disc.mNormal = disc.mNormal.normalize();
  } // end if
} // end OcclusionTree::createApproximatingDisc()

void OcclusionTree
  ::createApproximatingDisc(const Node &n0,
                            const Node &n1,
                            Disc &disc) const
{
  createApproximatingDisc(n0.mDisc, n1.mDisc, disc);
} // end OcclusionTree::createApproximatingDisc()

OcclusionTree::NodeIndex OcclusionTree
  ::computeNextIndex(const NodeIndex i) const
{
  NodeIndex result = mRootIndex;

  // case 1
  // there is no next node to visit after the root
  if(i == mRootIndex)
  {
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
      result = computeNextIndex(mNodes[i].mParent);
    }
  }

  return result;
} // end OcclusionTree::computeNextIndex()

OcclusionTree::NodeIndex OcclusionTree
  ::computeRightBrotherIndex(const NodeIndex i) const
{
  NodeIndex result = NULL_NODE;

  const Node &node = mNodes[i];
  if(i == mNodes[node.mParent].mLeftChild)
  {
    result = mNodes[node.mParent].mRightChild;
  }

  return result;
} // OcclusionTree::computeRightBrotherIndex()

static float rsqrt(float v)
{
  return 1.0f / sqrt(v);
}

static float saturate(float v)
{
  return std::max<float>(0.0f, std::min<float>(1.0f, v));
}

#ifndef PI
#define PI 3.14159265f
#endif // PI

static float solidAngle(float3 v, float d2, float3 receiverNormal,
                        float3 emitterNormal, float emitterArea)
{
  //return (1.0f - rsqrt(emitterArea / (PI * d2) + 1.0f)) *
  //  saturate(emitterNormal.dot(v)) *
  //  saturate(3.0f * receiverNormal.dot(v));
 float result = emitterArea
   * saturate(emitterNormal.dot(-v))
   * saturate(receiverNormal.dot(v))
   / (d2 + emitterArea / PI); 
 return result / PI;
}

float OcclusionTree
  ::computeOcclusion(const float3 &p,
                     const float3 &n,
                     const float epsilon) const
{
  float result = 0.0f;
  float3 v;
  float3 bentNormal = n;
  float d2;
  float contribution;
  float eArea;

  unsigned int currentNode = mRootIndex;
  while(currentNode != NULL_NODE)
  {
    const Node &node = mNodes[currentNode];

    Disc disc = node.mDisc;

    v = disc.mCentroid - p;
    d2 = v.dot(v) + 1e-16f;
    eArea = disc.mArea;

    // we have to stop if:
    // there's no left child under this node
    // or we are approximating and we are far enough away from this element
    if(node.mLeftChild == NULL_NODE
       || (d2 >= epsilon*eArea))
    {
      // compute contribution from this element
      v /= sqrt(d2);
      contribution = solidAngle(v, d2, n,
                                disc.mNormal, eArea);

      // modulate by last result
      contribution *= node.mOcclusion;
      bentNormal -= contribution * v;
      result += contribution;

      // step across the hierarchy
      currentNode = node.mNextNode;
    }
    else
    {
      // traverse deeper
      currentNode = node.mLeftChild;
    }
  }

  result = saturate(1.0f - result);
  return result;
}

void visibleQuad(const float3 &p, const float3 &n,
                 const float3 &v0, const float3 &v1, const float3 &v2,
                 float3 &q0, float3 &q1, float3 &q2, float3 &q3,
                 const bool verbose)
{
  static const float epsilon = 1e-6f;
  float c = n.dot(p);

  // Compute the signed distances from the vertices to the plane.
  float sd[3];
  sd[0] = n.dot(v0) - c;
  if(fabs(sd[0]) <= epsilon) sd[0] = 0;
  sd[1] = n.dot(v1) - c;
  if(fabs(sd[1]) <= epsilon) sd[1] = 0;
  sd[2] = n.dot(v2) - c;
  if(fabs(sd[2]) <= epsilon) sd[2] = 0;

  if(sd[0] > 0)
  {
    if(sd[1] > 0)
    {
      if(sd[2] > 0)
      {
        // +++
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // ++-
        q0 = v0;
        q1 = v1;
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
      }
      else
      {
        // ++0
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
    else if(sd[1] < 0)
    {
      if(sd[2] > 0)
      {
        // +-+
        q0 = v0;
        q1 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = v2;
      }
      else if(sd[2] < 0)
      {
        // +--
        q0 = v0;
        q1 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q2 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q3 = q2;
      }
      else
      {
        // +-0
        q0 = v0;
        q1 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q2 = v2;
        q3 = q2;
      }
    }
    else
    {
      if(sd[2] > 0)
      {
        // +0+
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // +0-
        q0 = v0;
        q1 = v1;
        q2 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q3 = q2;
      }
      else
      {
        // +00
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
  }
  else if(sd[0] < 0)
  {
    if(sd[1] > 0)
    {
      if(sd[2] > 0)
      {
        // -++
        q0 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q1 = v1;
        q2 = v2;
        q3 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
      }
      else if(sd[2] < 0)
      {
        // -+-
        q0 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q1 = v1;
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = q2;
      }
      else
      {
        // -+0
        q0 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
    else if(sd[1] < 0)
    {
      if(sd[2] > 0)
      {
        // --+
        q0 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q1 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // ---
        q0 = q1 = q2 = q3 = p;
      }
      else
      {
        // --0
        q0 = q1 = q2 = q3 = p;
      }
    }
    else
    {
      if(sd[2] > 0)
      {
        // -0+
        q0 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // -0-
        q0 = q1 = q2 = q3 = p;
      }
      else
      {
        // -00
        q0 = q1 = q2 = q3 = p;
      }
    }
  }
  else
  {
    if(sd[1] > 0)
    {
      if(sd[2] > 0)
      {
        // 0++
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // 0+-
        q0 = v0;
        q1 = v1;
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = q2;
      }
      else
      {
        // 0+0
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
    else if(sd[1] < 0)
    {
      if(sd[2] > 0)
      {
        // 0-+
        q0 = v0;
        q1 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // 0--
        q0 = q1 = q2 = q3 = p;
      }
      else
      {
        // 0-0
        q0 = q1 = q2 = q3 = p;
      }
    }
    else
    {
      if(sd[2] > 0)
      {
        // 00+
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else if(sd[2] < 0)
      {
        // 00-
        q0 = q1 = q2 = q3 = p;
      }
      else
      {
        // 000
        q0 = q1 = q2 = q3 = p;
      }
    }
  }
} // end visibleQuad()

float clamp(float m, float M, float val)
{
  return std::max(m, std::min(M, val));
} // end clamp()

float OcclusionTree
  ::computeFormFactor(const float3 &p, const float3 &n,
                      const float3 &q0, const float3 &q1,
                      const float3 &q2, const float3 &q3,
                      const bool verbose)
{
  float3 r0 = q0 - p;
  r0 = r0.normalize();

  float3 r1 = q1 - p;
  r1 = r1.normalize();

  float3 r2 = q2 - p;
  r2 = r2.normalize();

  float3 r3 = q3 - p;
  r3 = r3.normalize();

  float3 g0 = r1.cross(r0).normalize();
  float3 g1 = r2.cross(r1).normalize();
  float3 g2 = r3.cross(r2).normalize();
  float3 g3 = r0.cross(r3).normalize();

  float a = acosf(clamp(-1.0f, 1.0f, r0.dot(r1)));
  float dot = clamp(-1.0f, 1.0f, n.dot(g0));
  float contrib = a * dot;
  float result = contrib;

  if(verbose)
  {
    std::cerr << "a: " << a << std::endl;
    std::cerr << "dot: " << dot << std::endl;
    std::cerr << "contrib: " << contrib << std::endl;
    std::cerr << "result: " << result << std::endl;
  } // end if

  a = acosf(clamp(-1.0f, 1.0f, r1.dot(r2)));
  dot = clamp(-1.0f, 1.0f, n.dot(g1));
  contrib = a * dot;
  result += contrib;

  if(verbose)
  {
    std::cerr << "a: " << a << std::endl;
    std::cerr << "dot: " << dot << std::endl;
    std::cerr << "contrib: " << contrib << std::endl;
    std::cerr << "result: " << result << std::endl;
  } // end if

  a = acosf(clamp(-1.0f, 1.0f, r2.dot(r3)));
  dot = clamp(-1.0f, 1.0f, n.dot(g2));
  contrib = a * dot;
  result += contrib;

  if(verbose)
  {
    std::cerr << "a: " << a << std::endl;
    std::cerr << "dot: " << dot << std::endl;
    std::cerr << "contrib: " << contrib << std::endl;
    std::cerr << "result: " << result << std::endl;
  } // end if

  a = acosf(clamp(-1.0f, 1.0f, r3.dot(r0)));
  dot = clamp(-1.0f, 1.0f, n.dot(g3));
  contrib = a * dot;
  result += contrib;

  if(verbose)
  {
    std::cerr << "a: " << a << std::endl;
    std::cerr << "dot: " << dot << std::endl;
    std::cerr << "contrib: " << contrib << std::endl;
    std::cerr << "result: " << result << std::endl;
    std::cerr << std::endl;
  } // end if

  result *= 0.5f;
  result /= PI;

  return std::max(0.0f, result);
} // end OcclusionTree::computeFormFactor()

float OcclusionTree
  ::computeFormFactor(const float3 &p, const float3 &n,
                      const float3 &v0, const float3 &v1, const float3 &v2)
{
  float3 r0 = v0 - p;
  r0 = r0.normalize();

  float3 r1 = v1 - p;
  r1 = r1.normalize();

  float3 r2 = v2 - p;
  r2 = r2.normalize();

  float3 g0 = r1.cross(r0).normalize();
  float3 g1 = r2.cross(r1).normalize();
  float3 g2 = r0.cross(r2).normalize();

  float a = acosf(clamp(-1.0f, 1.0f, r0.dot(r1)));
  float dot = clamp(-1.0f, 1.0f, n.dot(g0));
  float contrib = a * dot;
  float result = contrib;

  a = acosf(clamp(-1.0f, 1.0f, r1.dot(r2)));
  dot = clamp(-1.0f, 1.0f, n.dot(g1));
  contrib = a * dot;
  result += contrib;

  a = acosf(clamp(-1.0f, 1.0f, r2.dot(r0)));
  dot = clamp(-1.0f, 1.0f, n.dot(g2));
  contrib = a * dot;
  result += contrib;

  result *= 0.5f;
  result /= PI;

  return std::max(0.0f, result);
} // end OcclusionTree::computeFormFactor()

float OcclusionTree
  ::computeFormFactor(const float3 &p, const float3 &n,
                      const Triangle &tri, const float eArea, const bool verbose) const
{
  float3 triNorm = (mVertexPositions[tri[1]] - mVertexPositions[tri[0]])
    .cross(mVertexPositions[tri[2]] - mVertexPositions[tri[0]]).normalize();
  
  float3 q0,q1,q2,q3;
  visibleQuad(p,n,
              mVertexPositions[tri[0]], mVertexPositions[tri[1]], mVertexPositions[tri[2]],
              q0, q1, q2, q3, verbose);
  return computeFormFactor(p,n,q0,q1,q2,q3,verbose);
} // end OcclusionTree::computeFormFactor()

float OcclusionTree
  ::computeOcclusionUseTriangles(const float3 &p,
                                 const float3 &n,
                                 const float epsilon) const
{
  float result = 0.0f;
  float3 v;
  float3 bentNormal = n;
  float d2;
  float contribution = 0.0f;
  float eArea;

  unsigned int currentNode = mRootIndex;
  while(currentNode != NULL_NODE)
  {
    const Node &node = mNodes[currentNode];

    Disc disc = node.mDisc;

    v = disc.mCentroid - p;
    d2 = v.dot(v) + 1e-16f;
    eArea = disc.mArea;

    // we have to stop if:
    // there's no left child under this node
    // or we are approximating and we are far enough away from this element
    if(node.mLeftChild == NULL_NODE
       || (d2 >= epsilon*eArea))
    {
      // compute contribution from this element
      v /= sqrt(d2);

      if(node.mLeftChild == NULL_NODE
         && node.mRightChild == NULL_NODE)
      {
        // compute the contribution of the triangle
        contribution = computeFormFactor(p, n, mTriangles[currentNode], eArea, false);
      } // end if
      else
      {
        contribution = solidAngle(v, d2, n,
                                  disc.mNormal, eArea);
      } // end else


      // modulate by last result
      contribution *= node.mOcclusion;


      bentNormal -= contribution * v;
      result += contribution;

      // step across the hierarchy
      currentNode = node.mNextNode;
    }
    else
    {
      // traverse deeper
      currentNode = node.mLeftChild;
    }
  }

  result = saturate(1.0f - result);
  return result;
} // end OcclusionTree::computeOcclusionUseTriangles()

void OcclusionTree
  ::computeOcclusionPasses(const unsigned int numPasses,
                           const float epsilon)
{
  std::vector<float> occlusion(mNodes.size());

  for(unsigned int pass = 0;
      pass < numPasses;
      ++pass)
  {
    for(unsigned int i = 0;
        i != mNodes.size();
        ++i)
    {
      occlusion[i] = computeOcclusion(mNodes[i].mDisc.mCentroid,
                                      mNodes[i].mDisc.mNormal,
                                      epsilon);
    } // end for i

    // now assign occlusion
    for(unsigned int i = 0;
        i != occlusion.size();
        ++i)
    {
      if(pass == numPasses - 1)
      {
        // force convergence
        float m = std::min(mNodes[i].mOcclusion, occlusion[i]);
        float M = std::max(mNodes[i].mOcclusion, occlusion[i]);

        // since this method tends to overestimate occlusion, bias towards the smaller value
        mNodes[i].mOcclusion = 0.70f * m + 0.30f * M;
      } // end if
      else
      {
        mNodes[i].mOcclusion = occlusion[i];
      } // end else
    }

    std::cerr << "OcclusionTree::computeOcclusionPasses(): Finished pass " << pass << std::endl;
  } // end for pass
}
