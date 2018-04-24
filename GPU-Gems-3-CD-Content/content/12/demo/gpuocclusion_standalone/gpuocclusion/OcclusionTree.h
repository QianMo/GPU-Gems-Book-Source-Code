/*! \file OcclusionTree.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a
 *         hierarchy for triangles.
 */

#ifndef OCCLUSION_TREE_H
#define OCCLUSION_TREE_H

#include <gpcpu/Vector.h>
#include <vector>

class OcclusionTree
{
  public:
    typedef float3 Point;
    typedef float3 Normal;
    typedef Vector<unsigned int, 3> uint3;

    struct Disc
    {
      Point mCentroid;
      Normal mNormal;
      float mArea;
    }; // end Disc

    /*! A Triangle is a triplet of vertex indices.
     */
    typedef uint3 Triangle;

    typedef std::vector<Triangle> Triangles;

    typedef unsigned int NodeIndex;
    static const NodeIndex NULL_NODE = UINT_MAX;

    struct Node
    {
      Disc mDisc;
      float mOcclusion;
      NodeIndex mParent;
      NodeIndex mLeftChild;
      NodeIndex mRightChild;
      NodeIndex mNextNode;

      // null constructor does nothing
      Node(void);
    }; // end Node

    float computeOcclusion(const float3 &p,
                           const float3 &n,
                           const float epsilon = 4.0f) const;

    float computeOcclusionUseTriangles(const float3 &p,
                                       const float3 &n,
                                       const float epsilon = 4.0f) const;

    void computeOcclusionPasses(const unsigned int numPasses,
                                const float epsilon = 4.0f);

    float computeFormFactor(const float3 &p, const float3 &n,
                            const Triangle &tri, const float triArea, const bool verbose) const;

    static float computeFormFactor(const float3 &p, const float3 &n,
                                   const float3 &q0, const float3 &q1,
                                   const float3 &q2, const float3 &q3, const bool verbose);

    static float computeFormFactor(const float3 &p, const float3 &n,
                                   const float3 &v0, const float3 &v1, const float3 &v2);

    void build(const std::vector<Point> &positions,
               const Triangles &triangles);

    // recursive version of build().
    NodeIndex build(const NodeIndex parent,
                    std::vector<unsigned int>::iterator &begin,
                    std::vector<unsigned int>::iterator &end,
                    const Triangles &triangles);

    /*! This method computes the index of the next node in a
     *  depth first traversal of this tree, from node i.
     *  \param i The Node of interest.
     *  \return The index of the next Node from i, if it exists;
     *          UINT_MAX, otherwise.
     */
    NodeIndex computeNextIndex(const NodeIndex i) const;

    /*! This method computes the index of a Node's brother to the right,
     *  if it exists.
     *  \param i The index of the Node of interest.
     *  \return The index of Node i's brother to the right, if it exists;
     *          UINT_MAX, otherwise.
     */
    NodeIndex computeRightBrotherIndex(const NodeIndex i) const;

    void computeTriangleAreaAndNormal(const Triangle &tri,
                                      float &area,
                                      Normal &n) const;

    void findBounds(const std::vector<unsigned int>::iterator &begin,
                    const std::vector<unsigned int>::iterator &end,
                    const Triangles &triangles,
                    Point &minCorner, Point &maxCorner);

    unsigned int findPrincipalAxis(const Point &min,
                                   const Point &max) const;

    void createApproximatingDisc(const Triangle &tri,
                                 Disc &disc) const;

    void createApproximatingDisc(const Disc &d0,
                                 const Disc &d1,
                                 Disc &disc) const;

    void createApproximatingDisc(const Node &n0,
                                 const Node &n1,
                                 Disc &disc) const;

    static void sort(std::vector<unsigned int>::iterator &begin,
                     std::vector<unsigned int>::iterator &end,
                     const std::vector<float3> &vertexPositions,
                     const Triangles &triangles,
                     const unsigned int axis);

    std::vector<Point> mVertexPositions;
    Triangles mTriangles;
    std::vector<Node> mNodes;

    NodeIndex mRootIndex;
}; // end class OcclusionTree

#endif // OCCLUSION_TREE_H

