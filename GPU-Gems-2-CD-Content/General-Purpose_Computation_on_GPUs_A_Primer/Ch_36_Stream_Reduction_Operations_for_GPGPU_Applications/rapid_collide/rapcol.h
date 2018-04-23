/*
    Copyright (C) 1998,2000 by Jorrit Tyberghein
    Written by Alex Pfaffe.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
#ifndef __CS_RAPCOL_H__
#define __CS_RAPCOL_H__

#include "cs_compat.h"
#include "matrix3.h"
#include "collider.h"
#include <vector>

class csReversibleTransform;

class csCdModel;
class csCdBBox;
struct csCdTriangle;
struct csCollisionPair;
typedef struct bbox_t{
  float4 Rotationx;
  float3 Rotationy;
  // float3 mRotationz  // since Rotationx and Rotationy are orthogonal
  /// cross(Rotationx,Rotationy);
  float3 Translation;
  float4 Radius;// if it's a leaf Radius.w is 1 else Radius.w = 0

  // if leaf, the Children.xy is an index to the Triangle
  // if node, the Children.xy is an index to left child
  // assert right.xy is always left + {1,0} this may require gaps in the tree
  float2 Children;  
}BBox;

typedef struct Tri_t{
  float3 A;
  float3 B;
  float3 C;
}Tri;
#define tri_vertex_t float3
struct csTraverser {
  csCdBBox *b1;
  csCdBBox *b2;
  csMatrix3 R;
  csVector3 T;
  csTraverser(csCdBBox *b1, csCdBBox *b2,
                const csMatrix3& R, const csVector3& T){
    this->b1 = b1;this->b2=b2;
    this->R=R;this->T=T;
  }
};
extern std::vector<std::vector<csTraverser> > guide;

#define bsp_polygon Tri
class PathPolygonMesh;

/// Low level collision detection using the RAPID algorithm.
class csRapidCollider : public iCollider
{
private:
  friend class csCdBBox;

  /// The internal collision object.
  csCdModel* m_pCollisionModel;

  /// Get top level bounding box.
  const csCdBBox* GetBbox () const;

  /**
   * The smallest dimension in object space. This is the smallest size
   * of one side of the object space bounding box (either x, y, or z).
   * This is used for CollidePath().
   */
  float smallest_box_dim;

  /// Delete and free memory of this objects oriented bounding box.
  void DestroyBbox ();

  /// Recursively test collisions of bounding boxes.
  static int CollideRecursive (csCdBBox *b1, csCdBBox *b2,
  	const csMatrix3& R, const csVector3& T);

  /**
    * Global variables
    * Matrix, and Vector used for collision testing.
    */
  static csMatrix3 mR;
  static csVector3 mT;

  /**
   * Statistics, to allow early bailout.
   * If the number of triangles tested is too high the BBox structure
   * probably isn't very good.
   */
  static int trianglesTested;		// TEMPORARY.
  /// The number of boxes tested.
  static int boxesTested;		// TEMPORARY.
  /**
   * If bbox is less than this size, dont bother testing further,
   * just return with the results so far.
   */
  static float minBBoxDiam;
  /// Number of levels to test.
  static int testLevel;
  /// Test only up to the 1st hit.
  static bool firstHit;

  void GeometryInitialize (const std::vector<bsp_polygon> &mesh);
  friend int main(int argc, char ** argv);
public:
  static int numHits;
  static int numTriChecks;
  void createBrookGeometryRecurse(const csCdBBox *curr, BBox & curw, std::vector <BBox> &bbox, std::vector<Tri> & tri);
  void createBrookGeometry(std::vector <BBox> &bbox, std::vector<Tri> & tri);
  /// Create a collider based on geometry.
  csRapidCollider (const std::vector<bsp_polygon> &mesh);

  /// Destroy the RAPID collider object
  virtual ~csRapidCollider();

  /**
   * Check if this collider collides with pOtherCollider.
   * Returns true if collision detected and adds the pair to the collisions
   * hists vector.
   * This collider and pOtherCollider must be of comparable subclasses, if
   * not false is returned.
   */
  bool Collide (csRapidCollider &pOtherCollider,
                        const csReversibleTransform *pThisTransform = NULL,
                        const csReversibleTransform *pOtherTransform = NULL);

  /// Test collision with an array of colliders.
  bool CollideArray (
  	const csReversibleTransform *pThisTransform,
  	int num_colliders,
	iCollider** colliders,
	csReversibleTransform **transforms);

  int CollidePath (
  	const csReversibleTransform* thisTransform,
	csVector3& newpos,
	int num_colliders,
	iCollider** colliders,
	csReversibleTransform** transforms);

  /// Query the array with collisions (and their count).
  static csCollisionPair *GetCollisions ();

  static void CollideReset ();
  static void SetFirstHit (bool fh) { firstHit = fh; }
  static bool GetFirstHit () { return firstHit; }
  static int Report (csRapidCollider **id1, csRapidCollider **id2);
  const csVector3 &GetRadius() const;

  SCF_DECLARE_IBASE;
};


#endif // __CS_RAPCOL_H__

