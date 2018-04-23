/*
    Copyright (C) 1998,2000 by Written by Alex Pfaffe

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

// The following classes are utility classes for the RAPID collision detection
// algorithm. The code is based on the UNC implementation of the RAPID
// algorithm:

/*************************************************************************\

  Copyright 1995 The University of North Carolina at Chapel Hill.
  All Rights Reserved.

  Permission to use, copy, modify and distribute this software and its
  documentation for educational, research and non-profit purposes, without
  fee, and without a written agreement is hereby granted, provided that the
  above copyright notice and the following three paragraphs appear in all
  copies.

  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE
  LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
  CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
  USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
  OF NORTH CAROLINA HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGES.

  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
  PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
  NORTH CAROLINA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

  The authors may be contacted via:

  US Mail:             S. Gottschalk
                       Department of Computer Science
                       Sitterson Hall, CB #3175
                       University of N. Carolina
                       Chapel Hill, NC 27599-3175

  Phone:               (919)962-1749

  EMail:              {gottscha}@cs.unc.edu


\**************************************************************************/
#include <stdio.h>
#include <queue>
#include <algorithm>
#include "cs_compat.h"
#include "qsqrt.h"
#include "qint.h"
#include "rapcol.h"
#include "prapid.h"
#include "garray.h"

#include "csgeom/transfrm.h"
#define CD_MAX_COLLISION    1000

// This array contains the colliding pairs
static CS_DECLARE_GROWING_ARRAY_REF (CD_contact,csCollisionPair);

static int hits = 0;
// Array of hits.
static csRapidCollider* hitv[CD_MAX_COLLISION][2];
static int currHit;

///
csMatrix3 csRapidCollider::mR;
csVector3 csRapidCollider::mT (0, 0, 0);
///
int   csRapidCollider::trianglesTested = 0;
int   csRapidCollider::boxesTested     = 0;
int   csRapidCollider::testLevel       = 0;
bool  csRapidCollider::firstHit        = true;
int   csRapidCollider::numHits         = 0;
int   csRapidCollider::numTriChecks    = 0;
float csRapidCollider::minBBoxDiam     = 0.0;
///

SCF_IMPLEMENT_IBASE (csRapidCollider)
  SCF_IMPLEMENTS_INTERFACE (iCollider)
SCF_IMPLEMENT_IBASE_END


Moment *Moment::stack = NULL;

csRapidCollider::csRapidCollider (const std::vector <bsp_polygon> &polygons)
{

  SCF_CONSTRUCT_IBASE (NULL);
  GeometryInitialize (polygons);
}

static float min3 (float a, float b, float c)
{ return (a < b ? (a < c ? a : (c < b ? c : b)) : (b < c ? b : c)); }
static float max3(float a, float b, float c)
{ return (a > b ? (a > c ? a : (c > b ? c : b)) : (b > c ? b : c)); }
float3 tofloat3(const csVector3 &a) {
  return float3(a.x,a.y,a.z);
}
void csRapidCollider::createBrookGeometryRecurse(const csCdBBox *curr,
                                                 BBox &curw,
                                                 std::vector<BBox>&bbox,
                                                 std::vector<Tri>&tri){
  float childx,childy;
  curw.Children.x = (float)(bbox.size()%2048);
  curw.Children.y = (float)(bbox.size()/2048);
  childx=curw.Children.x;
  childy=curw.Children.y;
  float3 tmp = tofloat3(curr->m_Rotation.Row1());
  curw.Rotationx.x = tmp.x;
  curw.Rotationx.y = tmp.y;
  curw.Rotationx.z = tmp.z;
  curw.Rotationx.w =((curr->m_Rotation.Row1() % curr->m_Rotation.Row2())*curr->m_Rotation.Row3()>.8)?1.0f:-1.0f;
  curw.Rotationy = tofloat3(curr->m_Rotation.Row2());
  //  curw.Rotationz = tofloat3(curr->m_Rotation.Row3());
  curw.Translation = tofloat3(curr->m_Translation);


  curw.Radius.x = curr->m_Radius.x;
  curw.Radius.y = curr->m_Radius.y;
  curw.Radius.z = curr->m_Radius.z;

  if (curr->IsLeaf()) {
    curw.Radius.w = 1.0f;
    curw.Children.x = (float)(tri.size()%2048);
    curw.Children.y = (float)(tri.size()/2048);
    tri.push_back(Tri());
    tri.back().A = tofloat3(curr->m_pTriangle->p1);
    tri.back().B = tofloat3(curr->m_pTriangle->p2);
    tri.back().C = tofloat3(curr->m_pTriangle->p3);
                            
    //
  }else {
    assert (curr->m_pTriangle==0);
    curw.Radius.w = 0;
    bbox.push_back(BBox());
    unsigned int secondchild = bbox.size();
    bbox.push_back(BBox());
    curr->m_pChild0->ind[0]=childx;
    curr->m_pChild0->ind[1]=childy;
    createBrookGeometryRecurse(curr->m_pChild0,
                               *(bbox.end()-2),
                               bbox,
                               tri);
    curr->m_pChild1->ind[0]=childx+1.0f;
    curr->m_pChild1->ind[1]=childy;
    createBrookGeometryRecurse(curr->m_pChild1,
                               bbox[secondchild],
                               bbox,
                               tri);
  }
}
void csRapidCollider::createBrookGeometry(std::vector<BBox>&bbox,
                                          std::vector<Tri>&tri){
  bbox.push_back(BBox());
  bbox.push_back(BBox());//keep things two-box aligned
  bbox.back().Rotationx=float4(1,0,0,1);
  bbox.back().Rotationy=float3(0,1,0);
  bbox.back().Translation=float3(1024.*1024.*16384,-1024.*1024.*16384,1024.*1024.*16384);
  bbox.back().Radius=float4(1.0f/65536.0f,1.0f/65536.0f,1.0f/65536.0f,1);
  bbox.back().Children=float2(0,0);
  tri.push_back(Tri());
  tri.back().A=float3(1024.*1024.*16384,-1024.*1024.*16384,1024.*1024.*16384);;
  tri.back().B=float3(1025.*1024.*16384,-1024.*1024.*16384,1024.*1024.*16384);
  tri.back().C=float3(1025.*1024.*16384,-1025.*1024.*16384,1024.*1024.*16384);
  createBrookGeometryRecurse(m_pCollisionModel->GetTopLevelBox(),
                             bbox.front(),
                             bbox,
                             tri);
  while (tri.size()%2048)
    tri.push_back(Tri());
  while (bbox.size()%2048)
    bbox.push_back(BBox());
}
void csRapidCollider::GeometryInitialize (const std::vector <bsp_polygon> &polygons)
{
  m_pCollisionModel = NULL;

  CD_contact.IncRef ();

  int i;
  int tri_count = 0;
  // first, count the number of triangles polyset contains
  //  csVector3* vertices = mesh->GetVertices ();
  /*
  for (i = 0; i < (int) polygons.size() ; i++)
  {
    tri_count += polygons[i].v.size() - 2;
  }
  */
  tri_count+=polygons.size();

  csBox3 object_bbox;
  object_bbox.StartBoundingBox ();
  if (tri_count)
  {
    m_pCollisionModel = new csCdModel (tri_count);
    if (!m_pCollisionModel)
      return;
    
    for (i = 0; i < (int) polygons.size() ; i++)
    {
      const bsp_polygon * p = (&polygons[i]);
      //      int* vidx = p.vertices;
      // Collision detection only works with triangles.
      object_bbox.AddBoundingVertex (p->A);
      object_bbox.AddBoundingVertex (p->B);
      //for (v = 2; v < (int)p->v.size(); v++)
      {
        m_pCollisionModel->AddTriangle (p->A,
                                        p->B, 
                                        p->C);
        object_bbox.AddBoundingVertex (p->C);
      }
    }
    m_pCollisionModel->BuildHierarchy ();
  }

  float dx = object_bbox.MaxX () - object_bbox.MinX ();
  float dy = object_bbox.MaxY () - object_bbox.MinY ();
  float dz = object_bbox.MaxZ () - object_bbox.MinZ ();
  smallest_box_dim = min3 (dx, dy, dz);
  if (smallest_box_dim < .1f) smallest_box_dim = .1f;
}

csRapidCollider::~csRapidCollider ()
{

  if (m_pCollisionModel)
  {
    delete m_pCollisionModel;
    m_pCollisionModel = NULL;
  }

  CD_contact.DecRef ();
}

bool csRapidCollider::Collide (csRapidCollider &otherCollider, 
                               const csReversibleTransform *pTransform1, 
                               const csReversibleTransform *pTransform2)
{
  csRapidCollider *pRAPIDCollider2 = (csRapidCollider *)&otherCollider;
// VC does not understand that preprocessor command
//#warning is comparing against self collide tree b4d??
#if 0
  if (pRAPIDCollider2 == this) return false;
#endif
  // JTY: also skip objects with m_pCollisionModel NULL. This fixes
  // a bug with bezier curves and collision detection.
  if (!m_pCollisionModel || 
      !pRAPIDCollider2->m_pCollisionModel) return 0;

  // I don't know, why this is commented out. I need to elaborate this
  // further. thieber 2000-02-19
  //  csRapidCollider::firstHit = true; 

  // call the low level collision detection routine.

  csCdBBox *b1 = m_pCollisionModel->m_pBoxes;
  csCdBBox *b2 = pRAPIDCollider2->m_pCollisionModel->m_pBoxes;

  csMatrix3 R1;
  csVector3 T1 (0, 0, 0);
  if (pTransform1 && pTransform2)
  {
    csReversibleTransform trans1 = *pTransform1;
    trans1 *= pTransform2->GetInverse ();
    T1 = trans1.GetO2TTranslation ();
    R1 = trans1.GetO2T ();
  }
  else if (pTransform1)
  {
    T1 = pTransform1->GetO2TTranslation ();
    R1 = pTransform1->GetO2T ();
  }
  else if (pTransform2)
  {
    csReversibleTransform trans1 = pTransform2->GetInverse ();
    T1 = trans1.GetO2TTranslation ();
    R1 = trans1.GetO2T ();
  }

  // [R1,T1] is how the first triangle set is positioned relative to the
  // second one (combination of two transforms) (in world space).
  // [tR1,tT1] and [tR2,tT2] are how the top level boxes are
  // positioned in world space

  csMatrix3 tR1 = R1 * b1->m_Rotation;
  csVector3 tT1 = (R1 * b1->m_Translation) + T1;
  csMatrix3 tR2 = b2->m_Rotation;
  csVector3 tT2 = b2->m_Translation;

  // [R,T] is the placement of b2's top level box within
  // the coordinate system of b1's top level box.

  csMatrix3 R = tR1.GetTranspose () * tR2;
  csVector3 T = tR1.GetTranspose () * (tT2 - tT1);

  // To transform tri's from model1's CS to model2's CS use this:
  //    x2 = mR . x1 + mT

  csRapidCollider::mR = R1;
  csRapidCollider::mT = T1;

  // reset the report fields
  csRapidCollider::numHits = 0;
  csRapidCollider::trianglesTested = 0;
  csRapidCollider::boxesTested = 0;
  // make the call
  if (CollideRecursive (b1, b2, R, T))
  {
    // Error.
  }
  else if (csRapidCollider::numHits != 0)
  {
    hitv [hits][0] = this;
    hitv [hits][1] = pRAPIDCollider2;
    hits++;
    return 1;
  }
  return 0;
}

bool csRapidCollider::CollideArray (
  	const csReversibleTransform* trans,
  	int num_colliders,
	iCollider** colliders,
	csReversibleTransform **transforms)
{
  int i;
  for (i = 0 ; i < num_colliders ; i++)
  {
    bool rc = Collide (*(csRapidCollider*)colliders[i],
    	trans, transforms[i]);
    if (rc) return rc;
  }
  return false;
}

int csRapidCollider::CollidePath (
  	const csReversibleTransform* trans,
	csVector3& newpos,
	int num_colliders,
	iCollider** colliders,
	csReversibleTransform** transforms)
{
  // At initialization time (GeometryInitialize()) we calculated
  // the smallest dimension of the object space bounding box (either
  // x, y, or z space). We are going to use half this size to slowly
  // move the object along its path. By using half the size we can
  // be fairly sure that we will not miss anything.

  csReversibleTransform test = *trans;
  csVector3 start = test.GetOrigin ();
  csVector3 end = newpos;
  csVector3 testpos;
  float step = 1.0f / (smallest_box_dim/2.0f);
  float curdist = 0;
  bool rc = false;
  bool firsthit = true;
  for (;;)
  {
    testpos = start+curdist * (end-start);
    test.SetOrigin (testpos);
    CollideReset ();
    numHits = 0;
    rc = CollideArray (&test, num_colliders, colliders, transforms);
    if (rc) break;
    firsthit = false;

    if (curdist >= 1) break;
    curdist += step;
    if (curdist > 1) curdist = 1;
  }

  if (rc)
  {
    // We had a collision.
    if (firsthit)
    {
      // The collision happened on the start point. In that case
      // we cannot move at all. Return -1.
      return -1;
    }

    // Here we try to find more exactly where the collision occured by
    // doing a binary search.
    end = testpos;
    while (csSquaredDist::PointPoint (start, end) > .05)
    {
      testpos = (start+end) / 2;
      test.SetOrigin (testpos);
      CollideReset ();
      numHits = 0;
      rc = CollideArray (&test, num_colliders, colliders, transforms);
      if (rc)
      {
        // Use left segment.
        end = testpos;
      }
      else
      {
        // Use right segment.
	start = testpos;
      }
    }
    // We know that the object can move to the 'start' position safely
    // because of the way we handle the binary search and the starting
    // condition that firsthit == false.
    newpos = start;

    // But first we set the collision detection array to the position
    // which resulted in collision.
    test.SetOrigin (end);
    CollideReset ();
    numHits = 0;
    CollideArray (&test, num_colliders, colliders, transforms);

    return 0;
  }
  else
  {
    // There was no collision.
    return 1;
  }
}

csCollisionPair *csRapidCollider::GetCollisions ()
{
  return CD_contact.GetArray ();
}

int project6 (csVector3 ax, csVector3 p1, csVector3 p2, csVector3 p3,
  csVector3 q1, csVector3 q2, csVector3 q3)
{
  float P1 = ax * p1;
  float P2 = ax * p2;
  float P3 = ax * p3;
  float Q1 = ax * q1;
  float Q2 = ax * q2;
  float Q3 = ax * q3;

  float mx1 = max3 (P1, P2, P3);
  float mn1 = min3 (P1, P2, P3);
  float mx2 = max3 (Q1, Q2, Q3);
  float mn2 = min3 (Q1, Q2, Q3);

  if (mn1 > mx2) return 0;
  if (mn2 > mx1) return 0;
  return 1;
}

/* Triangle/triangle intersection test routine,
 * by Tomas Moller, 1997.
 * See article "A Fast Triangle-Triangle Intersection Test",
 * Journal of Graphics Tools, 2(2), 1997
 *
 * int tri_tri_intersect (float V0[3],float V1[3],float V2[3],
 *                         float U0[3],float U1[3],float U2[3])
 *
 * parameters: vertices of triangle 1: V0,V1,V2
 *             vertices of triangle 2: U0,U1,U2
 * result    : returns 1 if the triangles intersect, otherwise 0
 *
 */

/*
   if USE_EPSILON_TEST is true
     then we do a check:
       if |dv| < EPSILON
         then dv = 0.0;
         else no check is done (which is less robust)
*/
//#define USE_EPSILON_TEST	1

/* some macros */
#define CROSS(dest, v1, v2)			\
  dest [0] = v1 [1] * v2 [2] - v1 [2] * v2 [1];	\
  dest [1] = v1 [2] * v2 [0] - v1 [0] * v2 [2];	\
  dest [2] = v1 [0] * v2 [1] - v1 [1] * v2 [0];

#define DOT(v1, v2)				\
  (v1 [0] * v2 [0] + v1 [1] * v2 [1] + v1 [2] * v2 [2])

#define SUB(dest, v1, v2)			\
  dest [0] = v1 [0] - v2 [0];			\
  dest [1] = v1 [1] - v2 [1];			\
  dest [2] = v1 [2] - v2 [2];

/* sort so that a <= b */
#define SORT(a, b)				\
  if (a > b)					\
  {						\
    float c;					\
    c = a;					\
    a = b;					\
    b = c;					\
  }

#define ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1)\
  {						\
    isect0 = VV0 + (VV1 - VV0) * D0 / (D0 - D1);\
    isect1 = VV0 + (VV2 - VV0) * D0 / (D0 - D2);\
  }

#define COMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,isect0,isect1) \
  if (D0D1 > 0.0f)					\
    /* here we know that D0D2<=0.0 */			\
    /* that is D0, D1 are on the same side, */		\
    /* D2 on the other or on the plane */		\
    ISECT (VV2, VV0, VV1, D2, D0, D1, isect0, isect1)	\
  else if (D0D2 > 0.0f)					\
    /* here we know that d0d1 <= 0.0 */			\
    ISECT (VV1, VV0, VV2, D1, D0, D2, isect0, isect1)	\
  else if (D1 * D2 > 0.0f || D0 != 0.0f)		\
    /* here we know that d0d1<=0.0 or that D0!=0.0 */	\
    ISECT (VV0, VV1, VV2, D0, D1, D2, isect0, isect1)	\
  else if (D1 != 0.0f)					\
    ISECT (VV1, VV0, VV2, D1, D0, D2, isect0, isect1)	\
  else if(D2!=0.0f)                                     \
    ISECT (VV2, VV0, VV1, D2, D0, D1, isect0, isect1)	\
  else							\
    /* triangles are coplanar */			\
    return coplanar_tri_tri (N1, V0, V1, V2, U0, U1, U2);

/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define EDGE_EDGE_TEST(V0, U0, U1)			\
  Bx = U0 [i0] - U1 [i0];				\
  By = U0 [i1] - U1 [i1];				\
  Cx = V0 [i0] - U0 [i0];				\
  Cy = V0 [i1] - U0 [i1];				\
  f = Ay * Bx - Ax * By;				\
  d = By * Cx - Bx * Cy;				\
  if ((f > 0 && d >= 0 && d <= f)			\
   || (f < 0 && d <= 0 && d >= f))			\
  {							\
    e = Ax * Cy - Ay * Cx;				\
    if (f > 0)						\
    {							\
      if (e >= 0 && e <= f) return 1;			\
    }							\
    else						\
    {							\
      if (e <= 0 && e >= f) return 1;			\
    }							\
  }

#define EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2)	\
  {							\
    float Ax, Ay, Bx, By, Cx, Cy, e, d, f;		\
    Ax = V1 [i0] - V0 [i0];				\
    Ay = V1 [i1] - V0 [i1];				\
    /* test edge U0,U1 against V0,V1 */			\
    EDGE_EDGE_TEST (V0, U0, U1);			\
    /* test edge U1,U2 against V0,V1 */			\
    EDGE_EDGE_TEST (V0, U1, U2);			\
    /* test edge U2,U1 against V0,V1 */			\
    EDGE_EDGE_TEST (V0, U2, U0);			\
  }

#define POINT_IN_TRI(V0, U0, U1, U2)			\
  {							\
    float a, b, c, d0, d1, d2;				\
    /* is T1 completly inside T2? */			\
    /* check if V0 is inside tri(U0,U1,U2) */		\
    a = U1 [i1] - U0 [i1];				\
    b = -(U1 [i0] - U0 [i0]);				\
    c = -a * U0 [i0] - b * U0 [i1];			\
    d0 = a * V0 [i0] + b * V0 [i1] + c;			\
							\
    a = U2 [i1] - U1 [i1];				\
    b = -(U2 [i0] - U1 [i0]);				\
    c = -a * U1 [i0] - b * U1 [i1];			\
    d1 = a * V0 [i0] + b * V0 [i1] + c;			\
							\
    a = U0 [i1] - U2 [i1];				\
    b = -(U0 [i0] - U2 [i0]);				\
    c = -a * U2 [i0] - b * U2 [i1];			\
    d2 = a * V0 [i0] + b * V0 [i1] + c;			\
    if (d0 * d1 > 0.0)					\
    {							\
      if (d0 * d2 > 0.0) return 1;			\
    }							\
  }

int coplanar_tri_tri (float N[3],
  const csVector3 &V0, const csVector3 &V1, const csVector3 &V2,
  const csVector3 &U0, const csVector3 &U1, const csVector3 &U2)
{
  float A[3];
  short i0,i1;
  /* first project onto an axis-aligned plane, that maximizes the area */
  /* of the triangles, compute indices: i0,i1. */
  A [0] = ABS  (N [0]);
  A [1] = ABS  (N [1]);
  A [2] = ABS  (N [2]);
  if (A [0] > A [1])
  {
    if (A [0] > A [2])
    {
      i0 = 1;			/* A[0] is greatest */
      i1 = 2;
    }
    else
    {
      i0 = 0;			/* A[2] is greatest */
      i1 = 1;
    }
  }
  else				/* A[0]<=A[1] */
  {
    if (A [2] > A [1])
    {
      i0 = 0;			/* A[2] is greatest */
      i1 = 1;
    }
    else
    {
      i0 = 0;			/* A[1] is greatest */
      i1 = 2;
    }
  }

  /* test all edges of triangle 1 against the edges of triangle 2 */
  EDGE_AGAINST_TRI_EDGES (V0, V1, U0, U1, U2);
  EDGE_AGAINST_TRI_EDGES (V1, V2, U0, U1, U2);
  EDGE_AGAINST_TRI_EDGES (V2, V0, U0, U1, U2);

  /* finally, test if tri1 is totally contained in tri2 or vice versa */
  POINT_IN_TRI (V0, U0, U1, U2);
  POINT_IN_TRI (U0, V0, V1, V2);

  return 0;
}

int tri_contact(const csVector3 &V0, const csVector3 &V1, const csVector3 &V2,
                const csVector3 &U0, const csVector3 &U1, const csVector3 &U2)
{
  float E1 [3], E2 [3];
  float N1 [3], N2 [3], d1, d2;
  float du0, du1, du2, dv0, dv1, dv2;
  float D [3];
  float isect1 [2], isect2 [2];
  float du0du1, du0du2, dv0dv1, dv0dv2;
  short index;
  float vp0, vp1, vp2;
  float up0, up1, up2;
  float b, c, max;

  /* compute plane equation of triangle(V0,V1,V2) */
  SUB (E1, V1, V0);
  SUB (E2, V2, V0);
  CROSS (N1, E1, E2);
  d1 = -DOT (N1, V0);
  /* plane equation 1: N1.X+d1=0 */

  /* put U0,U1,U2 into plane equation 1 to compute signed distances to
     the plane */
  du0 = DOT (N1, U0) + d1;
  du1 = DOT (N1, U1) + d1;
  du2 = DOT (N1, U2) + d1;

  /* coplanarity robustness check */
#if USE_EPSILON_TEST
  if (ABS  (du0) < EPSILON) du0 = 0.0;
  if (ABS  (du1) < EPSILON) du1 = 0.0;
  if (ABS  (du2) < EPSILON) du2 = 0.0;
#endif
  du0du1 = du0 * du1;
  du0du2 = du0 * du2;

  /* same sign on all of them + not equal 0 ? */
  if (du0du1 > 0.0f && du0du2 > 0.0f)
    return 0;			/* no intersection occurs */

  /* compute plane of triangle (U0,U1,U2) */
  SUB (E1, U1, U0);
  SUB (E2, U2, U0);
  CROSS (N2, E1, E2);
  d2 = -DOT (N2, U0);
  /* plane equation 2: N2.X+d2=0 */

  /* put V0,V1,V2 into plane equation 2 */
  dv0 = DOT (N2, V0) + d2;
  dv1 = DOT (N2, V1) + d2;
  dv2 = DOT (N2, V2) + d2;

#if USE_EPSILON_TEST
  if (ABS  (dv0) < EPSILON) dv0 = 0.0;
  if (ABS  (dv1) < EPSILON) dv1 = 0.0;
  if (ABS  (dv2) < EPSILON) dv2 = 0.0;
#endif

  dv0dv1 = dv0 * dv1;
  dv0dv2 = dv0 * dv2;

  /* same sign on all of them + not equal 0 ? */
  if (dv0dv1 > 0.0f && dv0dv2 > 0.0f)
    return 0;			/* no intersection occurs */

  /* compute direction of intersection line */
  CROSS (D, N2, N1);

  /* compute and index to the largest component of D */
  max = ABS  (D [0]);
  index = 0;
  b = ABS  (D [1]);
  c = ABS  (D [2]);
  if (b > max) max = b, index = 1;
  if (c > max) max = c, index = 2;

  /* this is the simplified projection onto L*/
  vp0 = V0 [index];
  vp1 = V1 [index];
  vp2 = V2 [index];

  up0 = U0 [index];
  up1 = U1 [index];
  up2 = U2 [index];

  /* compute interval for triangle 1 */
  COMPUTE_INTERVALS (
    vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, isect1 [0], isect1 [1]);

  /* compute interval for triangle 2 */
  COMPUTE_INTERVALS (
    up0, up1, up2, du0, du1, du2, du0du1, du0du2, isect2 [0], isect2 [1]);

  SORT (isect1 [0], isect1 [1]);
  SORT (isect2 [0], isect2 [1]);

  if (isect1 [1] < isect2 [0] || isect2 [1] < isect1 [0])
    return 0;
  return 1;
}

/*
// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

bool tri_contact (csVector3 P1, csVector3 P2, csVector3 P3,
                  csVector3 Q1, csVector3 Q2, csVector3 Q3)
{
  //
  // One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
  // Edges are (e1,e2,e3) and (f1,f2,f3).
  // Normals are n1 and m1
  // Outwards are (g1,g2,g3) and (h1,h2,h3).
  //
  // We assume that the triangle vertices are in the same coordinate system.
  //
  // First thing we do is establish a new c.s. so that p1 is at (0,0,0).
  //

  csVector3 p1 = P1 - P2;
  csVector3 p2 = P2 - P1;
  csVector3 p3 = P3 - P1;

  csVector3 q1 = Q1 - P1;
  csVector3 q2 = Q2 - P1;
  csVector3 q3 = Q3 - P1;

  csVector3 e1 = p2 - p1;
  csVector3 e2 = p3 - p2;
  csVector3 e3 = p1 - p3;

  csVector3 f1 = q2 - q1;
  csVector3 f2 = q3 - q2;
  csVector3 f3 = q1 - q3;

  csVector3 n1 = e1 % e2;
  csVector3 m1 = f1 % f2;

  csVector3 g1 = e1 % n1;
  csVector3 g2 = e2 % n1;
  csVector3 g3 = e3 % n1;
  csVector3 h1 = f1 % m1;
  csVector3 h2 = f2 % m1;
  csVector3 h3 = f3 % m1;

  csVector3 ef11 = e1 % f1;
  csVector3 ef12 = e1 % f2;
  csVector3 ef13 = e1 % f3;

  csVector3 ef21 = e2 % f1;
  csVector3 ef22 = e2 % f2;
  csVector3 ef23 = e2 % f3;

  csVector3 ef31 = e3 % f1;
  csVector3 ef32 = e3 % f2;
  csVector3 ef33 = e3 % f3;

  // now begin the series of tests

  if (!project6(n1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(m1, p1, p2, p3, q1, q2, q3)) return false;

  if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;

  if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

  return true;
}
*/

int add_collision (csCdTriangle *tr1, csCdTriangle *tr2)
{
  int limit = CD_contact.Limit ();
  if (csRapidCollider::numHits >= limit)
  {
//  is this really needed? - A.Z.
//  if (!limit)
//    csRapidCollidernumHits = 0;
    CD_contact.SetLimit (limit + 16);
  }

  CD_contact [csRapidCollider::numHits].a1 = tr1->p1;
  CD_contact [csRapidCollider::numHits].b1 = tr1->p2;
  CD_contact [csRapidCollider::numHits].c1 = tr1->p3;
  CD_contact [csRapidCollider::numHits].a2 = tr2->p1;
  CD_contact [csRapidCollider::numHits].b2 = tr2->p2;
  CD_contact [csRapidCollider::numHits].c2 = tr2->p3;
  csRapidCollider::numHits++;

  return false;
}

int obb_disjoint (const csMatrix3& B, const csVector3& T,
	const csVector3& a, const csVector3& b)
{
  register float t, s;
  register int r;
  csMatrix3 Bf;
  const float reps = (float)1e-6;

  // Bf = ABS (B)
  Bf.m11 = ABS (B.m11);  Bf.m11 += reps;
  Bf.m12 = ABS (B.m12);  Bf.m12 += reps;
  Bf.m13 = ABS (B.m13);  Bf.m13 += reps;
  Bf.m21 = ABS (B.m21);  Bf.m21 += reps;
  Bf.m22 = ABS (B.m22);  Bf.m22 += reps;
  Bf.m23 = ABS (B.m23);  Bf.m23 += reps;
  Bf.m31 = ABS (B.m31);  Bf.m31 += reps;
  Bf.m32 = ABS (B.m32);  Bf.m32 += reps;
  Bf.m33 = ABS (B.m33);  Bf.m33 += reps;

  // if any of these tests are one-sided, then the polyhedra are disjoint
  r = 1;

  // A1 x A2 = A0
  t = ABS (T.x);

  r &= (t <= (a.x + b.x * Bf.m11 + b.y * Bf.m12 + b.z * Bf.m13));
  if (!r) return 1;

  // B1 x B2 = B0
  s = T.x * B.m11 + T.y * B.m21 + T.z * B.m31;
  t = ABS (s);

  r &= (t <= (b.x + a.x * Bf.m11 + a.y * Bf.m21 + a.z * Bf.m31));
  if (!r) return 2;

  // A2 x A0 = A1
  t = ABS (T.y);

  r &= (t <= (a.y + b.x * Bf.m21 + b.y * Bf.m22 + b.z * Bf.m23));
  if (!r) return 3;

  // A0 x A1 = A2
  t = ABS (T.z);

  r &= (t <= (a.z + b.x * Bf.m31 + b.y * Bf.m32 + b.z * Bf.m33));
  if (!r) return 4;

  // B2 x B0 = B1
  s = T.x * B.m12 + T.y * B.m22 + T.z * B.m32;
  t = ABS (s);

  r &= (t <= (b.y + a.x * Bf.m12 + a.y * Bf.m22 + a.z * Bf.m32));
  if (!r) return 5;

  // B0 x B1 = B2
  s = T.x * B.m13 + T.y * B.m23 + T.z * B.m33;
  t = ABS (s);

  r &= (t <= (b.z + a.x * Bf.m13 + a.y * Bf.m23 + a.z * Bf.m33));
  if (!r) return 6;

  // A0 x B0
  s = T.z * B.m21 - T.y * B.m31;
  t = ABS (s);

  r &= (t <= (a.y * Bf.m31 + a.z * Bf.m21 + b.y * Bf.m13 + b.z * Bf.m12));
  if (!r) return 7;

  // A0 x B1
  s = T.z * B.m22 - T.y * B.m32;
  t = ABS (s);

  r &= (t <= (a.y * Bf.m32 + a.z * Bf.m22 + b.x * Bf.m13 + b.z * Bf.m11));
  if (!r) return 8;

  // A0 x B2
  s = T.z * B.m23 - T.y * B.m33;
  t = ABS (s);

  r &= (t <= (a.y * Bf.m33 + a.z * Bf.m23 + b.x * Bf.m12 + b.y * Bf.m11));
  if (!r) return 9;

  // A1 x B0
  s = T.x * B.m31 - T.z * B.m11;
  t = ABS (s);

  r &= (t <= (a.x * Bf.m31 + a.z * Bf.m11 + b.y * Bf.m23 + b.z * Bf.m22));
  if (!r) return 10;

  // A1 x B1
  s = T.x * B.m32 - T.z * B.m12;
  t = ABS (s);

  r &= (t <= (a.x * Bf.m32 + a.z * Bf.m12 + b.x * Bf.m23 + b.z * Bf.m21));
  if (!r) return 11;

  // A1 x B2
  s = T.x * B.m33 - T.z * B.m13;
  t = ABS (s);

  r &= (t <= (a.x * Bf.m33 + a.z * Bf.m13 + b.x * Bf.m22 + b.y * Bf.m21));
  if (!r) return 12;

  // A2 x B0
  s = T.y * B.m11 - T.x * B.m21;
  t = ABS (s);

  r &= (t <= (a.x * Bf.m21 + a.y * Bf.m11 + b.y * Bf.m33 + b.z * Bf.m32));
  if (!r) return 13;

  // A2 x B1
  s = T.y * B.m12 - T.x * B.m22;
  t = ABS (s);

  r &= (t <= (a.x * Bf.m22 + a.y * Bf.m12 + b.x * Bf.m33 + b.z * Bf.m31));
  if (!r) return 14;

  // A2 x B2
  s = T.y * B.m13 - T.x * B.m23;
  t = ABS (s);

  r &= (t <= (a.x * Bf.m23 + a.y * Bf.m13 + b.x * Bf.m32 + b.y * Bf.m31));
  if (!r) return 15;

  return 0;  // should equal 0
}
std::vector<std::vector<csTraverser> > guide;

#if 0

int csRapidCollider::CollideRecursive (csCdBBox *b1, csCdBBox *b2,
	const csMatrix3& R, const csVector3& T)
{
  int rc;      // return codes
  if (0&&csRapidCollider::firstHit && (csRapidCollider::numHits > 0))
    return false;

  // test top level
  csRapidCollider::boxesTested++;

  int f1 = obb_disjoint (R, T, b1->GetRadius(), b2->GetRadius());

  if (f1 != 0)
    return false;  // stop processing this test, go to top of loop

  // contact between boxes
  if (b1->IsLeaf() && b2->IsLeaf())
  {
    // it is a leaf pair - compare the polygons therein
    // TrianglesHaveContact uses the model-to-model transforms stored in
    // csRapidCollider::mR, csRapidCollider::mT.

    // this will pass along any OUT_OF_MEMORY return codes which
    // may be generated.
    return csCdBBox::TrianglesHaveContact(b1, b2);
  }

  csMatrix3 cR;
  csVector3 cT (0, 0, 0);

  // Currently, the transform from model 2 to model 1 space is
  // given by [B T], where y = [B T].x = B.x + T.

  if (b2->IsLeaf() || (!b1->IsLeaf() && (b1->GetSize() > b2->GetSize())))
  {
    // here we descend to children of b1.  The transform from
    // a child of b1 to b1 is stored in
    // [b1->N->m_Rotation,b1->N->m_Translation],
    // but we will denote it [B1 T1] for short.

    // Here, we compute [B1 T1]'[B T] = [B1'B B1'(T-T1)]
    // for each child, and store the transform into the collision
    // test queue.

    csMatrix3 rot_transp = b1->m_pChild0->m_Rotation.GetTranspose ();
    cR = rot_transp * R;
    cT = rot_transp * (T - b1->m_pChild0->m_Translation);

    if ((rc = CollideRecursive (b1->m_pChild0, b2, cR, cT)) != false)
      return rc;

    rot_transp = b1->m_pChild1->m_Rotation.GetTranspose ();
    cR = rot_transp * R;
    cT = rot_transp * (T - b1->m_pChild1->m_Translation);

    if ((rc = CollideRecursive (b1->m_pChild1, b2, cR, cT)) != false)
      return rc;
	
  }
  else
  {
    // here we descend to the children of b2.  See comments for
    // other 'if' clause for explanation.
    cR = R * b2->m_pChild0->m_Rotation;
    cT = ( R * b2->m_pChild0->m_Translation) + T;

    if ((rc = CollideRecursive (b1, b2->m_pChild0, cR, cT)) != false)
      return rc;
 
    cR = R * b2->m_pChild1->m_Rotation;
    cT = ( R * b2->m_pChild1->m_Translation) + T;
	
    if ((rc = CollideRecursive (b1, b2->m_pChild1, cR, cT)) != false)
      return rc;
	
  }

  return false;
}
#else
static void Barken() {

}
static void Bark() {

}
int csRapidCollider::CollideRecursive (csCdBBox *b1, csCdBBox *b2,
                                       const csMatrix3& R, const csVector3& T)
{
  std::queue <csTraverser> breadth;
  
  unsigned int pass_count=0;
  unsigned int pass_size=1;
  unsigned int num_passes=0;
  std::vector<csTraverser> temp_guide;
  breadth.push(csTraverser(b1,b2,R,T));
  while (breadth.size()) {
  if (pass_count==pass_size) {
    pass_size=breadth.size();
    pass_count=0;
    static int lastCount=0;
    static int lastCheckCount=0;
    printf ("Pass %d Num Triangles %d num nodes %d\n",num_passes,csRapidCollider::trianglesTested-lastCheckCount,pass_size);
    if (csRapidCollider::numHits-lastCheckCount) {
         printf ("Detected %d hits\n",csRapidCollider::numHits-lastCount);      
    }
    guide.push_back(temp_guide);
    temp_guide.clear();
    lastCount = csRapidCollider::numHits;
    lastCheckCount = csRapidCollider::trianglesTested;
    num_passes++;
  }
  pass_count++;
  csCdBBox *b1; csCdBBox *b2;
  csMatrix3 R;csVector3 T;
  b1 = breadth.front().b1;
  b2 = breadth.front().b2;
  R = breadth.front().R;
  T = breadth.front().T;
  //if (pass_count!=1)
  temp_guide.push_back(breadth.front());
  breadth.pop();
  // test top level
  csRapidCollider::boxesTested++;
  if (num_passes==12) {
    if (pass_count==216) {
      Barken();
    }
    if (pass_count==186)
      Bark();
  }
  int f1 = obb_disjoint (R, T, b1->GetRadius(), b2->GetRadius());

  if (f1 != 0)
    continue;  // stop processing this test, go to top of loop

  // contact between boxes
  if (b1->IsLeaf() && b2->IsLeaf())
  {
    // it is a leaf pair - compare the polygons therein
    // TrianglesHaveContact uses the model-to-model transforms stored in
    // csRapidCollider::mR, csRapidCollider::mT.

    // this will pass along any OUT_OF_MEMORY return codes which
    // may be generated.
    csCdBBox::TrianglesHaveContact(b1, b2);
    continue;
  }

  csMatrix3 cR;
  csVector3 cT (0, 0, 0);

  // Currently, the transform from model 2 to model 1 space is
  // given by [B T], where y = [B T].x = B.x + T.

  if (b2->IsLeaf() || (!b1->IsLeaf() && (b1->GetSize() > b2->GetSize())))
  {
    // here we descend to children of b1.  The transform from
    // a child of b1 to b1 is stored in
    // [b1->N->m_Rotation,b1->N->m_Translation],
    // but we will denote it [B1 T1] for short.

    // Here, we compute [B1 T1]'[B T] = [B1'B B1'(T-T1)]
    // for each child, and store the transform into the collision
    // test queue.

    csMatrix3 rot_transp = b1->m_pChild0->m_Rotation.GetTranspose ();
    cR = rot_transp * R;
    cT = rot_transp * (T - b1->m_pChild0->m_Translation);

    //if ((rc = CollideRecursive (b1->m_pChild0, b2, cR, cT)) != false)
    //      return rc;
    breadth.push(csTraverser(b1->m_pChild0, b2, cR, cT));

    rot_transp = b1->m_pChild1->m_Rotation.GetTranspose ();
    cR = rot_transp * R;
    cT = rot_transp * (T - b1->m_pChild1->m_Translation);

    breadth.push(csTraverser(b1->m_pChild1, b2, cR, cT));
	
  }
  else
  {
    // here we descend to the children of b2.  See comments for
    // other 'if' clause for explanation.
    cR = R * b2->m_pChild0->m_Rotation;
    cT = ( R * b2->m_pChild0->m_Translation) + T;

    breadth.push(csTraverser (b1, b2->m_pChild0, cR, cT));
 
    cR = R * b2->m_pChild1->m_Rotation;
    cT = ( R * b2->m_pChild1->m_Translation) + T;
	
    breadth.push (csTraverser(b1, b2->m_pChild1, cR, cT));
	
  }
  }
  return false;
}

#endif
typedef struct traverser_t {
  float4 index;//.xy is index into the aTree  .zw is index into bTree
  float4 translation; 
  float4 rotationX;
  float4 rotationY;
  float3 Rotationx() {return float3(rotationX.x,rotationX.y,rotationX.z);}
  float3 Rotationy() {return float3(rotationY.x,rotationY.y,rotationY.z);}
  float3 Rotationz() {return float3(rotationX.w,rotationY.w,translation.w);}
  float3 Translation() {return float3(translation.x,
                                      translation.y,
                                      translation.z);}
}Traverser;
bool assertf(bool b) {
  if (b) {
    return false;
  }return true;
}
void assertv(csVector3 a, float3 b) {
  assert(fabs(a.x-b.x)<.0001);
  assert(fabs(a.y-b.y)<.0001);
  assert(fabs(a.y-b.y)<.0001);
}
bool myeq(float x, float y) {
  return fabs(x-y)<.001;
}
class PassSorter{
public:
  bool operator () (const Traverser &x, const Traverser&y) {
    //return x.index.x<y.index.x;
    return x.index.x<y.index.x||
      (x.index.x==y.index.x&&
       (x.index.y<y.index.y||
        (x.index.y==y.index.y&&
         x.index.z<y.index.z||
         (x.index.z==y.index.z&&
          x.index.w<y.index.w))));
  }
};
class csPassSorter{
public:
  bool operator () (const csTraverser &x, const csTraverser&y) {
    return x.b1[0].ind[0]<y.b1[0].ind[0]||
      (x.b1[0].ind[0]==y.b1[0].ind[0]&&
       (x.b1[0].ind[1]<y.b1[0].ind[1]||
        (x.b1[0].ind[1]==y.b1[0].ind[1]&&
         x.b2[0].ind[0]<y.b2[0].ind[0]||
         (x.b2[0].ind[0]==y.b2[0].ind[0]&&
          x.b2[0].ind[1]<y.b2[0].ind[1]))));
   
  }
};
int checkPassCorrectness(Traverser * traverser,int numTraverser, int numpass) {
  int error_count=0;

  if (numpass==0||numpass==1||numpass>=(int)guide.size())
    return 1;
  std::vector<csTraverser>* b = &guide[numpass];
  bool dosort=true;
  if (dosort) {
    std::sort(traverser,traverser+numTraverser,PassSorter());
    std::sort(b->begin(),b->end(),csPassSorter());
  }
  int j=0;
  for (int i=0;j<(int)b->size()&&i<numTraverser;++i) {
    csTraverser csTrav = (*b)[j];
    Traverser trav=traverser[i];
    if (trav.index.y==0&&trav.index.x<1.5&&trav.index.x>.5) {
      //if (!dosort)++j;
      continue;
    }
    j++;
    if (assertf ( trav.index.x == csTrav.b1->ind[0])
        ||assertf ( trav.index.y == csTrav.b1->ind[1])
        ||assertf ( trav.index.z == csTrav.b2->ind[0])
        ||assertf ( trav.index.w == csTrav.b2->ind[1])) {
      Traverser tmp;
      tmp.index.x = csTrav.b1->ind[0];
      tmp.index.y = csTrav.b1->ind[1];
      tmp.index.z = csTrav.b2->ind[0];
      tmp.index.w = csTrav.b2->ind[1];
      error_count++;
      PassSorter p;
      if (p(tmp,trav)) {
        i--;        
      }else {
        j--;
      }

      continue;
    }
    csVector3 rX = csTrav.R.Row1();
    csVector3 rY = csTrav.R.Row2();
    csVector3 rZ = csTrav.R.Row3();
    csVector3 T = csTrav.T;
    //assertv(rX,trav.Rotationx());
    //assertv(rY,trav.Rotationy());
    //assertv(rZ,trav.Rotationz());
    //assertv(T,trav.Translation());
  }
  if (error_count)
    printf ("error detected %d\n",error_count);
      
  //assert (numTraverser==(int)guide[numpass].size());
  return 1;
}

void csRapidCollider::CollideReset (void)
{
  hits = 0;
  currHit = 0;
}

int csRapidCollider::Report (csRapidCollider **id1, csRapidCollider **id2)
{
  if (currHit >= hits) return 0;

  *id1 = hitv [currHit] [0];
  *id2 = hitv [currHit] [1];
  currHit++;
  return 1;
}

bool csCdBBox::TrianglesHaveContact(csCdBBox *pBox1, csCdBBox *pBox2)
{
  // assume just one triangle in each box.

  // the vertices of the CDTriangle in b2 is in model1 C.S.  The vertices of
  // the other triangle is in model2 CS.
  // Use csRapidCollider::mR, csRapidCollider::mT, and to transform into
  // model2 CS.

  int rc;  // return code

  csVector3 i1 =
    ((csRapidCollider::mR * pBox1->m_pTriangle->p1) + csRapidCollider::mT);
  csVector3 i2 =
    ((csRapidCollider::mR * pBox1->m_pTriangle->p2) + csRapidCollider::mT);
  csVector3 i3 =
    ((csRapidCollider::mR * pBox1->m_pTriangle->p3) + csRapidCollider::mT);

  csRapidCollider::trianglesTested++;

  bool f = ::tri_contact(i1, i2, i3, 
                         pBox2->m_pTriangle->p1, 
                         pBox2->m_pTriangle->p2, 
                         pBox2->m_pTriangle->p3)?true:false;

  if (f)
  {
    // add_collision may be unable to allocate enough memory,
    // so be prepared to pass along an OUT_OF_MEMORY return code.
    if ((rc = add_collision(pBox1->m_pTriangle, pBox2->m_pTriangle)) != false)
      return rc?true:false;
  }

  return false;
}

const csVector3 &csRapidCollider::GetRadius() const 
{
  return GetBbox()->GetRadius();
}

const csCdBBox* csRapidCollider::GetBbox(void) const 
{
  return m_pCollisionModel->GetTopLevelBox();
}

