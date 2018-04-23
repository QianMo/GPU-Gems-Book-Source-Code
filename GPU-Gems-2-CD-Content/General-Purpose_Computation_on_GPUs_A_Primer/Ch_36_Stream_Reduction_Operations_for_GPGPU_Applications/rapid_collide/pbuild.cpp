/*
    Copyright (C) 1998,2000 by Jorrit Tyberghein
  
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

#include "cs_compat.h"
#include "csgeom/matrix3.h"
#include "csgeom/vector3.h"
#include "rapcol.h"
#include "prapid.h"

/**
 * Located in eigen.cpp, this function returns the eigenvectors of M,
 * Sorted so that the vector corresponding to the largest eigenvalue
 * is first.
 */
int SortedEigen (csMatrix3& M, csMatrix3& evecs);

csCdModel::csCdModel (int NumberOfTriangles)
{
  m_pBoxes          = NULL;
  m_NumBoxesAlloced = 0;

  m_pTriangles = new csCdTriangle [NumberOfTriangles];
  m_NumTriangles          = 0;
  m_NumTrianglesAllocated = m_pTriangles ? NumberOfTriangles : 0;
}

csCdModel::~csCdModel ()
{
  // the boxes pointed to should be deleted.
  delete [] m_pBoxes;
  // the triangles pointed to should be deleted.
  delete [] m_pTriangles;
}

bool csCdModel::AddTriangle (const csVector3 &p1, const csVector3 &p2,
  const csVector3 &p3)
{
  // first make sure that we haven't filled up our allocation.
  if (m_NumTriangles >= m_NumTrianglesAllocated)
    return false;

  // now copy the new tri into the array
  m_pTriangles [m_NumTriangles].p1 = p1;
  m_pTriangles [m_NumTriangles].p2 = p2;
  m_pTriangles [m_NumTriangles].p3 = p3;

  // update the counter
  m_NumTriangles++;

  return true;
}

/*
 * There are <n> csCdTriangle structures in an array starting at <t>.
 *
 * We are told that the mean point is <mp> and the orientation
 * for the parent box will be <or>.  The split axis is to be the 
 * vector given by <ax>.
 *
 * <or>, <ax>, and <mp> are model space coordinates.
 */
bool csCdModel::BuildHierarchy ()
{
  // Delete the boxes if they're already allocated.
  delete [] m_pBoxes;

  // Allocate the boxes and set the box list globals.
  m_NumBoxesAlloced = m_NumTriangles * 2;
  m_pBoxes = new csCdBBox [m_NumBoxesAlloced];
  if (!m_pBoxes) return false;
  
  // Determine initial orientation, mean point, and splitting axis.
  int i; 
  Accum _M;
  
  // Should never be allocated at this point, but this is safer.
  delete[] Moment::stack;
  Moment::stack = new Moment[m_NumTriangles];

  if (!Moment::stack)
  {
    delete [] m_pBoxes; 
    m_pBoxes = NULL;
    return false;
  }

  // first collect all the moments, and obtain the area of the 
  // smallest nonzero area triangle.
  float Amin = 0.0;
  int zero = 0;
  int nonzero = 0;
  for (i = 0; i < m_NumTriangles; i++)
  {
    Moment::stack [i].compute (m_pTriangles[i].p1, 
                               m_pTriangles[i].p2, 
                               m_pTriangles[i].p3);
 
    if (Moment::stack[i].A == 0.0)
      zero = 1;
    else
    {
      nonzero = 1;
      if (Amin == 0.0)
        Amin = Moment::stack [i].A;
      else if (Moment::stack [i].A < Amin)
        Amin = Moment::stack [i].A;
    }
  }

  if (zero)
  {
    // if there are any zero area triangles, go back and set their area
    // if ALL the triangles have zero area, then set the area thingy
    // to some arbitrary value. Should never happen.
    if (Amin == 0.0)
      Amin = 1.0;

    for (i = 0; i < m_NumTriangles; i++)
      if (Moment::stack [i].A == 0.0)
        Moment::stack [i].A = Amin;
  }

  _M.clear ();

  for (i = 0; i < m_NumTriangles; i++)
    _M.moment (Moment::stack [i]);

  // csVector3 _pT;
  csMatrix3 mac_C;
  _M.mean (&(m_pBoxes[0].m_Translation));

  _M.covariance (&mac_C);

  SortedEigen(mac_C, m_pBoxes[0].m_Rotation);

  // create the index list
  int *t = new int [m_NumTriangles];
  if (t == 0)
  {
    delete [] Moment::stack; 
    Moment::stack = NULL;
    delete [] m_pBoxes; 
    m_pBoxes = NULL;
    delete [] t;
    return false;
  }
  for (i = 0; i < m_NumTriangles; i++)
    t [i] = i;

  // do the build
  csCdBBox *pool = m_pBoxes + 1;
  if (!m_pBoxes[0].BuildBBoxTree(t, m_NumTriangles, m_pTriangles, pool))
  {
    delete [] m_pBoxes; 
    m_pBoxes = NULL;
    delete [] t;
    return false;
  }
  
  // free the moment list
  delete [] Moment::stack;
  Moment::stack = NULL;

  // free the index list
  delete [] t;

  return true;
}

bool csCdBBox::BuildBBoxTree (
	int* TriangleIndices, 
	int NumTriangles, 
	csCdTriangle* Triangles,
	csCdBBox*& box_pool)
{
  // The orientation for the parent box is already assigned to this->m_Rotation.
  // The axis along which to split will be column 0 of this->m_Rotation.
  // The mean point is passed in on this->m_Translation.

  // When this routine completes, the position and orientation in model
  // space will be established, as well as its dimensions.  Child boxes
  // will be constructed and placed in the parent's CS.

  if (NumTriangles == 1)
    return SetLeaf(&Triangles[TriangleIndices[0]]);
  
  // walk along the triangles for the box, and do the following:
  //   1. collect the max and min of the vertices along the axes of <or>.
  //   2. decide which group the triangle goes in, performing appropriate swap.
  //   3. accumulate the mean point and covariance data for that triangle.

  Accum _M1, _M2;
  csMatrix3 C;

  float axdmp;
  int n1 = 0;  // The number of triangles in group 1.  
  // Group 2 will have n - n1 triangles.

  // project approximate mean point onto splitting axis, and get coord.
  axdmp = (m_Rotation.m11 * m_Translation.x +
           m_Rotation.m21 * m_Translation.y + 
           m_Rotation.m31 * m_Translation.z);

  _M1.clear ();
  _M2.clear ();

  csVector3 c = m_Rotation.GetTranspose () * Triangles [TriangleIndices[0]].p1;
  csVector3 minval = c, maxval = c;

  int i;
  for (i=0 ; i<NumTriangles ; i++)
  {
    int CurrentTriangleIndex = TriangleIndices[i];
    csCdTriangle *ptr = &Triangles[CurrentTriangleIndex];

    c = m_Rotation.GetTranspose () * ptr->p1;
    csMath3::SetMinMax (c, minval, maxval); 

    c = m_Rotation.GetTranspose () * ptr->p2;
    csMath3::SetMinMax (c, minval, maxval); 

    c = m_Rotation.GetTranspose () * ptr->p3;
    csMath3::SetMinMax (c, minval, maxval); 

    // grab the mean point of the in'th triangle, project
    // it onto the splitting axis (1st column of m_Rotation) and
    // see where it lies with respect to axdmp.
     
    Moment::stack[CurrentTriangleIndex].mean (&c);

    if ((( m_Rotation.m11 * c.x + 
           m_Rotation.m21 * c.y + 
           m_Rotation.m31 * c.z) < axdmp)
	  && ((NumTriangles!=2)) || ((NumTriangles==2) && (i==0)))    
    {
      // accumulate first and second order moments for group 1
      _M1.moment (Moment::stack[CurrentTriangleIndex]);
      // put it in group 1 by swapping t[i] with t[n1]
      int temp            = TriangleIndices[i];
      TriangleIndices[i]  = TriangleIndices[n1];
      TriangleIndices[n1] = temp;
      n1++;
    }
    else
    {
      // accumulate first and second order moments for group 2
     _M2.moment (Moment::stack[CurrentTriangleIndex]);
      // leave it in group 2
      // do nothing...it happens by default
    }
  }

  // done using this->m_Translation as a mean point.

  // error check!
  if ((n1 == 0) || (n1 == NumTriangles))
  {
    // our partitioning has failed: all the triangles fell into just
    // one of the groups.  So, we arbitrarily partition them into
    // equal parts, and proceed.

    n1 = NumTriangles/2;
      
    // now recompute accumulated stuff
    _M1.clear ();
    _M2.clear ();
    _M1.moments (TriangleIndices,n1);
    _M2.moments (TriangleIndices+n1,NumTriangles-n1);
  }

  // With the max and min data, determine the center point and dimensions
  // of the parent box.
  c = (minval + maxval) * 0.5f; 
  m_Translation = m_Rotation * c;

  // delta.
  m_Radius = (maxval - minval ) * 0.5f;

  // allocate new boxes
  m_pChild0 = box_pool++;
  m_pChild1 = box_pool++;

  // Compute the orientations for the child boxes (eigenvectors of
  // covariance matrix).  Select the direction of maximum spread to be
  // the split axis for each child.
  csMatrix3 tR;
  if (n1 > 1)
  {
    _M1.mean (&m_pChild0->m_Translation);
    _M1.covariance (&C);

    int nn = SortedEigen(C, tR);
    if ( nn > 30 || nn == -1)
    {
      // unable to find an orientation.  We'll just pick identity.
      tR.Identity ();
    }

    m_pChild0->m_Rotation = tR;
    if (!m_pChild0->BuildBBoxTree (TriangleIndices, n1, 
                                     Triangles, box_pool))
    {
      return false;
    }
  }
  else
  {
    if (!m_pChild0->SetLeaf(&Triangles[TriangleIndices[0]]))
      return false;
  }

  C = m_pChild0->m_Rotation;
  m_pChild0->m_Rotation    = m_Rotation.GetTranspose () * C;

  c = m_pChild0->m_Translation - m_Translation;
  m_pChild0->m_Translation = m_Rotation.GetTranspose () * c;

  if ((NumTriangles-n1) > 1)
  {      
    _M2.mean (&m_pChild1->m_Translation);
    _M2.covariance (&C);
    int nn = SortedEigen(C, tR);

    if (nn > 30 || nn == -1)
    {
      // unable to find an orientation.  We'll just pick identity.
      tR.Identity ();
    }
      
    m_pChild1->m_Rotation = m_Rotation;
    if (!m_pChild1->BuildBBoxTree(TriangleIndices + n1, NumTriangles - n1, 
                                    Triangles, box_pool))
    {
      return false;
    }
  }
  else
  {
    if (!m_pChild1->SetLeaf(&Triangles[TriangleIndices[n1]]))
      return false;
  }

  C = m_pChild1->m_Rotation;
  m_pChild1->m_Rotation = m_Rotation.GetTranspose () * C;
 
  c = m_pChild1->m_Translation - m_Translation;
  m_pChild1->m_Translation = m_Rotation.GetTranspose () * c;

  return true;
}

bool csCdBBox::SetLeaf(csCdTriangle* pTriangle)
{
  // For a single triangle, orientation is easily determined.
  // The major axis is parallel to the longest edge.
  // The minor axis is normal to the triangle.
  // The in-between axis is determine by these two.

  // this->m_Rotation, this->m_Radius, and 
  // this->m_Translation are set herein.

  m_pChild0 = NULL;
  m_pChild1 = NULL;

  // Find the major axis: parallel to the longest edge.
  // First compute the squared-lengths of each edge

  csVector3 u12 = pTriangle->p1 - pTriangle->p2;
  float d12 = u12 * u12;
 
  csVector3 u23 = pTriangle->p2 - pTriangle->p3;
  float d23 = u23 * u23;

  csVector3 u31 = pTriangle->p3 - pTriangle->p1;
  float d31 = u31 * u31;

  // Find the edge of longest squared-length, normalize it to
  // unit length, and put result into a0.
  csVector3 a0;
  float sv; // Return value of the squaroot.

  if (d12 > d23)
  {
    if (d12 > d31)
    {
      a0 = u12;
      sv = d12;
    }
    else 
    {
      a0 = u31;
      sv = d31;
    }
  }
  else 
  {
    if (d23 > d23)
    {
      a0 = u23;
      sv = d23;
    }
    else
    {
      a0 = u31;
      sv = d31;
    }
  }

  sv = sqrt (sv);
  a0 = a0 / (float)(sv > SMALL_EPSILON ? sv : SMALL_EPSILON);
  // Now compute unit normal to triangle, and put into a2.
  csVector3 a2 = u12 % u23;
  if (a2.Norm () != 0) a2 = csVector3::Unit (a2);

  // a1 is a2 cross a0.
  csVector3 a1 = a2 % a0;
  // Now make the columns of this->m_Rotation the vectors a0, a1, and a2.
  m_Rotation.m11 = a0.x; m_Rotation.m12 = a1.x; m_Rotation.m13 = a2.x;
  m_Rotation.m21 = a0.y; m_Rotation.m22 = a1.y; m_Rotation.m23 = a2.y;
  m_Rotation.m31 = a0.z; m_Rotation.m32 = a1.z; m_Rotation.m33 = a2.z;

  // Now compute the maximum and minimum extents of each vertex 
  // along each of the box axes.  From this we will compute the 
  // box center and box dimensions.
  csVector3 c = m_Rotation.GetTranspose () * pTriangle->p1;
  csVector3 minval = c, maxval = c;

  c = m_Rotation.GetTranspose () * pTriangle->p2;
  csMath3::SetMinMax (c, minval, maxval);

  c = m_Rotation.GetTranspose () * pTriangle->p3;
  csMath3::SetMinMax (c, minval, maxval);
 
  // With the max and min data, determine the center point and dimensions
  // of the box
  c = (minval + maxval) * 0.5f;
  m_Translation = m_Rotation * c;

  m_Radius = (maxval - minval) * 0.5f;

  // Assign the one triangle to this box
  m_pTriangle = pTriangle;

  return true;
}

