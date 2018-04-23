/*
    Copyright (C) 1998 by Jorrit Tyberghein
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

/***************************************************************************/

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


/// A triangle, to be used in collision detection
struct csCdTriangle
{
  /// The three edges of the triangle
  csVector3 p1, p2, p3;
};

/**
  * A bounding box, used in collision detection. Any bounding box, can 
  * either be a node or a leaf. A leaf will contain a single polygon, while 
  * a node contains pointers to two other bounding boxes. This means, that 
  * this class in fact represents a tree of hierarchical bounding boxes.
  * THIS CLASS IS FOR INTERNAL USE OF COLLISION DETECTION
  */
class csCdBBox
{
  friend class csCdModel;
  friend class csRapidCollider;

public:
  /// Pointer to the contained triangle. May be NULL, if the BBox is a node.
  csCdTriangle* m_pTriangle;

  // placement in parent's space
  // box (x_b) to parent (x_m) space: x_m = m_Rotation*x_b + m_Translation
  // parent (x_m) to box (x_b) space: x_b = m_Rotation.T()*(x_m - m_Translation)
  csMatrix3 m_Rotation;    
  csVector3 m_Translation; 

  // this is "radius", that is, half the measure of a side length
  csVector3 m_Radius;

  /**
    * Pointers to child boxes. These pointers are only for reference, they
    * do not indicate ownership. (these boxes are deleted elsewhere)
    */
  csCdBBox* m_pChild0, * m_pChild1;

  /**
    * Checks if two Bounding Boxes do collide. Thes routine assumes, 
    * that each Bounding Box contains _exactly_ one Triangle in 
    * m_pTriangle!
    */
  static bool TrianglesHaveContact (csCdBBox* pBox1, csCdBBox* pBox2);

  /**
    * Assign a Triangle to this Bounding box. This will make this 
    * Bounding Box into a leaf.
    */
  bool SetLeaf(csCdTriangle* pTriangle);

  /**
    * Build a tree structure of Bounding Boxes. TriangleIndices is an array 
    * of indices into the Triangles array. NumTriangles is the number of 
    * valid indices in the TriangleIndices array. The idea behind this (at 
    * first glance very odd) datastructure is, to keep the original order 
    * of "Triangles" intact, and only shuffle the much smaller indices. In 
    * fact, there is only one TriangleInidices array, that is sorted over 
    * and over again and then passed recursively in two halfs to the same 
    * routine again and again, until there are only leafes left.
    */
  bool BuildBBoxTree(int*          TriangleIndices, 
                     int           NumTriangles, 
                     csCdTriangle* Triangles,
                     csCdBBox*&    pBoxPool);

  /**
    * returns true, if this is a leaf bounding box, Maybe, this would be 
    * faster and more secure, if we would return true, if m_pTriangle is 
    * set. For this we need to make sure, that m_pTriangle is always
    * properly initialised to NULL, which is currently not the case.
    * - thieber 14.03.2000 -
    */
  bool IsLeaf() const { return (!m_pChild0 && !m_pChild1); } 
  
  /**
    * return the size of the bounding box. Why this returns d.x and not 
    * d.y or d.z is not obious to me. - thieber 13.03.2000 -
    */
  float GetSize() const { return m_Radius.x; } 

public:
  float ind[2];
  /// Construct a default bounding box
  csCdBBox() :
  	m_pTriangle (NULL),
  	m_Translation(0, 0, 0),
	m_Radius(0, 0, 0),
  	m_pChild0 (NULL),
	m_pChild1 (NULL)
  { }

  /// returns the "Radius", that is, half the measure of each side's length
  const csVector3& GetRadius() const {return m_Radius;}
};

/**
  * This class organizes a set of triangles for collision detection.
  * This class is used by csRapidCollider to handle 3D sprites and polygon
  * sets in a uniform way. This class is also responsible for allocating
  * and freeing memory for the bounding boxes and the triangles.
  * THIS CLASS IS FOR INTERNAL USE OF COLLISION DETECTION
  */
class csCdModel
{
  friend class csRapidCollider;
protected:
  //------ BOXES ----------------
  /// An array containing all the bounding boxes to be used in this model
  csCdBBox* m_pBoxes;
  /// The number of boxes in this array. (twice the number of triangles...)
  int m_NumBoxesAlloced;
  //------------------------

  //------ TRIANGLES ------------
  /// All triangles that appear in this model
  csCdTriangle* m_pTriangles;
  int m_NumTriangles;
  int m_NumTrianglesAllocated;
  //------------------------
  
  /// Build a tree of bounding boxes from the given Triangles
  bool BuildHierarchy();

public:

  /// Create a model object given number of triangles
  csCdModel(int NumberOfTriangles);

  /// Free the memory allocated for this model
  ~csCdModel();

  csCdBBox* GetTopLevelBox() {return &m_pBoxes[0];}

  /// Add a triangle to the model
  bool AddTriangle (const csVector3& p1, 
                    const csVector3& p2, 
                    const csVector3& p3);
};

/***************************************************************************/

// this is the collision query invocation.  It assumes that the 
// models are not being scaled up or down, but have their native
// dimensions.

// Classes to organize triangles in bounding boxes with.
class Moment
{
public:
  float A;	// Area of triangle.
  csVector3 m;	// Centriod.
  csMatrix3 s;	// Moment.
  inline void mean( csVector3 *v ) { *v = m; }
  static Moment *stack;

  inline void compute(csVector3 p, csVector3 q, csVector3 r)
  {
    csVector3 u, v, w;

    // compute the area of the triangle
    u = q - p;
    v = r - p;
    w = u % v; 

    if (ABS (w.x)+ABS (w.y)+ABS (w.z) > SMALL_EPSILON)
        A = 0.5f * w.Norm();
    else
        A = 0.0f;

    // centroid
    m = (p + q + r) /3;

    if (A == 0.0f)
    {
      // This triangle has zero area.  The second order components
      // would be eliminated with the usual formula, so, for the 
      // sake of robustness we use an alternative form.  These are the 
      // centroid and second-order components of the triangle's vertices.

      // second-order components
      s.m11 = (p.x*p.x + q.x*q.x + r.x*r.x);
      s.m12 = (p.x*p.y + q.x*q.y + r.x*r.y);
      s.m13 = (p.x*p.z + q.x*q.z + r.x*r.z);
      s.m22 = (p.y*p.y + q.y*q.y + r.y*r.y);
      s.m23 = (p.y*p.z + q.y*q.z + r.y*r.z);
      s.m33 = (p.z*p.z + q.z*q.z + r.z*r.z);      
    }
    else
    {
      // get the second order components weighted by the area
      s.m11 = A*(9*m.x*m.x+p.x*p.x+q.x*q.x+r.x*r.x)/12;
      s.m12 = A*(9*m.x*m.y+p.x*p.y+q.x*q.y+r.x*r.y)/12;
      s.m22 = A*(9*m.y*m.y+p.y*p.y+q.y*q.y+r.y*r.y)/12;
      s.m13 = A*(9*m.x*m.z+p.x*p.z+q.x*q.z+r.x*r.z)/12;
      s.m23 = A*(9*m.y*m.z+p.y*p.z+q.y*q.z+r.y*r.z)/12;
      s.m33 = A*(9*m.z*m.z+p.z*p.z+q.z*q.z+r.z*r.z)/12;
    }
    s.m32 = s.m23;
    s.m21 = s.m12;
    s.m31 = s.m13;
  } 
};

class Accum : public Moment
{
public:
  inline void clear ()
  {
    A = m.x = m.y = m.z = 0;
    s.m11 = s.m12 = s.m13 = 0;
    s.m21 = s.m22 = s.m23 = 0;
    s.m31 = s.m32 = s.m33 = 0;
  }
  inline void moment( Moment b )
  { m = m + b.m * b.A; s = s + b.s;  A += b.A; }
  inline void mean( csVector3 *v )
  { *v = m / A; }
  inline void covariance( csMatrix3 *C )
  {
    C->m11 = s.m11 - m.x*m.x/A;
    C->m21 = s.m21 - m.y*m.x/A;
    C->m31 = s.m31 - m.z*m.x/A;
    C->m12 = s.m12 - m.x*m.y/A;
    C->m22 = s.m22 - m.y*m.y/A;
    C->m32 = s.m32 - m.z*m.y/A;
    C->m13 = s.m13 - m.x*m.z/A;
    C->m23 = s.m23 - m.y*m.z/A;
    C->m33 = s.m33 - m.z*m.z/A;
  }
  inline void moments(int *t, int n)
  {
    clear ();
    for (int i = 0; i < n; i++)
      moment (Moment::stack [t[i]]);
  }
};

