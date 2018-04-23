/*
    Copyright (C) 1998,1999,2000 by Jorrit Tyberghein
    Largely rewritten by Ivan Avramovic <ivan@avramovic.com>
  
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

#ifndef __CS_BOX_H__
#define __CS_BOX_H__


#include "vector3.h"


class csPlane3;

/**
 * The maximum value that a coordinate in the bounding box can use.
 * This is considered the 'infinity' value used for empty bounding boxes.
 */
#define CS_BOUNDINGBOX_MAXVALUE 1000000000.


/**
 * Indices of corner vertices for csBox3.
 * Used by csBox3::GetCorner().
 */
#define BOX_CORNER_xyz 0
#define BOX_CORNER_xyZ 1
#define BOX_CORNER_xYz 2
#define BOX_CORNER_xYZ 3
#define BOX_CORNER_Xyz 4
#define BOX_CORNER_XyZ 5
#define BOX_CORNER_XYz 6
#define BOX_CORNER_XYZ 7

/**
 * Indices of faces for csBox3.
 * Used by csBox3::GetSide().
 */
#define BOX_SIDE_x 0
#define BOX_SIDE_X 1
#define BOX_SIDE_y 2
#define BOX_SIDE_Y 3
#define BOX_SIDE_z 4
#define BOX_SIDE_Z 5
#define BOX_INSIDE 6

/**
 * Indices of edges for cxBox3.
 * Index e+1 is opposite edge of e (with e even).
 */
#define BOX_EDGE_Xyz_xyz 0
#define BOX_EDGE_xyz_Xyz 1
#define BOX_EDGE_xyz_xYz 2
#define BOX_EDGE_xYz_xyz 3
#define BOX_EDGE_xYz_XYz 4
#define BOX_EDGE_XYz_xYz 5
#define BOX_EDGE_XYz_Xyz 6
#define BOX_EDGE_Xyz_XYz 7
#define BOX_EDGE_Xyz_XyZ 8
#define BOX_EDGE_XyZ_Xyz 9
#define BOX_EDGE_XyZ_XYZ 10
#define BOX_EDGE_XYZ_XyZ 11
#define BOX_EDGE_XYZ_XYz 12
#define BOX_EDGE_XYz_XYZ 13
#define BOX_EDGE_XYZ_xYZ 14
#define BOX_EDGE_xYZ_XYZ 15
#define BOX_EDGE_xYZ_xYz 16
#define BOX_EDGE_xYz_xYZ 17
#define BOX_EDGE_xYZ_xyZ 18
#define BOX_EDGE_xyZ_xYZ 19
#define BOX_EDGE_xyZ_xyz 20
#define BOX_EDGE_xyz_xyZ 21
#define BOX_EDGE_xyZ_XyZ 22
#define BOX_EDGE_XyZ_xyZ 23

/**
 * A bounding box in 3D space.
 * In order to operate correctly, this bounding box assumes that all values
 * entered or compared against lie within the range 
 * (-CS_BOUNDINGBOX_MAXVALUE, CS_BOUNDINGBOX_MAXVALUE).  It is not
 * recommended to use points outside of this range.
 */
class csBox3
{
protected:
  /// The top-left of this bounding box.
  csVector3 minbox;
  /// The bottom-right.
  csVector3 maxbox;

  struct bEdge
  {
    uint8 v1, v2; // Indices of vertex in bounding box (BOX_CORNER_...)
    uint8 fl, fr; // Indices of left/right faces sharing edge (BOX_SIDE_...)
  };
  typedef uint8 bFace[4];	// Indices of four clock-wise edges (0..23)
  // Index by edge number. Edge e and e+1 with e even are opposite edges.
  // (BOX_EDGE_...)
  static bEdge edges[24];
  // Index by BOX_SIDE_? number.
  static bFace faces[6];
public:
  /// Get the minimum X value of the box
  float MinX () const { return minbox.x; }
  /// Get the minimum Y value of the box
  float MinY () const { return minbox.y; }
  /// Get the minimum Z value of the box
  float MinZ () const { return minbox.z; }
  /// Get the maximum X value of the box
  float MaxX () const { return maxbox.x; }
  /// Get the maximum Y value of the box
  float MaxY () const { return maxbox.y; }
  /// Get the maximum Z value of the box
  float MaxZ () const { return maxbox.z; }
  /// Get Min component for 0 (x), 1 (y), or 2 (z).
  float Min (int idx) const
  { return idx == 1 ? minbox.y : idx == 0 ? minbox.x : minbox.z; }
  /// Get Max component for 0 (x), 1 (y), or 2 (z).
  float Max (int idx) const
  { return idx == 1 ? maxbox.y : idx == 0 ? maxbox.x : maxbox.z; }
  /// Get the 3d vector of minimum (x, y, z) values
  const csVector3& Min () const { return minbox; }
  /// Get the 3d vector of maximum (x, y, z) values
  const csVector3& Max () const { return maxbox; }

  /**
   * Return every corner of this bounding box from 0
   * to 7. This contrasts with Min() and Max() because
   * those are only the min and max corners.
   * Corner 0 = xyz, 1 = xyZ, 2 = xYz, 3 = xYZ,
   *        4 = Xyz, 5 = XyZ, 6 = XYz, 7 = XYZ.
   * Use BOX_CORNER_??? defines.
   */
  csVector3 GetCorner (int corner) const;

  /**
   * Given an edge index (BOX_EDGE_???) return the two vertices
   * (index BOX_CORNER_???) and left/right faces (BOX_SIDE_???).
   */
  void GetEdgeInfo (int edge, int& v1, int& v2, int& fleft, int& fright) const
  {
    v1 = edges[edge].v1;
    v2 = edges[edge].v2;
    fleft = edges[edge].fl;
    fright = edges[edge].fr;
  }

  /**
   * Given a face index (BOX_SIDE_???) return the four edges oriented
   * clockwise around this face (BOX_EDGE_???).
   */
  uint8* GetFaceEdges (int face) const
  {
    return faces[face];
  }
  
  /**
   * Get the center of this box.
   */
  csVector3 GetCenter () const { return (minbox+maxbox)/2; }

  /**
   * Set the center of this box. This will not change the size
   * of the box but just relocate the center.
   */
  void SetCenter (const csVector3& c);

  /**
   * Set the size of the box but keep the center intact.
   */
  void SetSize (const csVector3& s);

  /**
   * Get a side of this box as a 2D box.
   * Use BOX_SIDE_??? defines.
   
  csBox2 GetSide (int side) const;
  */
  /**
   * Fill the array (which should be three long at least)
   * with all visible sides (BOX_SIDE_??? defines) as seen
   * from the given point.
   * Returns the number of visible sides.
   */
  int GetVisibleSides (const csVector3& pos, int* visible_sides) const;

  /**
   * Static function to get the 'other' side (i.e. BOX_SIDE_X
   * to BOX_SIDE_x, ...).
   */
  static int OtherSide (int side)
  {
    return side ^ 1;
  }

  /**
   * Return every edge (segment) of this bounding box
   * from 0 to 23 (use one of the BOX_EDGE_??? indices).
   * The returned edge is undefined for any other index.
   *
  csSegment3 GetEdge (int edge) const
  {
    return csSegment3 (GetCorner (edges[edge].v1), GetCorner (edges[edge].v2));
    }*/

  /**
   * Return every edge (segment) of this bounding box
   * from 0 to 23 (use one of the BOX_EDGE_??? indices).
   * The returned edge is undefined for any other index.
   *
  void GetEdge (int edge, csSegment3& e) const
  {
    e.SetStart (GetCorner (edges[edge].v1));
    e.SetEnd (GetCorner (edges[edge].v2));
  }
  */
  /// Test if the given coordinate is in this box.
  bool In (float x, float y, float z) const
  {
    if (x < minbox.x || x > maxbox.x) return false;
    if (y < minbox.y || y > maxbox.y) return false;
    if (z < minbox.z || z > maxbox.z) return false;
    return true;
  }

  /// Test if the given coordinate is in this box.
  bool In (const csVector3& v) const
  {
    return In (v.x, v.y, v.z);
  }

  /// Test if this box overlaps with the given box.
  bool Overlap (const csBox3& box) const
  {
    if (maxbox.x < box.minbox.x || minbox.x > box.maxbox.x) return false;
    if (maxbox.y < box.minbox.y || minbox.y > box.maxbox.y) return false;
    if (maxbox.z < box.minbox.z || minbox.z > box.maxbox.z) return false;
    return true;
  }

  /// Test if this box contains the other box.
  bool Contains (const csBox3& box) const
  {
    return (box.minbox.x >= minbox.x && box.maxbox.x <= maxbox.x) &&
           (box.minbox.y >= minbox.y && box.maxbox.y <= maxbox.y) &&
           (box.minbox.z >= minbox.z && box.maxbox.z <= maxbox.z);
  }

  /// Test if this box is empty.
  bool Empty () const
  {
    if (minbox.x > maxbox.x) return true;
    if (minbox.y > maxbox.y) return true;
    if (minbox.z > maxbox.z) return true;
    return false;
  }

  /// Initialize this box to empty.
  void StartBoundingBox ()
  {
    minbox.x =  CS_BOUNDINGBOX_MAXVALUE;
    minbox.y =  CS_BOUNDINGBOX_MAXVALUE;
    minbox.z =  CS_BOUNDINGBOX_MAXVALUE;
    maxbox.x = -CS_BOUNDINGBOX_MAXVALUE;
    maxbox.y = -CS_BOUNDINGBOX_MAXVALUE;
    maxbox.z = -CS_BOUNDINGBOX_MAXVALUE;
  }

  /// Initialize this box to one vertex.
  void StartBoundingBox (const csVector3& v)
  {
    minbox = v; maxbox = v;
  }

  /// Add a new vertex and recalculate the bounding box.
  void AddBoundingVertex (float x, float y, float z)
  {
    if (x < minbox.x) minbox.x = x; if (x > maxbox.x) maxbox.x = x;
    if (y < minbox.y) minbox.y = y; if (y > maxbox.y) maxbox.y = y;
    if (z < minbox.z) minbox.z = z; if (z > maxbox.z) maxbox.z = z;
  }

  /// Add a new vertex and recalculate the bounding box.
  void AddBoundingVertex (const csVector3& v)
  {
    AddBoundingVertex (v.x, v.y, v.z);
  }

  /**
   * Add a new vertex and recalculate the bounding box.
   * This version is a little more optimal. It assumes however
   * that at least one point has been added to the bounding box.
   */
  void AddBoundingVertexSmart (float x, float y, float z)
  {
    if (x < minbox.x) minbox.x = x; else if (x > maxbox.x) maxbox.x = x;
    if (y < minbox.y) minbox.y = y; else if (y > maxbox.y) maxbox.y = y;
    if (z < minbox.z) minbox.z = z; else if (z > maxbox.z) maxbox.z = z;
  }

  /**
   * Add a new vertex and recalculate the bounding box.
   * This version is a little more optimal. It assumes however
   * that at least one point has been added to the bounding box.
   */
  void AddBoundingVertexSmart (const csVector3& v)
  {
    AddBoundingVertexSmart (v.x, v.y, v.z);
  }

  //-----
  // Maintenance Note: The csBox3 constructors and Set() appear at this point
  // in the file, rather than earlier, in order to appease the OpenStep 4.2
  // compiler.  Specifically, the problem is that the compiler botches code
  // generation if an unseen method (which is later declared inline) is
  // called from within another inline method.  For instance, if the
  // constructors were listed at the top of the file, rather than here, the
  // compiler would see calls to Empty() and StartBoundingBox() before seeing
  // declarations for them.  In such a situation, the buggy compiler
  // generated a broken object file.  The simple work-around of textually
  // reorganizing the file ensures that the declarations for Empty() and
  // StartBoundingBox() are seen before they are called.
  //-----

  /// Initialize this box to empty.
  csBox3 () :
    minbox ( CS_BOUNDINGBOX_MAXVALUE,
             CS_BOUNDINGBOX_MAXVALUE,
	     CS_BOUNDINGBOX_MAXVALUE),
    maxbox (-CS_BOUNDINGBOX_MAXVALUE,
            -CS_BOUNDINGBOX_MAXVALUE,
	    -CS_BOUNDINGBOX_MAXVALUE) {}

  /// Initialize this box with one point.
  csBox3 (const csVector3& v) : minbox (v), maxbox (v) { }

  /// Initialize this box with two points.
  csBox3 (const csVector3& v1, const csVector3& v2) :
  	minbox (v1), maxbox (v2)
  { if (Empty ()) StartBoundingBox (); }

  /// Initialize this box with the given values.
  csBox3 (float x1, float y1, float z1, float x2, float y2, float z2) :
    minbox (x1, y1, z1), maxbox (x2, y2, z2)
  { if (Empty ()) StartBoundingBox (); }

  /// Sets the bounds of the box with the given values.
  void Set (const csVector3& bmin, const csVector3& bmax)
  {
    minbox = bmin;
    maxbox = bmax;
  }

  /// Sets the bounds of the box with the given values.
  void Set (float x1, float y1, float z1, float x2, float y2, float z2)
  {
    if (x1>x2 || y1>y2 || z1>z2) StartBoundingBox();
    else
    {
      minbox.x = x1; minbox.y = y1; minbox.z = z1;
      maxbox.x = x2; maxbox.y = y2; maxbox.z = z2;
    }
  }

  /**
   * Test if this box is adjacent to the other on the X side.
   */
  bool AdjacentX (const csBox3& other) const;

  /**
   * Test if this box is adjacent to the other on the Y side.
   */
  bool AdjacentY (const csBox3& other) const;

  /**
   * Test if this box is adjacent to the other on the Z side.
   */
  bool AdjacentZ (const csBox3& other) const;

  /**
   * Test if this box is adjacent to the other one.
   * Return -1 if not adjacent or else any of the BOX_SIDE_???
   * flags to indicate the side of this box that the other
   * box is adjacent with.
   */
  int Adjacent (const csBox3& other) const;

  /**
   * Get a convex outline (not a polygon unless projected to 2D)
   * for for this box as seen from the given position.
   * The coordinates returned are world space coordinates.
   * Note that you need place for at least six vectors in the array.
   * If you set bVisible true, you will get all visible corners - this
   * could be up to 7.
   */
  void GetConvexOutline (const csVector3& pos,
  	csVector3* array, int& num_array, bool bVisible=false) const;

  /**
   * Test if this box is between two others.
   */
  bool Between (const csBox3& box1, const csBox3& box2) const;

  /**
   * Calculate the minimum manhattan distance between this box
   * and another one.
   */
  void ManhattanDistance (const csBox3& other, csVector3& dist) const;

  /**
   * Calculate the squared distance between (0,0,0) and the box
   * This routine is extremely efficient.
   */
  float SquaredOriginDist () const;

  /**
   * Calculate the squared distance between (0,0,0) and the point
   * on the box which is furthest away from (0,0,0).
   * This routine is extremely efficient.
   */
  float SquaredOriginMaxDist () const;

  /// Compute the union of two bounding boxes.
  csBox3& operator+= (const csBox3& box);
  /// Compute the union of a point with this bounding box.
  csBox3& operator+= (const csVector3& point);
  /// Compute the intersection of two bounding boxes.
  csBox3& operator*= (const csBox3& box);

  /// Compute the union of two bounding boxes.
  friend csBox3 operator+ (const csBox3& box1, const csBox3& box2);
  /// Compute the union of a bounding box and a point.
  friend csBox3 operator+ (const csBox3& box, const csVector3& point);
  /// Compute the intersection of two bounding boxes.
  friend csBox3 operator* (const csBox3& box1, const csBox3& box2);

  /// Tests if two bounding boxes are equal.
  friend bool operator== (const csBox3& box1, const csBox3& box2);
  /// Tests if two bounding boxes are unequal.
  friend bool operator!= (const csBox3& box1, const csBox3& box2);
  /// Tests if box1 is a subset of box2.
  friend bool operator< (const csBox3& box1, const csBox3& box2);
  /// Tests if box1 is a superset of box2.
  friend bool operator> (const csBox3& box1, const csBox3& box2);
  /// Tests if a point is contained in a box.
  friend bool operator< (const csVector3& point, const csBox3& box);
};

#endif // __CS_BOX_H__
