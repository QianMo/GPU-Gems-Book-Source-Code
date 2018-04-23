/*
    Copyright (C) 1998 by Jorrit Tyberghein
  
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

#include <math.h>
#include "cs_compat.h"
#include "qint.h"
#include "csgeom/matrix3.h"

//---------------------------------------------------------------------------

#define rotate(a1,a2) g=a1; h=a2; a1=g-s*(h+g*tau); a2=h+s*(g-h*tau);
#define swap(a,b)     { float t = a; a = b; b = t; }

// M is the matrix for which we seek to find the eigen values.
// vout is the matrix of eigen vectors.
// dout is the vector of dominant eigen values.
// returns:  -1   - error failed to converge within 50 iterations.
//           0-50 - number of iterations.
int Eigen (csMatrix3& M, csMatrix3& vout, csVector3& dout)
{
  int i;
  float tresh,theta,tau,t,sm,s,h,g,c;
  int nrot;

  csMatrix3 v;
  csVector3 z (0, 0, 0);

  // Load b and d with the diagonals of a.
  csVector3 b (M.m11, M.m22, M.m33), d (M.m11, M.m22, M.m33);

  nrot = 0;
  
  // Try up to 50 times.
  for(i=0; i<50; i++)
    {
      // See if bottom half of matrix a is non zero.
      sm=0.0; sm+=ABS (M.m12); sm+=ABS (M.m13); sm+=ABS (M.m23);
      // If it is 0 we are done.  Return the current vector v and d.
      if (sm == 0.0)
	{
	  vout = v;
	  dout = d;
	  return i;
	}
      
      if (i < 3) tresh=0.2f*sm/(3*3); else tresh=0.0;
      
      // Try rotations in 1st dimension
      {
	g = 100.0f*ABS (M.m12);  
	// Does this make sense??
	// equiv to   if (i>3 && g == 0) 
	if (i>3 && ABS (d.x)+g==ABS (d.x) && ABS (d.y)+g==ABS (d.y))
	  M.m12 =0.0f;
	else if (ABS (M.m12)>tresh)
	  {
	    h = d.y-d.x;
	    if (ABS (h)+g == ABS (h)) t=(M.m12)/h;
	    else
	      {
		theta=0.5f*h/(M.m12);
		t=1.0f/(ABS (theta)+sqrt(1.0f+theta*theta));
		if (theta < 0.0) t = -t;
	      }
	    c=1.0f/sqrt(1+t*t); s=t*c; tau=s/(1.0f+c); h=t*M.m12;
	    z.x -= h; z.y += h; d.x -= h; d.y += h;
	    M.m12=0.0f;
	    rotate(M.m13,M.m23); rotate(v.m11,v.m12);
            rotate(v.m21,v.m22); rotate(v.m31,v.m32); 
	    nrot++;
	  }
      }

      // Try rotations in the 2nd dimension.
      {
	g = 100.0f*ABS (M.m13);
	// See above, can be simplified.
	if (i>3 && ABS (d.x)+g==ABS (d.x) && ABS (d.z)+g==ABS (d.z))
	  M.m13=0.0f;
	else if (ABS (M.m13)>tresh)
	  {
	    h = d.z-d.x;
	    if (ABS (h)+g == ABS (h)) t=(M.m13)/h;
	    else
	      {
		theta=0.5f*h/(M.m13);
		t=1.0f/(ABS (theta)+sqrt(1.0f+theta*theta));
		if (theta < 0.0f) t = -t;
	      }
	    c=1.0f/sqrt(1+t*t); s=t*c; tau=s/(1.0f+c); h=t*M.m13;
	    z.x -= h; z.z += h; d.x -= h; d.z += h;
	    M.m13=0.0f;
	    rotate(M.m12,M.m23); rotate(v.m11,v.m13);
            rotate(v.m21,v.m23); rotate(v.m31,v.m33); 
	    nrot++;
	  }
      }


      // Try rotations in 3rd dimension.
      {
	g = 100.0f*ABS (M.m23);
	if (i>3 && ABS (d.y)+g==ABS (d.y) && ABS (d.z)+g==ABS (d.z))
	  M.m23=0.0f;
	else if (ABS (M.m23)>tresh)
	  {
	    h = d.z-d.y;
	    if (ABS (h)+g == ABS (h)) t=(M.m23)/h;
	    else
	      {
		theta=0.5f*h/(M.m23);
		t=1.0f/(ABS (theta)+sqrt(1.0f+theta*theta));
		if (theta < 0.0f) t = -t;
	      }
	    c=1.0f/sqrt(1+t*t); s=t*c; tau=s/(1.0f+c); h=t*M.m23;
	    z.y -= h; z.z += h; d.y -= h; d.z += h;
	    M.m23=0.0;
	    rotate(M.m12,M.m13); rotate(v.m12,v.m13);
            rotate(v.m22,v.m23); rotate(v.m32,v.m33); 
	    nrot++;
	  }
      }

      b = b + z; d = b; z.Set( 0, 0, 0);      
    }

  return -1;
}

// Find eigen vectors of matrix M and sort them to have the largest
// eigen vector in the first column.
int SortedEigen (csMatrix3& M, csMatrix3& evecs)
{
  int n;
  csVector3 evals (0, 0, 0);

  n = Eigen (M, evecs, evals);

  if (evals.z > evals.x)
    {
      if (evals.z > evals.y)
	{
	  // 3 is largest, swap with column 1
	  swap(evecs.m13,evecs.m11);
	  swap(evecs.m23,evecs.m21);
	  swap(evecs.m33,evecs.m31);
	}
      else
	{
	  // 2 is largest, swap with column 1
	  swap(evecs.m12,evecs.m11);
	  swap(evecs.m22,evecs.m21);
	  swap(evecs.m32,evecs.m31);
	}
    }
  else
    {
      if (evals.x > evals.y)
	{
	  // 1 is largest, do nothing
	}
      else
	{
  	  // 2 is largest
	  swap(evecs.m12,evecs.m11);
	  swap(evecs.m22,evecs.m21);
	  swap(evecs.m32,evecs.m31);
	}
    }
  // we are returning the number of iterations Meigen took.
  // too many iterations means our chosen orientation is bad.
  return n; 
}
