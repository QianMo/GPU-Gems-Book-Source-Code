/**
 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite       Written by Nate Miller, nathanm@uci.edu
 @created    2002-08-10
 @edited     2002-08-10
 */

#ifndef IFSMODEL_H
#define IFSMODEL_H

#include <g3Dall.h>

class IFSVertex {
private:
   Vector3 pos;
   Vector3 norm;

public:
   const Vector3& getPosition() const {
      return pos;
   }

   inline void setPosition(const Vector3& v) {   
      pos = v;
   }

   inline const Vector3& getNormal() const {
      return norm;
   }

   inline void setNormal(const Vector3& n) {
      norm = n;
   }
};


/**
 Indexed triangle
 */
class IFSTriangle {
private:
   /**
    Vertex indices
    */
   uint32           ndx[3];

public:
   inline const uint32* getIndices(void) const {
      return ndx;
   }

   inline void setIndices(const uint32 &a, const uint32 &b, const uint32 &c) {
      ndx[0] = a;
      ndx[1] = b;
      ndx[2] = c;
   }
};


class IFSModel {
private:
   std::string      modelName;

   uint32           numVerts;
   uint32           numFaces;

   IFSVertex*       verts;
   IFSTriangle*     tris;

public:

   IFSModel();

   virtual ~IFSModel();

   const std::string& getMeshName(void) const  {return modelName;}
   const IFSVertex* getVerts(void) const       {return verts;}
   const IFSTriangle* getTriangles(void) const {return tris;}
   const uint32& getNumTriangles(void) const   {return numFaces;}
   const uint32& getNumVerts(void) const       {return numVerts;}

    /**
    Setup the model.
       mname - mesh name
       nv    - number of verts in the model
       vts   - pointer to the verts that make up the model   
       nt    - numver of triangles in the model
       trs   - pointer to the triangles of the model
    */
   void setup(
       const std::string&       mname,
       uint32                   nv,
       IFSVertex*               vts, 
       uint32                   nt,
       IFSTriangle*             trs);

   /**
    Scale all verts in the model by scaleFactor
   */
   void scaleMesh(float scaleFactor);
};

#endif