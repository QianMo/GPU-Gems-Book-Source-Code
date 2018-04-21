/**
 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite       Written by Nate Miller, nathanm@uci.edu
 @created    2002-08-10
 @edited     2002-08-10
 */

#include "IFSModel.h"

IFSModel::IFSModel() : 
    numFaces(0),
    numVerts(0),
    verts(0),
    tris(0) {
}


IFSModel::~IFSModel() {
    delete [] verts;
    delete [] tris;
}


void IFSModel::setup(
    const std::string&  mname,
    uint32              nv,
    IFSVertex*          vts, 
    uint32              nt,
    IFSTriangle*        trs) {

   debugAssertM(nv > 0, "Bad number of model verts");
   debugAssertM(vts,    "Null vertex pointer");
   debugAssertM(nt > 0, "Bad number of model triangles");
   debugAssertM(trs,    "Null triangle pointer");

   modelName    = mname;
   numVerts     = nv;
   verts        = vts;
   numFaces     = nt;
   tris         = trs;

   // Compute per-face normals.
   Array<Vector3> faceNormal;
   faceNormal.resize(numFaces);
   for (int f = 0; f < numFaces; ++f) {
      const uint32* ndx = tris[f].getIndices();

      const Vector3& v0 = verts[ndx[0]].getPosition();
      const Vector3& v1 = verts[ndx[1]].getPosition();
      const Vector3& v2 = verts[ndx[2]].getPosition();

      faceNormal[f] = (v1 - v0).cross(v2 - v1);
   }

   // Calcuate per-vertex normals.  This method weighs
   // each adjacent face normal by its area.  This is fast
   // and easy to implement.  An alternative method is to
   // weigh each face normal by the angle it subtends around
   // the vertex.  That method produces normals that
   // are independent of tesselation but requires adjacency
   // information in order to run in linear time.
   for (f = 0; f < numFaces; ++f) {
      const uint32* ndx = tris[f].getIndices();

      // Not unit length
      const Vector3& fn = faceNormal[f];

      const Vector3& n0 = verts[ndx[0]].getNormal();
      const Vector3& n1 = verts[ndx[1]].getNormal();
      const Vector3& n2 = verts[ndx[2]].getNormal();

      verts[ndx[0]].setNormal(n0 + fn);
      verts[ndx[1]].setNormal(n1 + fn);
      verts[ndx[2]].setNormal(n2 + fn);
   }

   for (int v = 0; v < numVerts; ++v) {
      verts[v].setNormal(verts[v].getNormal().unit());
   }
}

void IFSModel::scaleMesh(float scaleFactor) {

    for (int i = 0; i < numVerts; ++i) {
        verts[i].setPosition(verts[i].getPosition() * scaleFactor);
    }
}  
