#ifdef USE_RCSID
static const char RCSid_RigidHexahedron[] = "$Id$";
#endif

/*----------------------------------------------------------------------
|
|
| $Log$
|
|
+---------------------------------------------------------------------*/

#include "RigidHexahedron.hh"

// #ifdef OUTLINE
// #include "RigidHexahedron.in"
// #endif 

RigidHexahedron::RigidHexahedron (float fSize, float fMass, const GbVec3<float>& rkPos,
				  const GbVec3<float>& rkLinMom, const GbVec3<float>& rkAngMom)
    : RigidShape()
{
    debugmsg("cube scale "<<fSize<<" mass "<<fMass);

    // appearance geometry
    meshVertices_.resize(8);
    meshVertices_[0] = GbVec3<float>(0.0f,0.0f,0.0f);   // front
    meshVertices_[1] = GbVec3<float>(fSize,0.0f,0.0f);
    meshVertices_[2] = GbVec3<float>(fSize,fSize,0.0f);
    meshVertices_[3] = GbVec3<float>(0.0f,fSize,0.0f);

    meshVertices_[4] = GbVec3<float>(0.0f,0.0f,fSize);  // back
    meshVertices_[5] = GbVec3<float>(fSize,0.0f,fSize);
    meshVertices_[6] = GbVec3<float>(fSize,fSize,fSize);
    meshVertices_[7] = GbVec3<float>(0.0f,fSize,fSize);

    // topology
    meshIndices_.resize(12);
    meshIndices_[ 0] = GbVec3i<int>(0, 3, 1); // front
    meshIndices_[ 1] = GbVec3i<int>(3, 2, 1);

    meshIndices_[ 2] = GbVec3i<int>(4, 5, 7); // back
    meshIndices_[ 3] = GbVec3i<int>(5, 6, 7);

    meshIndices_[ 4] = GbVec3i<int>(0, 4, 3); // left
    meshIndices_[ 5] = GbVec3i<int>(4, 7, 3);

    meshIndices_[ 6] = GbVec3i<int>(1, 2, 6); // right
    meshIndices_[ 7] = GbVec3i<int>(1, 6, 5);

    meshIndices_[ 8] = GbVec3i<int>(3, 7, 6); // top
    meshIndices_[ 9] = GbVec3i<int>(3, 6, 2);

    meshIndices_[10] = GbVec3i<int>(0, 1, 4); // bottom
    meshIndices_[11] = GbVec3i<int>(1, 5, 4);

    // convex hull geometry same as appearance
    hullVertices_.assign(meshVertices_.begin(),meshVertices_.end());

    // half spaces
    hullIndices_.resize(6);
    hullIndices_[0] = meshIndices_[0];
    hullIndices_[1] = meshIndices_[2];
    hullIndices_[2] = meshIndices_[4];
    hullIndices_[3] = meshIndices_[6];
    hullIndices_[4] = meshIndices_[8];
    hullIndices_[5] = meshIndices_[10];

    const float invNv = 1.0f / 8.0f;

    // inertia tensor
    GbMatrix3<float> kInertia;
    for (int i = 0; i < 3; i++) 
    {
        kInertia[i][i] = 0.0f;
        for (int j = 0; j < 3; j++) 
	{
            if (i != j) 
	    {
                kInertia[i][j] = 0.0f;
                for (int k = 0; k < 8; k++) 
		{
                    kInertia[i][i] += invNv * fMass * hullVertices_[k][j] * hullVertices_[k][j];
                    kInertia[i][j] -= invNv * fMass * hullVertices_[k][i] * hullVertices_[k][j];
                }
            }
        }
    }

    // bounding sphere radius
    const GbVec3<float> kCentroid = fSize * GbVec3<float>(0.5f,0.5f,0.5f);
    radius_ = 0.0f;
    for (int j = 0; j < 8; j++) 
    {
        meshVertices_[j] -= kCentroid;
        hullVertices_[j] -= kCentroid;
	const float fTemp = hullVertices_[j].getNorm();
        if (fTemp > radius_) 
	{
            radius_ = fTemp;
        }
    }
    debugmsg("broadphase radius "<<radius_);

    // set 
    setMass(fMass);
    setBodyInertia(kInertia);
    setPosition(rkPos);
    setQOrientation(GbQuaternion<float>::IDENTITY);
    setLinearMomentum(rkLinMom);
    setAngularMomentum(rkAngMom);

    // world space geocenter
    center_ = GbVec3<float>::ZERO;
    for (int i = 0; i < 8; i++) 
    {
	center_ += worldTransform_ * hullVertices_[i] + rkPos;
    }
    center_ *= invNv;
}
