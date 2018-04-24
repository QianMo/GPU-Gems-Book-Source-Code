#ifdef USE_RCSID
static const char RCSid_RigidTetrahedron[] = "$Id$";
#endif

/*----------------------------------------------------------------------
|
|
| $Log$
|
|
+---------------------------------------------------------------------*/

#include "RigidTetrahedron.hh"

// #ifdef OUTLINE
// #include "RigidTetrahedron.in"
// #endif 

RigidTetrahedron::RigidTetrahedron (float fSize, float fMass, const GbVec3<float>& rkPos,
				    const GbVec3<float>& rkLinMom, const GbVec3<float>& rkAngMom)
    : RigidShape()
{
    debugmsg("tet scale "<<fSize<<" mass "<<fMass);

    // geometry
    meshVertices_.resize(4);
    meshVertices_[0] = -(fSize/3.0f)*GbVec3<float>(1.0f,1.0f,1.0f);
    meshVertices_[1] = GbVec3<float>(+fSize,0.0f,0.0f);
    meshVertices_[2] = GbVec3<float>(0.0f,+fSize,0.0f);
    meshVertices_[3] = GbVec3<float>(0.0f,0.0f,+fSize);

    // topology
    meshIndices_.resize(4);
    meshIndices_[0] = GbVec3i<int>(0, 2, 1);
    meshIndices_[1] = GbVec3i<int>(0, 3, 2);
    meshIndices_[2] = GbVec3i<int>(0, 1, 3);
    meshIndices_[3] = GbVec3i<int>(1, 2, 3);

    // convex hull identical to appearance
    hullVertices_.assign(meshVertices_.begin(),meshVertices_.end());
    hullIndices_.assign(meshIndices_.begin(),meshIndices_.end());

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
                for (int k = 0; k < 4; k++) 
		{
                    kInertia[i][i] += 0.25f * fMass * hullVertices_[k][j] * hullVertices_[k][j];
                    kInertia[i][j] -= 0.25f * fMass * hullVertices_[k][i] * hullVertices_[k][j];
                }
            }
        }
    }

    // bounding sphere radius
    const GbVec3<float> kCentroid = (fSize/6.0f)*GbVec3<float>(1.0f,1.0f,1.0f);
    radius_ = 0.0f;
    for (int j = 0; j < 4; j++) 
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
    for (int i = 0; i < 4; i++) 
    {
	center_ += worldTransform_ * hullVertices_[i] + rkPos;
    }
    center_ *= 0.25f;
}
