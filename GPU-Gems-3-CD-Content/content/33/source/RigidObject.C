#ifdef USE_RCSID
static const char RCSid_RigidObject[] = "$Id$";
#endif

/*----------------------------------------------------------------------
|
|
| $Log$
|
|
+---------------------------------------------------------------------*/

#include "RigidObject.hh"

// #ifdef OUTLINE
// #include "RigidObject.in"
// #endif 

// this is an 8-sided capped cone

float app_v[18*3] = {
    -2.6319199f, -2.6315002f, 0.0f,
    247.36807f, -2.6315002f, 0.0f,
    174.14478f, 174.1452f, 0.0f,
    -2.6319308f, 247.3685f, 0.0f,
    -179.40863f, 174.14517f, 0.0f,
    -252.63193f, -2.6315222f, 0.0f,
    -179.4086f, -179.40822f, 0.0f,
    -2.631887f, -252.6315f, 0.0f,
    174.14481f, -179.40817f, 0.0f,
    97.36808f, -2.6315002f, 500.0f,
    68.078758f, 68.079178f, 500.0f,
    -2.6319242f, 97.3685f, 500.0f,
    -73.342606f, 68.07917f, 500.0f,
    -102.63192f, -2.6315091f, 500.0f,
    -73.34259f, -73.342186f, 500.0f,
    -2.6319067f, -102.6315f, 500.0f,
    68.078766f, -73.342171f, 500.0f,
    -2.6319199f, -2.6315002f, 500.0f
};

unsigned char app_i[32*3] = {
    0, 2, 1, 
    0, 3, 2, 
    0, 4, 3, 
    0, 5, 4, 
    0, 6, 5, 
    0, 7, 6, 
    0, 8, 7, 
    0, 1, 8, 
    1, 10, 9, 
    1, 2, 10, 
    2, 11, 10, 
    2, 3, 11, 
    3, 12, 11, 
    3, 4, 12, 
    4, 13, 12, 
    4, 5, 13, 
    5, 14, 13, 
    5, 6, 14, 
    6, 15, 14, 
    6, 7, 15, 
    7, 16, 15, 
    7, 8, 16, 
    8, 9, 16, 
    8, 1, 9, 
    17, 9, 10, 
    17, 10, 11, 
    17, 11, 12, 
    17, 12, 13, 
    17, 13, 14, 
    17, 14, 15, 
    17, 15, 16, 
    17, 16, 9
};

float hull_v[16*3] = {
    247.36807f, -2.6315002f, 0.0f,
    174.14478f, 174.1452f, 0.0f,
    -2.6319308f, 247.3685f, 0.0f,
    -179.40863f, 174.14517f, 0.0f,
    -252.63193f, -2.6315222f, 0.0f,
    -179.4086f, -179.40822f, 0.0f,
    -2.631887f, -252.6315f, 0.0f,
    174.14481f, -179.40817f, 0.0f,
    97.36808f, -2.6315002f, 500.0f,
    68.078758f, 68.079178f, 500.0f,
    -2.6319242f, 97.3685f, 500.0f,
    -73.342606f, 68.07917f, 500.0f,
    -102.63192f, -2.6315091f, 500.0f,
    -73.34259f, -73.342186f, 500.0f,
    -2.6319067f, -102.6315f, 500.0f,
    68.078766f, -73.342171f, 500.0f
};

unsigned char hull_i[10*3] = {
    4, 2, 0, 
    0, 9, 8, 
    1, 10, 9, 
    2, 11, 10, 
    3, 12, 11, 
    4, 13, 12, 
    5, 14, 13, 
    6, 15, 14, 
    7, 8, 15, 
    12, 8, 10
};

RigidObject::RigidObject (float fScale, float fMass, const GbVec3<float>& rkPos,
			  const GbVec3<float>& rkLinMom, const GbVec3<float>& rkAngMom)
    : RigidShape()
{
    debugmsg("object scale "<<fScale<<" mass "<<fMass);

    int nV = 18, nT = 32;
    debugmsg("appearance "<<nV<<" vertices "<<nT<<" triangles");

    // appearance geometry
    meshVertices_.resize(nV);
    for (int i=0; i<nV; ++i)
    {
	meshVertices_[i] = GbVec3<float>(app_v[i*3],app_v[i*3+1],app_v[i*3+2]) * fScale;
    }

    // topology
    meshIndices_.resize(nT);
    for (int i=0; i<nT; ++i)
    {
	meshIndices_[i] = GbVec3i<int>(app_i[i*3],app_i[i*3+1],app_i[i*3+2]);
    }

    nV = 16, nT = 10;
    debugmsg("hull "<<nV<<" vertices "<<nT<<" triangles");

    // convex hull geometry
    hullVertices_.resize(nV);
    for (int i=0; i<nV; ++i)
    {
	hullVertices_[i] = GbVec3<float>(hull_v[i*3],hull_v[i*3+1],hull_v[i*3+2]) * fScale;
    }

    // half spaces
    hullIndices_.resize(nT);
    for (int i=0; i<nT; ++i)
    {
	hullIndices_[i] = GbVec3i<int>(hull_i[i*3],hull_i[i*3+1],hull_i[i*3+2]);
    }

    // inertia tensor
    float invNv = 1.0f / float(nV);
    GbMatrix3<float> kInertia;
    for (int i = 0; i < 3; i++) 
    {
        kInertia[i][i] = 0.0f;
        for (int j = 0; j < 3; j++) 
	{
            if (i != j) 
	    {
                kInertia[i][j] = 0.0f;
                for (int k = 0; k < nV; k++) 
		{
                    kInertia[i][i] += invNv * fMass * hullVertices_[k][j] * hullVertices_[k][j];
                    kInertia[i][j] -= invNv * fMass * hullVertices_[k][i] * hullVertices_[k][j];
                }
            }
        }
    }

    // bounding sphere radius
    GbVec3<float> kCentroid(0.0f);
    for (int j = 0; j < nV; j++) 
    {
	kCentroid += hullVertices_[j];
    }
    kCentroid *= invNv;

    radius_ = 0.0f;
    for (int j=0; j<nV; ++j)
    {
        hullVertices_[j] -= kCentroid;
	const float fTemp = hullVertices_[j].getNorm();
        if (fTemp > radius_) 
	{
            radius_ = fTemp;
        }
    }
    debugmsg("broadphase radius "<<radius_);
    for (unsigned int j=0; j<meshVertices_.size(); ++j)
    {
        meshVertices_[j] -= kCentroid;
    }

    // set 
    setMass(fMass);
    setBodyInertia(kInertia);
    setPosition(rkPos);
    setQOrientation(GbQuaternion<float>::IDENTITY);
    setLinearMomentum(rkLinMom);
    setAngularMomentum(rkAngMom);

    // world space geocenter
    center_ = GbVec3<float>::ZERO;
    for (int i = 0; i < nV; i++) 
    {
	center_ += worldTransform_ * hullVertices_[i] + rkPos;
    }
    center_ *= invNv;
}
