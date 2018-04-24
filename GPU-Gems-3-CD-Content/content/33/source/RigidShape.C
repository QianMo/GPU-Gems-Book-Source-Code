#ifdef USE_RCSID
static const char RCSid_RigidShape[] = "$Id$";
#endif

/*----------------------------------------------------------------------
|
|
| $Log$
|
|
+---------------------------------------------------------------------*/

#include "RigidShape.hh"

#ifdef OUTLINE
#include "RigidShape.in"
#endif 

RigidShape::RigidShape ()
    : GoRigidBody<float>()
    , moved(false)
    , meshVertices_()
    , meshIndices_()
    , hullVertices_()
    , hullIndices_()
    , center_(GbVec3<float>::ZERO)
    , radius_(0.0f)
{
}

RigidShape::~RigidShape()
{
}

void 
RigidShape::getWorldSpaceVertices (GbVec3<float>* akVertex) const
{
    // compute the world space vertices
    for (unsigned int i = 0; i < meshVertices_.size(); i++) 
    {
	akVertex[i] = worldTransform_ * meshVertices_[i] + position_;
    }
}

void 
RigidShape::getWorldSpaceHullVertices (GbVec3<float>* akVertex) const
{
    // compute the world space vertices
    for (unsigned int i = 0; i < hullVertices_.size(); i++) 
    {
	akVertex[i] = worldTransform_ * hullVertices_[i] + position_;
    }
}

void
RigidShape::integrate(float fT, float fDT)
{
    GoRigidBody<float>::integrate(fT,fDT);

    center_ = GbVec3<float>::ZERO;

    for (unsigned int i = 0; i < hullVertices_.size(); i++) 
    {
	center_ += worldTransform_ * hullVertices_[i] + position_;
    }

    center_ /= float(hullVertices_.size());
}
