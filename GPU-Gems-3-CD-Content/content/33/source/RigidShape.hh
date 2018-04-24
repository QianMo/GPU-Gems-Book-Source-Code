/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/

#ifndef  RIGIDSHAPE_HH
#define  RIGIDSHAPE_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GoRigidBody.hh"
#include "GbVec3i.hh"
#include <vector>

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

// this class provides all data for convex objects
// derive your concrete objects from it

class RigidShape
    : public GoRigidBody<float>
{
public:
    RigidShape();
    virtual ~RigidShape();

    // appearance
    void getWorldSpaceVertices (GbVec3<float>* v) const;
    INLINE GbVec3<float> getVertex(unsigned int i) const;
    INLINE GbVec3<float> getWorldSpaceVertex(unsigned int i) const;
    INLINE const GbVec3i<int>* getFaces () const;
    INLINE GbVec3i<int> getTriangle(unsigned int i) const;
    INLINE unsigned int getNumVertices() const;
    INLINE unsigned int getNumFaces() const;

    // simulation
    void getWorldSpaceHullVertices (GbVec3<float>* v) const;
    INLINE GbVec3<float> getHullVertex(unsigned int i) const;
    INLINE GbVec3<float> getWorldSpaceHullVertex(unsigned int i) const;
    INLINE const GbVec3i<int>* getHullFaces () const;
    INLINE GbVec3i<int> getHullTriangle(unsigned int i) const;
    INLINE unsigned int getNumHullVertices() const;
    INLINE unsigned int getNumHullFaces() const;
    INLINE const GbVec3<float>& getCenter() const;
    INLINE float getRadius () const;

    virtual void integrate(float fT, float fDT);

    GbBool moved;

protected:
    std::vector<GbVec3<float> > meshVertices_;
    std::vector<GbVec3i<int> >  meshIndices_;

    std::vector<GbVec3<float> > hullVertices_;
    std::vector<GbVec3i<int> >  hullIndices_;

    GbVec3<float> center_;
    float radius_;
};

#ifndef OUTLINE
#include "RigidShape.in"
#endif  // OUTLINE

#endif  // RIGIDSHAPE_HH
/*----------------------------------------------------------------------
|
| $Log$
|
+---------------------------------------------------------------------*/
