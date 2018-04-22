/**
  @file PosedModel.h
  
  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2003-11-15
  @edited  2003-12-16
 */ 

#ifndef GLG3D_POSEDMODEL_H
#define GLG3D_POSEDMODEL_H

#include "graphics3D.h"

namespace G3D {

typedef ReferenceCountedPointer<class PosedModel> PosedModelRef;

/**
 Base class for posed models.

 G3D does not provide a scene graph structure because the visible
 surface determination, rendering, and caching behavior of a scene
 graph are different for each application.  There is currently no 
 one-size-fits-all scene graph design pattern for us to implement.
 
 G3D <B>does</B> provide some basic infrastructure for making your
 own scene graph, however.  Model classes (e.g. G3D::IFSModel, G3D::MD2Model),
 which have no common base class, should all implement a <CODE>pose</CODE> 
 method.  The arguments to this method are all of the parameters (e.g. 
 animation frame, limb position) needed for reducing the general purpose,
 poseable model to a specific world space mesh for rendering.  This
 instance specific mesh is a G3D::PosedModel.  Because all PosedModels
 have the same interface, they can be used interchangably.

 Use G3D::PosedModelWrapper to encapsulate an existing posed model
 with your own.
 */
class PosedModel : public ReferenceCountedObject {
protected:

    PosedModel() {}

public:

    virtual ~PosedModel() {}

    virtual std::string name() const = 0;

    /** Object to world space coordinate frame.*/
    virtual void getCoordinateFrame(CoordinateFrame& c) const = 0;

    virtual CoordinateFrame coordinateFrame() const;

    /** Get the <B>object space</B> geometry (faster than getWorldSpaceGeometry). */
    virtual const MeshAlg::Geometry& objectSpaceGeometry() const = 0;

    /** Get the <B>world space</B> geometry. */
    virtual void getWorldSpaceGeometry(MeshAlg::Geometry& geometry) const;

    virtual void getObjectSpaceFaceNormals(Array<Vector3>& faceNormals, bool normalize = true) const;

    virtual void getWorldSpaceFaceNormals(Array<Vector3>& faceNormals, bool normalize = true) const;

    // Returns a reference rather than filling out an array because most
    // PosedModels have this information available.
    virtual const Array<MeshAlg::Face>& faces() const = 0;

    virtual const Array<MeshAlg::Edge>& edges() const = 0;

    virtual const Array< Array<int> >& adjacentFaces() const = 0;

    virtual const Array<int>& triangleIndices() const = 0;

    virtual void getObjectSpaceBoundingSphere(Sphere&) const = 0;

    virtual Sphere objectSpaceBoundingSphere() const;

    virtual void getWorldSpaceBoundingSphere(Sphere& s) const;

    virtual Sphere worldSpaceBoundingSphere() const;

    virtual void getObjectSpaceBoundingBox(Box&) const = 0;

    virtual Box objectSpaceBoundingBox() const;

    virtual void getWorldSpaceBoundingBox(Box& box) const;

    virtual Box worldSpaceBoundingBox() const;

    virtual void render(class RenderDevice* renderDevice) const = 0;
    /**
     Number of edges that have only one adjacent face.
     */
    virtual int numBrokenEdges() const = 0;
};



/**
 A wrapper that trampolines every method to an internal model.
 Use this as a base class when you have a PosedModelRef and 
 want to override its render() [or some other] method.
 */
class PosedModelWrapper : public PosedModel {
protected:

    PosedModelRef model;

    PosedModelWrapper(PosedModelRef _model) : model(_model) {}

public:

    virtual ~PosedModelWrapper() {}

    virtual std::string name() const;

    virtual void getCoordinateFrame(CoordinateFrame& c) const;

    virtual CoordinateFrame coordinateFrame() const;

    virtual const MeshAlg::Geometry& objectSpaceGeometry() const;

    virtual void getWorldSpaceGeometry(MeshAlg::Geometry& geometry) const;

    virtual const Array<MeshAlg::Face>& faces() const;

    virtual const Array<MeshAlg::Edge>& edges() const;

    virtual void getObjectSpaceFaceNormals(Array<Vector3>& faceNormals, bool normalize = true) const;

    virtual void getWorldSpaceFaceNormals(Array<Vector3>& faceNormals, bool normalize = true) const;

    virtual const Array< Array<int> >& adjacentFaces() const;

    virtual const Array<int>& triangleIndices() const;

    virtual void getObjectSpaceBoundingSphere(Sphere&) const;

    virtual Sphere objectSpaceBoundingSphere() const;

    virtual void getWorldSpaceBoundingSphere(Sphere& s) const;

    virtual Sphere worldSpaceBoundingSphere() const;

    virtual void getObjectSpaceBoundingBox(Box&) const;

    virtual Box objectSpaceBoundingBox() const;

    virtual void getWorldSpaceBoundingBox(Box& box) const;

    virtual Box worldSpaceBoundingBox() const;

    virtual void render(class RenderDevice* renderDevice) const;

    virtual int numBrokenEdges() const;
};


}

#endif
