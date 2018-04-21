/**
 @file MD2Model.h

 Quake II MD2 file structure.

 @cite http://tfc.duke.free.fr/us/tutorials/models/md2.htm

 @maintainer Morgan McGuire, matrix@graphics3d.com
 @created 2003-02-21
 @edited  2003-12-16
 */

#ifndef G3D_MD2MODEL_H
#define G3D_MD2MODEL_H

#include "graphics3D.h"
#include "GLG3D/RenderDevice.h"
#include "GLG3D/Texture.h"
#include "GLG3D/PosedModel.h"

namespace G3D {

typedef ReferenceCountedPointer<class MD2Model> MD2ModelRef;

/**
 Quake II model class.
 <P>
 Models are centered about their waist.  To figure out where the feet are you
 might want to look at the bounding box for the stand/walk animations.
 
 <P>
  When available, this class uses SSE instructions for fast vertex blending.
  This cuts the time for getGeometry by a factor of 2.

 <P>This class is not threadsafe; you cannot
 even call methods on two different instances on different threads.

  <P>
 The posed model supplies texture coordinates and normals when rendering
 but the caller must
 bind a texture, set the object to world matrix, and set up lighting.
 <P>
 If VAR has been initialized for the renderDevice, the first time
 this is called (for any Model) a 512k VAR area is allocated.  This
 memory is shared between all MD2 models.
 <P>
 Really huge MD2 models will not fit into this small VAR area and
 will revert to non-VAR rendering.

 <P>
 When getting geometry from the posed model, the normalArray 
 values are interpolated and may have slightly less than unit length.

 <P>
 <IMG SRC="md2000.jpg">
 <DT>
 <IMG SRC="md2001.jpg">
 <DT>
 <IMG SRC="md2002.jpg">
 <DT>
 <IMG SRC="md2003.jpg">
 <DT>

 */
class MD2Model : public ReferenceCountedObject {
public:
    /**
     Amount of time to blend between two animations.
     */
    static const GameTime PRE_BLEND_TIME;

    /**
     These names are (mostly) from Quake II.
     FLIP, SALUTE, FALLBACK, WAVE, and POINT are all taunts.
     A negative number means to run the specified animation backwards.
     The JUMP animation is Quake's jump animation backwards followed
     by the same animation forwards.
     */
    enum Animation {
        JUMP_UP         = -6,
        CROUCH_WALK_BACKWARD = -13,
        RUN_BACKWARD   = -1,
        STAND           = 0,
        RUN             = 1,
        ATTACK          = 2,
        PAIN_A          = 3,
        PAIN_B          = 4,
        PAIN_C          = 5,
        JUMP_DOWN       = 6,
        FLIP            = 7,
        SALUTE          = 8,
        FALLBACK        = 9,
        WAVE            = 10,
        POINT           = 11,
        CROUCH_STAND    = 12,
        CROUCH_WALK     = 13,
        CROUCH_ATTACK   = 14,
        CROUCH_PAIN     = 15,
        CROUCH_DEATH    = 16, 
        DEATH_FALLBACK  = 17,
        DEATH_FALLFORWARD = 18,
        DEATH_FALLBACKSLOW = 19,
        JUMP            = 20,
        MAX_ANIMATIONS  = 21
    };

    class Pose {
    public:

        /**
         When time is negative, this frame is blended into the first
         frame of the animation (which will occur at time 0) over PRE_BLEND_TIME.  This allows disjoint animations to be smoothly connected. 
         MD2Model::choosePose will set time to -PRE_BLEND_TIME and set preFrame. If
         you are manually constructing a pose, MD2Model::getFrameNumber
         will return a value you can use.
         */
        int                 preFrameNumber;

        Animation           animation;

        /**
         Time since the start of the animation. Animations
         loop, so times after the final animation frame time
         are allowed.
         */
        GameTime            time;

        inline Pose() : animation(STAND), time(0) {}
        inline Pose(Animation a, GameTime t = 0) : animation(a), time(t) {}
        
        bool operator==(const Pose& other) const;
        bool operator!=(const Pose& other) const;

        /**
         Given a time and state flags indicating a character's desires,
         computes the new pose.
         <P>
         This may not be ideal for all applications; it is provided as a 
         helper function.
         <P>
         If any death is triggered while crouching, the crouch death will be
         played instead.
         <P>
         Game logic should generally not use the JUMP animation, or
         the jump parameter to choosePose that triggers it.  Instead, play
         the JUMP_UP animation when the character leaves the ground and
         the JUMP_DOWN animation when they hit it again.
         */
         void doSimulation(
            GameTime deltaTime,
            bool crouching,
            bool movingForward,
            bool movingBackward,
            bool attack,
            bool jump,
            bool flip,
            bool salute,
            bool fallback,
            bool wave,
            bool point,
            bool death1,
            bool death2,
            bool death3,
            bool pain1,
            bool pain2,
            bool pain3);
    };


protected:

    class PosedMD2Model : public PosedModel {
    public:
        MD2ModelRef             model;
        CoordinateFrame         cframe;
        Pose                    pose;

        /** Filled out the first time objectSpaceGeometry is called */
        MeshAlg::Geometry       geometry;

        PosedMD2Model(MD2ModelRef _model, const CoordinateFrame& _cframe, const Pose& _pose);
        virtual ~PosedMD2Model() {}
        virtual std::string name() const;
        virtual void getCoordinateFrame(CoordinateFrame&) const;
        virtual const MeshAlg::Geometry& objectSpaceGeometry() const;
        virtual const Array<MeshAlg::Face>& faces() const;
        virtual const Array<MeshAlg::Edge>& edges() const;
        virtual const Array< Array<int> >& adjacentFaces() const;
        virtual const Array<int>& triangleIndices() const;
        virtual void getObjectSpaceBoundingSphere(Sphere&) const;
        virtual void getObjectSpaceBoundingBox(Box&) const;
        virtual void render(RenderDevice* renderDevice) const;
        virtual int numBrokenEdges() const; 
    };

    friend class PosedMD2Model;

    class PackedGeometry {
    public:        
        Array<Vector3>          vertexArray;

        /** Indices into the normalTable. */
        Array<uint8>            normalArray;
    };

    /**
     Quake uses a set of canonical normal vectors.
     */
    static Vector3              normalTable[162];

    /** Loads data into the normalTable. */
    static void setNormalTable();

    class MD2AnimInfo {
    public:
        int     first;
        int     last;
        int     fps;
        bool    loops;
    };

    /**
     Information relating Animations to keyFrames.  Used by computeFrameNumbers().
     */
    static const MD2AnimInfo    animationTable[MAX_ANIMATIONS];

    /**
     Computes the previous and next frame indices and how far we are between them.
     */
    static void computeFrameNumbers(const MD2Model::Pose& pose, int& kf0, int& kf1, double& alpha);

    /** How long we hold in the air as a fraction of jump time. */
    static const double         hangTimePct;

    /** The pose currently stored in interpolatedFrame.  When the animation is MAX_ANIMATIONS
        interpolatedFrame is not yet initialized. */
    static MD2Model*            interpolatedModel;
    static Pose                 interpolatedPose;
    static MeshAlg::Geometry    interpolatedFrame;

    enum {NUM_VAR_AREAS = 10, NONE_ALLOCATED = -1};

    /** Shared dynamic vertex arrays. Allocated by allocateVertexArrays.
        We cycle through multiple VARAreas because the models are so small
        that we can send data to the card faster than it can be rendered
        and we end up spending all of our time waiting on the GPU.*/
    static VARAreaRef            varArea[NUM_VAR_AREAS];

    /**
     When NONE_ALLOCATED, the vertex arrays have not been allocated. 
    */
    static int                  nextVarArea;

    Array<std::string>          _textureFilenames;

    /**
     One RenderDevice primitive
     */
    class Primitive {
    public:
        /** RenderDevice::TRIANGLE_STRIP or RenderDevice::TRIANGLE_FAN */
        RenderDevice::Primitive type;

        class PVertex {
        public:
            /** Indices into a MeshAlg::Geometry's vertexArray */
            int                 index;

            /** One texture coordinate for each index */
            Vector2             texCoord;
        };

        Array<PVertex>          pvertexArray;
    };

    Array<PackedGeometry>       keyFrame;

    Array<Primitive>            primitiveArray;

    /**
     Texture array that parallels vertex and normal arrays.
     Set up by computeTexCoords
     */
    Array<Vector2int16>         _texCoordArray;

    /**
     Matrix mapping int16 texture coordinates to floats.
     */
    CoordinateFrame             texFrame;

    Sphere                      animationBoundingSphere[MAX_ANIMATIONS]; 
    Box                         animationBoundingBox[MAX_ANIMATIONS]; 

    void loadTextureFilenames(BinaryInput& b, int num, int offset);
    
    /**
     Creates a texCoordArray to parallel the vertex and normal arrays,
     duplicating vertices in the keyframes as needed. Called from load().
     @param inCoords Texture coords corresponding to the index array
     */
    void computeTexCoords(const Array<Vector2int16>& inCoords);

    /**
      Called from render() to create the vertex arrays.  Assumes VAR is
      available and the arrays are not initialized.
     */
    void allocateVertexArrays(RenderDevice* renderDevice);

    /**
     Triangle list array useful for generating all of the triangles,
     e.g. for collision detection.  Not used for rendering.
     */
    Array<int>                  indexArray;

    /** Called from create */
    MD2Model() {}

    /** Called from create */
    void load(const std::string& filename);

    /**
     Wipe all data structures.  Called from load.
     */
    virtual void reset();


    /**
     Called from PosedMD2Model::render
     */
    void render(RenderDevice* renderDevice, const Pose& pose);

    /**
     Fills the geometry out from the pose. 

     Called from PosedMD2Model::getGeometry
     */
    void getGeometry(const Pose& pose, MeshAlg::Geometry& geometry) const;

    Array<Vector3>              faceNormalArray;
    Array<MeshAlg::Face>        faceArray;
    Array< Array<int> >         adjacentFaceArray;
    Array<MeshAlg::Edge>        edgeArray;
    Sphere                      boundingSphere;
    Box                         boundingBox;
    int                         numBrokenEdges;
    std::string                 _name;

public:

    std::string name() const {
        return _name;
    }

    virtual void setName(const std::string& n) {
        _name = n;
    }

    /**
     @filename The tris.md2 file
     */
    static MD2ModelRef create(const std::string& filename);

    virtual ~MD2Model() {}

    PosedModelRef pose(const CoordinateFrame& cframe, const Pose& pose);

    inline const Array<Vector2int16>& texCoordArray() const {
        return _texCoordArray;
    }

    const Array<MeshAlg::Face>& faces() const;

    /**
     Edges built by assuming colocated vertices are identical.

     Assumes that vertices colocated in the 0 frame of the STAND animation 
     are colocated for all frames.

     When a degenerate polygon lies along an edge, that edge is present
     in this array once, not three times.
     */
    const Array<MeshAlg::Edge>& geometricEdges() const;

    /**
     adjacentFaces()[v] is an array of indices of faces that touch vertex v.
     Note that two vertices may be colocated but have different indices.
     */
    const Array< Array<int> >& adjacentFaces() const;

    /**
     Render the wireframe mesh.
     */
    void debugRenderWireframe(RenderDevice* renderDevice, const Pose& pose);

    /**
     A bounding sphere on the model.  Covers all vertices in all animations.
     */
    const Sphere& objectSpaceBoundingSphere() const;

    /**
     An oriented bounding box on the model.  Covers all vertices in all animations.
     */
    const Box& objectSpaceBoundingBox() const;

    /**
     Returns the total time of the animation.  If the animation loops (e.g. walking)
     this is the time from the first frame until that frame repeats.  If the
     animation does not loop (e.g. death) this is the time from the first frame
     until the last frame.
     */
    static GameTime animationLength(Animation a);

    /**
     Returns true for standing, running, crouching, and crouch walking animations.
     */
    static bool animationLoops(Animation a);

    /**
     Returns true for the crouching set of animations.
     */
    static bool animationCrouch(Animation a);

    /**
     Returns true for the death animations
     */
    static bool animationDeath(Animation a);

    static bool animationAttack(Animation a);

    static bool animationJump(Animation a);

    static bool animationPain(Animation A);

    /**
     STAND or CROUCH_STAND.
     */
    static bool animationStand(Animation a);

    /**
     running, forward or backward, standing or crouching
     */
    static bool animationRun(Animation a);
    static bool animationRunForward(Animation a);
    static bool animationRunBackward(Animation a);

    /**
     True for actions that can be interrupted, like running or saluting.
     Jumping (which is really more of a falling animation) is considered
     interruptible.
     */
    static bool animationInterruptible(Animation a);

    /** Filenames of textures this model can use. */
    inline const Array<std::string>& textureFilenames() const {
        return _textureFilenames;
    }

    /**
     Returns a value for MD2Model::Pose::preFrameNumber that will
     smoothly blend from this animation to the next one.
     */
    static int getFrameNumber(const Pose& pose);

    /**
     Returns the approximate amount of main memory, not counting the texture,
     occupied by this data structure.
     */
    virtual size_t mainMemorySize() const;
};

}

#endif
