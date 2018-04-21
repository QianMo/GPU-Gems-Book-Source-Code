/**
  @file Milestone.h

  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2003-08-09
  @edited  2003-08-09
*/

#ifndef GLG3D_MILESTONE_H
#define GLG3D_MILESTONE_H

#include "graphics3D.h"

namespace G3D {

typedef ReferenceCountedPointer<class Milestone> MilestoneRef;

/**
 Used by RenderDevice to force the CPU to wait for the
 GPU to complete for series of commands.  Create using
 RenderDevice::createMilestone.  These are equivalent to
 NVIDIA fences.  On ATI cards the semantics are identical
 but performanc is lower.
 */
class Milestone : public ReferenceCountedObject {
private:

    typedef unsigned int GLFence;

    friend class RenderDevice;

    /**
     Pooled storage of free fences to make allocation fast.
     */
    static Array<GLFence>   factory;

    GLFence                 glfence;
    std::string             _name;

    /**
     True when this milestone has been set by RenderDevice.
     */
    bool                    isSet;

    
    // The methods are private because in the future
    // we may want to move parts of the implementation
    // into RenderDevice.

    Milestone(const std::string& n);

    /** Set the milestone. */
    void set();

    /** Wait for it to be reached. */
    void wait();

public:
    ~Milestone();

    inline const std::string& name() const {
        return _name;
    }
};


}

#endif
