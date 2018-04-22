/**
  @file VARArea.h
  @maintainer Morgan McGuire, morgan@graphics3d.com
  @created 2003-08-09
  @edited  2004-01-06
*/

#ifndef GLG3D_VARAREA_H
#define GLG3D_VARAREA_H

#include "graphics3D.h"
#include "GLG3D/Milestone.h"

namespace G3D {

typedef ReferenceCountedPointer<class VARArea> VARAreaRef;

/**
 Wrapper for OpenGL Vertex Buffer Object
 http://oss.sgi.com/projects/ogl-sample/registry/ARB/vertex_buffer_object.txt
 http://developer.nvidia.com/docs/IO/8230/GDC2003_OGL_BufferObjects.ppt

 Allocate a VARArea, then allocate VARs within it.  VARAreas are garbage
 collected.  When no pointers remain to VARs inside it or the VARArea itself,
 it will automatically be reclaimed by the system.

 You cannot mix pointers from different VARAreas when rendering.  For
 example, if the vertex VAR is in one VARArea, the normal VAR and color
 VAR must come from the same area.
 */
class VARArea : public ReferenceCountedObject {
public:

    /**
     Use WRITE_EVERY_FRAME if you write at least once per frame.

     Correspond to:
      WRITE_ONCE : GL_STATIC_DRAW_ARB
      WRITE_EVERY_FRAME : GL_STREAM_DRAW_ARB
      WRITE_EVERY_FEW_FRAMEs : GL_STATIC_DRAW_ARB
     */
    enum UsageHint {
        WRITE_ONCE,
        WRITE_EVERY_FRAME,
        WRITE_EVERY_FEW_FRAMES};

private:

	friend class VAR;
    friend class RenderDevice;

    /**
     The milestone is used for finish().  It is created
     by RenderDevice::setVARAreaMilestones.  If NULL, there
     is no milestone pending.
     */
    MilestoneRef        milestone;

	/** Number of bytes currently allocated out of size total. */
	size_t				allocated;

	/**
	 This count prevents vertex arrays that have been freed from
	 accidentally being used-- it is incremented every time
     the VARArea is reset.
	 */
	uint64				generation;

	/** The maximum size of this area that was ever used. */
	size_t				peakAllocated;

    RenderDevice*       renderDevice;


	/** Total  number of bytes in this area. */
	size_t				size;

    /**
     The OpenGL buffer object associated with this area
     (only used when mode == VBO_MEMORY)
     */
    uint32              glbuffer;

	/** Pointer to the memory (NULL when
        the VBO extension is not present). */
	void*				basePointer;

    enum Mode {UNINITIALIZED, VBO_MEMORY, MAIN_MEMORY};
    static Mode         mode;

	VARArea(size_t _size, UsageHint h);

public:

    static VARAreaRef create(size_t s , UsageHint h = WRITE_EVERY_FRAME);

    ~VARArea();

	size_t totalSize() const;

	size_t freeSize() const;

	size_t allocatedSize() const;

	size_t peakAllocatedSize() const;

    /**
     Blocks the CPU until all rendering calls referencing 
     this area have completed.
     */
    void finish();

	/** Finishes, then frees all VAR memory inside this area.*/ 
	void reset();
};


} // namespace

inline unsigned int hashCode(const G3D::VARArea* v) {
    return (unsigned int)v;
}

#endif
