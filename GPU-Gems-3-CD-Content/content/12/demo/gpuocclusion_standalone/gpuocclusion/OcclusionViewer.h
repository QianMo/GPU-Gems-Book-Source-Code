/*! \file OcclusionViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a viewer class for the gpuocclusion application.
 */

#ifndef OCCLUSION_VIEWER_H
#define OCCLUSION_VIEWER_H

// undefine min() and max() macros
#define NOMINMAX 1

#include <GL/glew.h>
#include <glutviewer/GlutViewer.h>
#include <mesh/IndexFaceMeshViewer.h>
#include <mesh/IndexFaceMesh.h>
#include "GpuOcclusion.h"
#include "OcclusionTree.h"
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>

class OcclusionViewer
  : public IndexFaceMeshViewer<GlutViewer,KeyEvent>
{
  public:
    typedef IndexFaceMeshViewer<GlutViewer,KeyEvent> Parent;
    typedef IndexFaceMesh<float3, float2, float3> Mesh;

    virtual void init(void);
    bool loadMeshFile(const char *filename);
    virtual void keyPressEvent(KeyEvent *e);
    virtual void draw(void);
    virtual void drawMesh(void);
    virtual std::string helpString(void) const;

  protected:
    GpuOcclusion mGpuOcclusion;
    OcclusionTree mOcclusionTree;

    bool mDoFragmentOcclusion;
    bool mUseRobustProgram;

    float mEpsilon;
    float mDistanceAttenuation;
    float mTriangleAttenuation;
}; // end OcclusionViewer

#endif // OCCLUSION_VIEWER_H

