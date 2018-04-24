/*! \file ScatteringViewer.h
 *  \author Yuntao Jia, refers to ScatteringViewer class by Jared Hoberock
 *  \brief Defines the interface to a viewer class for the gpuscattering application.
 */

#ifndef SCATTERING_VIEWER_H
#define SCATTERING_VIEWER_H

#pragma warning( disable : 4996)

// undefine min() and max() macros
#define NOMINMAX 1

#include <GL/glew.h>
#include <glutviewer/GlutViewer.h>
#include <mesh/IndexFaceMeshViewer.h>
#include <mesh/IndexFaceMesh.h>
#include "GpuScattering.h"
#include "DiscTree.h"
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>
#include <buffer/Buffer.h>
#include <vector>

class ScatteringViewer
  : public IndexFaceMeshViewer<GlutViewer,KeyEvent>
{
  public:
    typedef IndexFaceMeshViewer<GlutViewer,KeyEvent> Parent;
    typedef IndexFaceMesh<float3, float2, float3> Mesh;

    virtual void init(void);
    bool loadMeshFile(const char *filename);
    virtual void keyPressEvent(KeyEvent *e);
    virtual void draw(void);
    virtual void animate(void);
    virtual void drawMesh(void);
    virtual void drawMeshVBOs(void);
    virtual std::string helpString(void) const;
    virtual void reloadShaders(void);

    enum renderMode{
      _RENDER_NORMAL,			// render mesh
      _RENDER_OCCLUSION,		// render per vertex occlusion
      _RENDER_SCATTERING,		// render per vertex scattering
      _RENDER_MODE_COUNT		// render mode count
    }; // end enum

  protected:
    // This method initializes the vertex buffer objects associated with
    // Parent::mMesh used for rendering.
    void initVBOs(std::vector<DiscTree::Disc>& discs);

    // This method updates the scattering object
    virtual void updateScattering(void);

    // A Disc for each vertex contained within Parent::mMesh
    std::vector<DiscTree::Disc> mDiscsCopy;	

    // A GpuScattering object contains data & functions for computing per-vertex scattering
    GpuScattering mGpuScattering;

    // A DiscTree contains a hierarchical representation of the mesh and its occlusion
    DiscTree mOcclusionTree;	

    renderMode mRenderMode;

    // per-vertex bent normal and accessibility, stored in a vertex array		
    Buffer mVertexBentNormalAndAccessibility;
    // per-vertex scattering, stored in a vertex array	
    Buffer mVertexScattering;
    // vertex coordinates, stored in a vertex array
    Buffer mVertexCoordinates;
    // vertex normals, stored in a vertex array
    Buffer mVertexNormals;    
    // triangle vertex indices, stored in an index array
    Buffer mTriangleIndices;

    float mEpsilon;	

    // a vertex shader to pass the bent normal to the fragment shader
    Shader mVertexShader;

    // a fragment shader to combine diffuse scattering with glossy reflection
    Shader mScatteringPlusGlossyFragmentShader;

    // a program to combine diffuse scattering with glossy reflection
    Program mScatteringPlusGlossyProgram;

    // a fragment shader to visualize ambient occlusion
    Shader mAmbientOcclusionFragmentShader;

    // a program to visualize ambient occlusion
    Program mAmbientOcclusionProgram;

    // light parameters
    float mLightIntensity;
    float3 mLightPosition;
    float mLightTheta;

    // glossy reflection scale
    float mScaleGlossyReflection;
}; // end ScatteringViewer

#endif // SCATTERING_VIEWER_H

