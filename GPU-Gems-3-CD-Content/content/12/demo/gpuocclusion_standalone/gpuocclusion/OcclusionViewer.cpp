/*! \file OcclusionViewer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of OcclusionViewer class.
 */

#include "OcclusionViewer.h"
#include <gpcpu/Vector.h>
#include <mesh/WavefrontObjUtility.h>

void OcclusionViewer
  ::init(void)
{
  Parent::init();

  mDoFragmentOcclusion = false;
  mUseRobustProgram = false;

  mEpsilon = 8.0f;
  mDistanceAttenuation = 0.0f;
  mTriangleAttenuation = 0.5f;

  // load up the chevy
  loadMeshFile("../data/57chevy_normals.obj");

  makeHelpWindow();
} // end OcclusionViewer::init()

void OcclusionViewer
  ::drawMesh(void)
{
  // lay down depth
  glPushAttrib(GL_COLOR_BUFFER_BIT);
  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  Parent::draw();
  glPopAttrib();

  if(mDoFragmentOcclusion)
  {
    if(mUseRobustProgram)
    {
      mGpuOcclusion.bindRobust(mEpsilon, mDistanceAttenuation, mTriangleAttenuation);
    } // end if
    else
    {
      mGpuOcclusion.bind(mEpsilon, mDistanceAttenuation);
    } // end else
  } // end if

  // deferred shade
  glPushAttrib(GL_DEPTH_BUFFER_BIT);
  glDepthMask(GL_FALSE);
  glDepthFunc(GL_LEQUAL);

  mDrawMesh();

  glFinish();

  glPopAttrib();

  if(mDoFragmentOcclusion)
  {
    if(mUseRobustProgram)
    {
      mGpuOcclusion.unbindRobust();
    } // end if
    else
    {
      mGpuOcclusion.unbind();
    } // end else
  } // end if
} // end OcclusionViewer::drawMesh()

void OcclusionViewer
  ::draw(void)
{
  if(mMesh.getFaces().size() == 0) return;

  drawMesh();
} // end OcclusionViewer::draw()

bool OcclusionViewer
  ::loadMeshFile(const char *filename)
{
  if(Parent::loadMeshFile(filename))
  {
    // scale the mesh to fit in a 10 unit sphere
    // we have to do this because apparently the shaders are computing
    // in half precision
    float3 m,M;
    getBoundingBox(mMesh, m, M);

    float3 c = (M + m) / 2.0f;
    float invDiagonalLength = 1.0f / (M - c).length();
    for(Mesh::PositionList::iterator p = mMesh.getPositions().begin();
        p != mMesh.getPositions().end();
        ++p)
    {
      *p = 10.0f * invDiagonalLength * (*p - c);
    } // end for v

    updateDisplayLists();

    // make a copy of mMesh's triangles
    std::vector<uint3> triangles;
    triangles.resize(mMesh.getFaces().size());
    for(unsigned int f = 0;
        f != mMesh.getFaces().size();
        ++f)
    {
      unsigned int i = 0;
      const Mesh::Face &face = mMesh.getFaces()[f];
      for(Mesh::Face::const_iterator v = face.begin();
          v != face.end();
          ++v, ++i)
      {
        triangles[f][i] = v->mPositionIndex;
      } // end for v
    } // end for f

    // build the occlusion tree
    mOcclusionTree.build(mMesh.getPositions(), triangles);

    // init the gpu occlusion object
    mGpuOcclusion.init(mOcclusionTree);

    return true;
  } // end if

  return false;
} // end OcclusionViewer::loadMeshFile()

void OcclusionViewer
  ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 'R':
    {
      if(e->modifiers() & SHIFT_MODIFIER)
      {
        mUseRobustProgram = !mUseRobustProgram;

        if(mUseRobustProgram && mDoFragmentOcclusion)
        {
          displayMessage(std::string("High quality result."));
        } // end if
        else if(mDoFragmentOcclusion)
        {
          displayMessage(std::string("Low quality result."));
        } // end else if
      } // end if
      else
      {
        mGpuOcclusion.reloadShaders();
      } // end else

      updateGL();
      break;
    } // end case R

    case 'F':
    {
      mDoFragmentOcclusion = !mDoFragmentOcclusion;
      updateGL();
      break;
    } // end case F

    case 'G':
    {
      mGpuOcclusion.computeOcclusion(mEpsilon, mDistanceAttenuation);
      updateGL();
      break;
    } // end case G

    case 'M':
    {
      mGpuOcclusion.computeWeightedMinimumOcclusion();
      updateGL();
      break;
    } // end case M

    case '+':
    {
      mEpsilon += 1.0f;
      char buffer[32];
      sprintf(buffer, "%f", mEpsilon);
      displayMessage(std::string("Epsilon: ") + std::string(buffer));
      updateGL();
      break;
    } // end case +

    case '-':
    {
      mEpsilon -= 1.0f;
      mEpsilon = std::max(1.0f, mEpsilon);
      char buffer[32];
      sprintf(buffer, "%f", mEpsilon);
      displayMessage(std::string("Epsilon: ") + std::string(buffer));
      updateGL();
      break;
    } // end case -

    case 'A':
    {
      if(e->modifiers() & SHIFT_MODIFIER)
      {
        mDistanceAttenuation += 0.1f;
      } // end if
      else
      {
        mDistanceAttenuation -= 0.1f;
        mDistanceAttenuation = std::max(0.0f, mDistanceAttenuation);
      } // end else

      char buffer[32];
      sprintf(buffer, "%f", mDistanceAttenuation);
      displayMessage(std::string("Distance attenuation scale: ") + std::string(buffer));
      updateGL();
      break;
    } // end case A

    case 'T':
    {
      if(e->modifiers() & SHIFT_MODIFIER)
      {
        mTriangleAttenuation += 0.1f;
      } // end if
      else
      {
        mTriangleAttenuation -= 0.1f;
        mTriangleAttenuation = std::max(0.01f, mTriangleAttenuation);
      } // end else

      char buffer[32];
      sprintf(buffer, "%f", mTriangleAttenuation);
      displayMessage(std::string("Triangle attenuation power: ") + std::string(buffer));
      updateGL();
      break;
    } // end case T

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end OcclusionViewer::keyPressEvent()

std::string OcclusionViewer
  ::helpString(void) const
{
  std::string help ="\
General usage:\n\
   Press 'o' to load a mesh by typing the path to the file in the \n\
   console. Next, press 'f' to visualize a low-quality per-pixel  \n\
   result.  Press 'g' a few times to compute a few passes of      \n\
   refinement.  If the solution doesn't seem to converge, press   \n\
   'm' a few times to compute a weighted minimum. Press shift-r to\n\
   enable high-quality occlusion to compute sharp contact shadows \n\
   and smooth penumbras.  To increase or decrease approximation   \n\
   error, use '+' and '-'.  Finally, adjust the influence of      \n\
   shadowing elements with A/a, and T/t.                          \n\
                                                                  \n\
Keys:                                                             \n\
   esc   quit                                                     \n\
   h     toggle help                                              \n\
   o     load a Wavefront OBJ mesh                                \n\
   g     computes a pass of GPU occlusion                         \n\
   m     computes a pass of weighted minimum occlusion            \n\
   f     visualizes per-fragment occlusion                        \n\
   R(r)  enables high(low)-quality robust(faster) occlusion       \n\
   +(-)  increase(decrease) epsilon: decreases(increases) error   \n\
   A(a)  increase(decrease) distance attenuation                  \n\
   T(t)  increase(decrease) triangle attenuation                  \n\
                                                                  \n\
Mouse:                                                            \n\
  Left button:    rotate                                          \n\
  Middle button:  zoom                                            \n\
  Right button:   pan                                             \n\
";

  return help;
} // end OcclusionViewer::helpString()

