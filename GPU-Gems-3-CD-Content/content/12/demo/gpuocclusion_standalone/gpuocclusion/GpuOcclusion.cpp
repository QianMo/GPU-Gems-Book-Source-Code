/*! \file GpuOcclusion.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of GpuOcclusion class. 
 */

#include "GpuOcclusion.h"

void GpuOcclusion
  ::init(const OcclusionTree &tree)
{
  mFramebuffer.create();
  initTextures(tree);
  reloadShaders();
} // end GpuOcclusion::init()

static float2 computeRectangularTexCoord(const unsigned int i,
                                         const unsigned int w,
                                         const unsigned int h)
{
  // handle special case
  if(i == OcclusionTree::NULL_NODE)
  {
    return float2(0,0);
  }

  return float2(static_cast<float>(i % w) + 0.5f,
                static_cast<float>(i / w) + 0.5f);
} // end computeRectanglularTexCoord()

void GpuOcclusion
  ::initTextures(const OcclusionTree &tree)
{
  unsigned int n = static_cast<unsigned int>(tree.mNodes.size());
  unsigned int w = static_cast<unsigned int>(sqrt(static_cast<float>(n))) + 1;
  unsigned int h = w;

  // note the root of the tree
  mTreeRoot = computeRectangularTexCoord(tree.mRootIndex, w, h);
  std::cerr << "GpuOcclusion::initTextures(): tree root: " << mTreeRoot << std::endl;

  OcclusionTree::Node root = tree.mNodes[tree.mRootIndex];

  // fill leftover room with nan
  float nan = std::numeric_limits<float>::quiet_NaN();
  float3 nan3(nan,nan,nan);
  float4 nan4(nan,nan,nan,nan);

  // create a copy of the discs & triangles
  std::vector<float3> centers(w*h, nan3);
  std::vector<float4> normalsAndAreas(w*h, nan4);
  std::vector<float4> pointers(w*h, nan4);
  std::vector<float> occlusion(w*h, 1.0f);
  std::vector<float3> vertices(w*h, nan3);
  std::vector<float3> triangles0(w*h,nan3);
  std::vector<float3> triangles1(w*h,nan3);

  for(unsigned int i = 0;
      i < tree.mVertexPositions.size();
      ++i)
  {
    vertices[i] = tree.mVertexPositions[i];
  } // end for i

  for(unsigned int i = 0;
      i < tree.mTriangles.size();
      ++i)
  {
    float2 v0 = computeRectangularTexCoord(tree.mTriangles[i][0], w, h);
    float2 v1 = computeRectangularTexCoord(tree.mTriangles[i][1], w, h);
    float2 v2 = computeRectangularTexCoord(tree.mTriangles[i][2], w, h);

    // pack 3 2-dimensional pointers into 2 3-channel textures
    triangles0[i] = float3(v0[0], v0[1], v1[0]);
    triangles1[i] = float3(v1[1], v2[0], v2[1]);
  } // end for i

  for(unsigned int i = 0;
      i < tree.mNodes.size();
      ++i)
  {
    centers[i] = tree.mNodes[i].mDisc.mCentroid;

    // tag interior nodes with a negative area
    float area = tree.mNodes[i].mDisc.mArea;
    if(tree.mNodes[i].mLeftChild != OcclusionTree::NULL_NODE)
    {
      area = -area;
    } // end if

    normalsAndAreas[i] = float4(tree.mNodes[i].mDisc.mNormal, area);

    // the xy portion pointer to the node's right brother or uncle
    float2 nextPointer = computeRectangularTexCoord(tree.mNodes[i].mNextNode, w,h);

    // the zw portion points to the node's left child
    float2 leftChildPointer = computeRectangularTexCoord(tree.mNodes[i].mLeftChild,
                                                         w, h);
    // pack the pointers into a float4
    pointers[i] = float4(nextPointer[0], nextPointer[1],
                         leftChildPointer[0], leftChildPointer[1]);

    // init occlusion to 1
    occlusion[i] = 1.0f;
  } // end for i

  // create textures
  mDiscCenters.create();
  mDiscNormalsAndAreas.create();
  mNodePointers.create();
  mDiscOcclusion[0].create();
  mDiscOcclusion[1].create();
  mVertices.create();
  mTriangles[0].create();
  mTriangles[1].create();

  // upload the data
  mDiscCenters.init(GL_RGB32F_ARB, w, h, 0, GL_RGB, &centers[0][0]);
  mDiscNormalsAndAreas.init(GL_RGBA32F_ARB, w, h, 0, GL_RGBA, &normalsAndAreas[0][0]);
  mNodePointers.init(GL_RGBA32F_ARB, w, h, 0, GL_RGBA, &pointers[0][0]);
  mDiscOcclusion[0].init(GL_LUMINANCE32F_ARB, w, h, 0, GL_LUMINANCE, static_cast<float*>(&occlusion[0]));
  mDiscOcclusion[1].init(GL_LUMINANCE32F_ARB, w, h, 0, GL_LUMINANCE, static_cast<float*>(&occlusion[0]));
  mVertices.init(GL_RGB32F_ARB, w, h, 0, GL_RGB, &vertices[0][0]);
  mTriangles[0].init(GL_RGB32F_ARB, w, h, 0, GL_RGB, &triangles0[0][0]);
  mTriangles[1].init(GL_RGB32F_ARB, w, h, 0, GL_RGB, &triangles1[0][0]);

  // ping-pong starts at 0
  mCurrentOcclusionTexture = 0;
} // end GpuOcclusion::initTextures()

void GpuOcclusion
  ::reloadShaders(void)
{
  mDiscOcclusionShader.create(GL_FRAGMENT_SHADER);
  if(!mDiscOcclusionShader.compileFromFile("discOcclusion.glsl"))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem loading discOcclusion.glsl." << std::endl;
    std::string log;
    mDiscOcclusionShader.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mDiscOcclusionProgram.create(0, mDiscOcclusionShader))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem creating mDiscOcclusionProgram." << std::endl;
    std::string log;
    mDiscOcclusionProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mWorldAndNormalPassthrough.createFromFile(GL_VERTEX_SHADER, "vertexWorldAndNormal.glsl"))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem loading vertexWorldAndNormal.glsl." << std::endl;
    std::string log;
    mWorldAndNormalPassthrough.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mFragmentOcclusionShader.createFromFile(GL_FRAGMENT_SHADER, "fragmentOcclusion.glsl"))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem loading fragmentOcclusion.glsl." << std::endl;
    std::string log;
    mFragmentOcclusionShader.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mFragmentOcclusionProgram.create(mWorldAndNormalPassthrough, mFragmentOcclusionShader))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem creating mFragmentOcclusionProgram." << std::endl;
    std::string log;
    mFragmentOcclusionProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mRobustOcclusionShader.createFromFile(GL_FRAGMENT_SHADER, "robustOcclusion.glsl"))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem creating mRobustOcclusionShader." << std::endl;
    std::string log;
    mRobustOcclusionProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mRobustOcclusionProgram.create(mWorldAndNormalPassthrough, mRobustOcclusionShader))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem creating mRobustOcclusionProgram." << std::endl;
    std::string log;
    mRobustOcclusionProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mWeightedMinimumShader.createFromFile(GL_FRAGMENT_SHADER, "weightedMinimum.glsl"))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Problem creating mWeightedMinimumShader." << std::endl;
    std::string log;
    mWeightedMinimumShader.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mWeightedMinimumProgram.create(0, mWeightedMinimumShader))
  {
    std::cerr << "GpuOcclusion::reloadShaders(): Program creating mWeightedMinimumProgram." << std::endl;
    std::string log;
    mWeightedMinimumProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if
} // end GpuOcclusion::reloadShaders()

void GpuOcclusion
  ::setupGLState(void)
{
  // push the appropriate attribs
  glPushAttrib(GL_TRANSFORM_BIT | GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_LIGHTING);

  // push the current matrices
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();

  // now setup the state
  // set matrices to identity
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // setup an orthographic projection
  glOrtho(0, mDiscCenters.getWidth(),
          0, mDiscCenters.getHeight(),
          0, 1);

  // setup viewport
  glViewport(0,0, mDiscCenters.getWidth(), mDiscCenters.getHeight());

  // disable lighting
  glDisable(GL_LIGHTING);
} // end GpuOcclusion::setupGLState()

void GpuOcclusion
  ::restoreGLState(void)
{
  // pop the matrix stacks
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // pop the attribs
  glPopAttrib();
} // end GpuOcclusion::restoreGLState()

void GpuOcclusion
  ::computeOcclusion(const float epsilon, const float distanceAttenuation)
{
  // setup the gl state
  setupGLState();

  // bind framebuffer
  mFramebuffer.bind();

  // attach non-current texture
  unsigned int nonCurrent = (mCurrentOcclusionTexture + 1) % 2;
  mFramebuffer.
    attachTexture(mDiscOcclusion[nonCurrent].getTarget(),
                  GL_COLOR_ATTACHMENT0_EXT,
                  mDiscOcclusion[nonCurrent]);

  // bind program
  bindDiscOcclusionProgram(epsilon, distanceAttenuation);

  // full screen quad
  float w = static_cast<float>(mDiscCenters.getWidth());
  float h = static_cast<float>(mDiscCenters.getHeight());

  glBegin(GL_QUADS);
  glVertex2f(0,0);
  glVertex2f(w, 0);
  glVertex2f(w, h);
  glVertex2f(0, h);
  glEnd();
  glFinish();

  // unbind program
  unbindDiscOcclusionProgram();

  // detach texture
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);

  // unbind framebuffer
  mFramebuffer.unbind();

  // ping-pong
  mCurrentOcclusionTexture = nonCurrent;

  // restore the gl state
  restoreGLState();
} // end GpuOcclusion::computeOcclusion()

void GpuOcclusion
  ::computeWeightedMinimumOcclusion(void)
{
  // setup the gl state
  setupGLState();

  // since we can't read and write to a Texture at the same
  // time, we need to read the non-current Texture into scratch space
  Texture scratch;
  scratch.create();
  scratch.init(mDiscOcclusion[0].getInternalFormat(),
               mDiscOcclusion[0].getWidth(),
               mDiscOcclusion[0].getHeight());

  // bind framebuffer
  mFramebuffer.bind();

  // copy the non-current Texture into scratch
  mFramebuffer.attachTexture(scratch.getTarget(), GL_COLOR_ATTACHMENT0_EXT, scratch);

  //// bind non-current Texture
  unsigned int nonCurrent = (mCurrentOcclusionTexture + 1) % 2;
  glEnable(mDiscOcclusion[nonCurrent].getTarget());
  mDiscOcclusion[nonCurrent].bind();

  // full screen quad
  float w = static_cast<float>(mDiscCenters.getWidth());
  float h = static_cast<float>(mDiscCenters.getHeight());

  // full screen quad
  glBegin(GL_QUADS);
  glColor4f(1,1,1,1);
  glTexCoord2f(0, 0);
  glVertex2f(0,0);
  glTexCoord2f(float(mDiscCenters.getMaxS()), 0);
  glVertex2f(w, 0);
  glTexCoord2f(float(mDiscCenters.getMaxS()), float(mDiscCenters.getMaxT()));
  glVertex2f(w, h);
  glTexCoord2f(0, float(mDiscCenters.getMaxT()));
  glVertex2f(0, h);
  glEnd();

  // unbind the non-current Texture
  mDiscOcclusion[nonCurrent].unbind();
  glDisable(mDiscOcclusion[nonCurrent].getTarget());

  // now take the weighted minimum of scratch and the current occlusion

  // attach the non current Texture
  mFramebuffer.
    attachTexture(mDiscOcclusion[nonCurrent].getTarget(),
                  GL_COLOR_ATTACHMENT0_EXT,
                  mDiscOcclusion[nonCurrent]);

  // bind current texture & scratch
  glActiveTexture(GL_TEXTURE0);
  mDiscOcclusion[mCurrentOcclusionTexture].bind();
  glActiveTexture(GL_TEXTURE1);
  scratch.bind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // bind the program
  mWeightedMinimumProgram.bind();

  // bind parameters
  mWeightedMinimumProgram.setUniform1f("minScale", 0.7f);
  mWeightedMinimumProgram.setUniform1f("maxScale", 0.3f);

  // bind texunits
  mWeightedMinimumProgram.setUniform1i("texture0", 0);
  mWeightedMinimumProgram.setUniform1i("texture1", 1);

  // full screen quad
  glBegin(GL_QUADS);
  glTexCoord2f(0, 0);
  glVertex2f(0,0);
  glTexCoord2f(float(mDiscCenters.getMaxS()), 0);
  glVertex2f(w, 0);
  glTexCoord2f(float(mDiscCenters.getMaxS()), float(mDiscCenters.getMaxT()));
  glVertex2f(w, h);
  glTexCoord2f(0, float(mDiscCenters.getMaxT()));
  glVertex2f(0, h);
  glEnd();

  // unbind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscOcclusion[mCurrentOcclusionTexture].unbind();
  glActiveTexture(GL_TEXTURE1);
  scratch.unbind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // unbind the program
  mWeightedMinimumProgram.unbind();

  // detach texture
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);

  // unbind framebuffer
  mFramebuffer.unbind();

  // ping-pong
  mCurrentOcclusionTexture = nonCurrent;

  // restore the gl state
  restoreGLState();
} // end GpuOcclusion::computeWeightedMinimumOcclusion()

void GpuOcclusion
  ::bind(const float epsilon, const float distanceAttenuation)
{
  // bind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.bind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.bind();
  glActiveTexture(GL_TEXTURE2);
  mDiscOcclusion[mCurrentOcclusionTexture].bind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.bind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // bind the program
  mFragmentOcclusionProgram.bind();

  // bind parameters
  mFragmentOcclusionProgram.setUniform2fv("treeRoot", mTreeRoot);
  mFragmentOcclusionProgram.setUniform1f("epsilon", epsilon);
  mFragmentOcclusionProgram.setUniform1f("distanceAttenuation", distanceAttenuation);

  // bind textures
  mFragmentOcclusionProgram.setUniform1i("discCenters", 0);
  mFragmentOcclusionProgram.setUniform1i("discNormalsAndAreas", 1);
  mFragmentOcclusionProgram.setUniform1i("discOcclusion", 2);
  mFragmentOcclusionProgram.setUniform1i("tree", 3);
} // end GpuOcclusion::bind()

void GpuOcclusion
  ::unbind(void)
{
  // unbind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.unbind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.unbind();
  glActiveTexture(GL_TEXTURE2);
  mDiscOcclusion[mCurrentOcclusionTexture].unbind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.unbind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // unbind the program
  mFragmentOcclusionProgram.unbind();
} // end GpuOcclusion::unbind()

void GpuOcclusion
  ::bindRobust(const float epsilon, const float distanceAttenuation, const float triangleAttenuation)
{
  // bind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.bind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.bind();
  glActiveTexture(GL_TEXTURE2);
  mDiscOcclusion[mCurrentOcclusionTexture].bind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.bind();
  glActiveTexture(GL_TEXTURE4);
  mTriangles[0].bind();
  glActiveTexture(GL_TEXTURE5);
  mTriangles[1].bind();
  glActiveTexture(GL_TEXTURE6);
  mVertices.bind();


  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // bind the program
  mRobustOcclusionProgram.bind();

  // bind parameters
  mRobustOcclusionProgram.setUniform2fv("treeRoot", mTreeRoot);
  mRobustOcclusionProgram.setUniform1f("epsilon", epsilon);
  mRobustOcclusionProgram.setUniform1f("distanceAttenuation", distanceAttenuation);
  mRobustOcclusionProgram.setUniform1f("triangleAttenuation", triangleAttenuation);

  // bind textures
  mRobustOcclusionProgram.setUniform1i("discCenters", 0);
  mRobustOcclusionProgram.setUniform1i("discNormalsAndAreas", 1);
  mRobustOcclusionProgram.setUniform1i("discOcclusion", 2);
  mRobustOcclusionProgram.setUniform1i("tree", 3);
  mRobustOcclusionProgram.setUniform1i("triangles0", 4);
  mRobustOcclusionProgram.setUniform1i("triangles1", 5);
  mRobustOcclusionProgram.setUniform1i("vertices", 6);
} // end GpuOcclusion::bindRobust()

void GpuOcclusion
  ::unbindRobust(void)
{
  // unbind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.unbind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.unbind();
  glActiveTexture(GL_TEXTURE2);
  mDiscOcclusion[mCurrentOcclusionTexture].unbind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.unbind();
  glActiveTexture(GL_TEXTURE4);
  mTriangles[0].unbind();
  glActiveTexture(GL_TEXTURE5);
  mTriangles[1].unbind();
  glActiveTexture(GL_TEXTURE6);
  mVertices.unbind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // unbind the program
  mRobustOcclusionProgram.unbind();
} // end GpuOcclusion::unbindRobust()

void GpuOcclusion
  ::bindDiscOcclusionProgram(const float epsilon, const float distanceAttenuation)
{
  // bind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.bind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.bind();
  glActiveTexture(GL_TEXTURE2);
  mDiscOcclusion[mCurrentOcclusionTexture].bind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.bind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // bind the program
  mDiscOcclusionProgram.bind();

  // bind parameters
  mDiscOcclusionProgram.setUniform2fv("treeRoot", mTreeRoot);
  mDiscOcclusionProgram.setUniform1f("epsilon", epsilon);
  mDiscOcclusionProgram.setUniform1f("distanceAttenuation", distanceAttenuation);

  // bind texunits
  mDiscOcclusionProgram.setUniform1i("discCenters", 0);
  mDiscOcclusionProgram.setUniform1i("discNormalsAndAreas", 1);
  mDiscOcclusionProgram.setUniform1i("discOcclusion", 2);
  mDiscOcclusionProgram.setUniform1i("tree", 3);
} // end GpuOcclusion::bindDiscOcclusionProgram()

void GpuOcclusion
  ::unbindDiscOcclusionProgram(void)
{
  // unbind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.unbind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.unbind();
  glActiveTexture(GL_TEXTURE2);
  mDiscOcclusion[mCurrentOcclusionTexture].unbind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.unbind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // unbind the program
  mDiscOcclusionProgram.unbind();
} // end GpuOcclusion::unbindDiscOcclusionProgram()

