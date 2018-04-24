/*! \file GpuScattering.cpp
 *  \author Yuntao Jia
 *  \brief Implementation of GpuScattering class. 
 */

#include "GpuScattering.h"

void GpuScattering
  ::init(const DiscTree &tree)
{
  mFramebuffer.create();
  initTextures(tree);
  reloadShaders();
} // end GpuScattering::init()

static float2 computeRectangularTexCoord(const unsigned int i,
                                         const unsigned int w,
                                         const unsigned int h)
{
  // handle special case
  if(i == DiscTree::NULL_NODE)
  {
    return float2(0,0);
  } // end if

  return float2(static_cast<float>(i % w) + 0.5f,
                static_cast<float>(i / w) + 0.5f);
} // end computeRectanglularTexCoord()

void GpuScattering
  ::initTextures(const DiscTree &tree)
{
  unsigned int n = static_cast<unsigned int>(tree.mNodes.size());
  unsigned int w = static_cast<unsigned int>(sqrt(static_cast<float>(n))) + 1;
  unsigned int h = w;

  // note the root of the tree
  mTreeRoot = computeRectangularTexCoord(tree.mRootIndex, w, h);
  
  // fill leftover room with nan
  float nan = std::numeric_limits<float>::quiet_NaN();
  float3 nan3(nan,nan,nan);
  float4 nan4(nan,nan,nan,nan);

  // create a copy of the discs & triangles
  std::vector<float4> centers(w*h, nan3);
  std::vector<float4> normalsAndAreas(w*h, nan4);
  std::vector<float4> pointers(w*h, nan4);
  std::vector<float4> bentNormalsAndAccessibility(w*h, nan4);
  std::vector<float3> scattering(w*h, nan3);

  for(unsigned int i = 0;
      i < tree.mNodes.size();
      ++i)
  {
    // center
    centers[i] = float4(tree.mNodes[i].mDisc.mPosition, tree.mNodes[i].mNextNodeContinue);

    // tag interior nodes with a negative area
    float area = tree.mNodes[i].mDisc.mArea;
    if(tree.mNodes[i].mLeftChild != DiscTree::NULL_NODE)
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

    // init bent normals to zero and accessibility to 1
    bentNormalsAndAccessibility[i] = float4(0,0,0,1);

    // init scattering to 1
    scattering[i] = float3(1,1,1);
  } // end for i

  // create textures
  mDiscCenters.create();
  mDiscNormalsAndAreas.create();
  mNodePointers.create();
  mBentNormalsAndAccessibility[0].create();
  mBentNormalsAndAccessibility[1].create();
  mDiscScattering.create();

  // upload the data
  mDiscCenters.init(GL_FLOAT_RGBA32_NV, w, h, 0, GL_RGBA, static_cast<float *>(centers[0]));
  mDiscNormalsAndAreas.init(GL_FLOAT_RGBA32_NV, w, h, 0, GL_RGBA, static_cast<float *>(normalsAndAreas[0]));
  mNodePointers.init(GL_FLOAT_RGBA16_NV, w, h, 0, GL_RGBA, static_cast<float *>(pointers[0]));
  mBentNormalsAndAccessibility[0].init(GL_FLOAT_RGBA32_NV, w, h, 0, GL_RGBA, static_cast<float*>(bentNormalsAndAccessibility[0]));
  mBentNormalsAndAccessibility[1].init(GL_FLOAT_RGBA32_NV, w, h, 0, GL_RGBA, static_cast<float*>(bentNormalsAndAccessibility[0]));
  mDiscScattering.init(GL_FLOAT_RGB32_NV, w, h, 0, GL_RGB, static_cast<float*>(scattering[0]));

  // ping-pong starts at 0
  mCurrentAccessibilityTexture = 0;
} // end GpuScattering::initTextures()

void GpuScattering
  ::reloadShaders(void)
{
  mDiscAccessibilityShader.create(GL_FRAGMENT_SHADER);
  if(!mDiscAccessibilityShader.compileFromFile("discAccessibility.glsl"))
  {
    std::cerr << "GpuScattering::reloadShaders(): Problem loading discAccessibility.glsl." << std::endl;
    std::string log;
    mDiscAccessibilityShader.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mDiscAccessibilityProgram.create(0, mDiscAccessibilityShader))
  {
    std::cerr << "GpuScattering::reloadShaders(): Problem creating mDiscAccessibilityProgram." << std::endl;
    std::string log;
    mDiscAccessibilityProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mDiscScatteringShader.createFromFile(GL_FRAGMENT_SHADER, "discScattering.glsl"))
  {
    std::cerr << "GpuScattering::reloadShaders(): Problem creating mDiscScatteringShader." << std::endl;
    std::string log;
    mDiscScatteringShader.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if

  if(!mDiscScatteringProgram.create(0, mDiscScatteringShader))
  {
    std::cerr << "GpuScattering::reloadShaders(): Program creating mDiscScatteringProgram." << std::endl;
    std::string log;
    mDiscScatteringProgram.getInfoLog(log);
    std::cerr << log << std::endl;
  } // end if
} // end GpuScattering::reloadShaders()

void GpuScattering
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
} // end GpuScattering::setupGLState()

void GpuScattering
  ::restoreGLState(void)
{
  // pop the matrix stacks
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // pop the attribs
  glPopAttrib();
} // end GpuScattering::restoreGLState()

void GpuScattering
  ::computeAccessibility(const float epsilon)
{
  // setup the gl state
  setupGLState();

  // bind framebuffer
  mFramebuffer.bind();

  // attach non-current texture
  unsigned int nonCurrent = (mCurrentAccessibilityTexture + 1) % 2;
  mFramebuffer.attachTexture(mBentNormalsAndAccessibility[nonCurrent].getTarget(),
                             GL_COLOR_ATTACHMENT0_EXT,
                             mBentNormalsAndAccessibility[nonCurrent]);

  // bind program
  bindDiscAccessibilityProgram(epsilon);

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
  unbindDiscAccessibilityProgram();

  // detach texture
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);

  // unbind framebuffer
  mFramebuffer.unbind();

  // ping-pong
  mCurrentAccessibilityTexture = nonCurrent;

  // restore the gl state
  restoreGLState();
} // end GpuScattering::computeOcclusion()

void GpuScattering
  ::computeScattering(const float epsilon,
                      const float3 &lightPosition)
{
	// setup the gl state
	setupGLState();

	// bind framebuffer
	mFramebuffer.bind();

	// attach scattering texture
	mFramebuffer.attachTexture(mDiscScattering.getTarget(),
                             GL_COLOR_ATTACHMENT0_EXT,
                             mDiscScattering);

	// bind program
	bindDiscScatteringProgram(epsilon, lightPosition);

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
	unbindDiscScatteringProgram();

	// detach texture
	mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);

	// unbind framebuffer
	mFramebuffer.unbind();

	// restore the gl state
	restoreGLState();
} // end GpuScattering::computeScattering()

void GpuScattering
  ::bindDiscScatteringProgram(const float epsilon,
                              const float3 &lightPosition)
{
  // bind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.bind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.bind();
  glActiveTexture(GL_TEXTURE2);
  mBentNormalsAndAccessibility[mCurrentAccessibilityTexture].bind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.bind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // bind the program
  mDiscScatteringProgram.bind();

  // bind parameters
  mDiscScatteringProgram.setUniform2fv("treeRoot", mTreeRoot);
  mDiscScatteringProgram.setUniform3fv("lightPosition", lightPosition);
  mDiscScatteringProgram.setUniform1f("epsilon", epsilon);  
  float3 albedo_p, sig_tr, zr, zv;
  mScatteringParams.GetScatteringParams(albedo_p, sig_tr, zr, zv);
  mDiscScatteringProgram.setUniform3fv("albedo_p", albedo_p);
  mDiscScatteringProgram.setUniform3fv("sig_tr", sig_tr);
  mDiscScatteringProgram.setUniform3fv("zr", zr);
  mDiscScatteringProgram.setUniform3fv("zv", zv);

  // bind textures
  mDiscScatteringProgram.setUniform1i("discCenters", 0);
  mDiscScatteringProgram.setUniform1i("discNormalsAndAreas", 1);
  mDiscScatteringProgram.setUniform1i("discBentNormalsAndAccessibility", 2);
  mDiscScatteringProgram.setUniform1i("tree", 3);
} // end GpuScattering::bindDiscScatteringProgram()

void GpuScattering
  ::unbindDiscScatteringProgram(void)
{
  // unbind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.unbind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.unbind();
  glActiveTexture(GL_TEXTURE2);
  mBentNormalsAndAccessibility[mCurrentAccessibilityTexture].unbind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.unbind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // unbind the program
  mDiscScatteringProgram.unbind();
} // end GpuScattering::unbindDiscScatteringProgram()

void GpuScattering
  ::bindDiscAccessibilityProgram(const float epsilon)
{
  // bind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.bind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.bind();
  glActiveTexture(GL_TEXTURE2);
  mBentNormalsAndAccessibility[mCurrentAccessibilityTexture].bind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.bind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // bind the program
  mDiscAccessibilityProgram.bind();

  // bind parameters
  mDiscAccessibilityProgram.setUniform2fv("treeRoot", mTreeRoot);
  mDiscAccessibilityProgram.setUniform1f("epsilon", epsilon);

  // bind texunits
  mDiscAccessibilityProgram.setUniform1i("discCenters", 0);
  mDiscAccessibilityProgram.setUniform1i("discNormalsAndAreas", 1);
  mDiscAccessibilityProgram.setUniform1i("discBentNormalsAndAccessibility", 2);
  mDiscAccessibilityProgram.setUniform1i("tree", 3);
} // end GpuScattering::bindDiscOcclusionProgram()

void GpuScattering
  ::unbindDiscAccessibilityProgram(void)
{
  // unbind textures
  glActiveTexture(GL_TEXTURE0);
  mDiscCenters.unbind();
  glActiveTexture(GL_TEXTURE1);
  mDiscNormalsAndAreas.unbind();
  glActiveTexture(GL_TEXTURE2);
  mBentNormalsAndAccessibility[mCurrentAccessibilityTexture].unbind();
  glActiveTexture(GL_TEXTURE3);
  mNodePointers.unbind();

  // restore texture state
  glActiveTexture(GL_TEXTURE0);

  // unbind the program
  mDiscAccessibilityProgram.unbind();
} // end GpuScattering::unbindDiscOcclusionProgram()

void GpuScattering
  ::copyAccessibility(Buffer &buffer, const unsigned int n)
{
  unsigned int w = mBentNormalsAndAccessibility[mCurrentAccessibilityTexture].getWidth();
  unsigned int h = mBentNormalsAndAccessibility[mCurrentAccessibilityTexture].getHeight();

  // we may have to do two readbacks
  unsigned int numFullScanlines = n / w;
  unsigned int partialScanline = n % w;
  unsigned int nCurrent = mCurrentAccessibilityTexture;

  // bind the framebuffer and readback
  setupGLState();
  mFramebuffer.bind();
  mFramebuffer.attachTexture(mBentNormalsAndAccessibility[nCurrent].getTarget(),
                             GL_COLOR_ATTACHMENT0_EXT,
                             mBentNormalsAndAccessibility[nCurrent]);

  // bind the buffer
  buffer.setTarget(GL_PIXEL_PACK_BUFFER_ARB);
  buffer.bind();
  glReadPixels(0,0,w,numFullScanlines,GL_RGBA,GL_FLOAT, 0);
 
  if(partialScanline > 0)
  {
    // get a pointer to the first byte to write to
    unsigned int writeHere = numFullScanlines * w * 4 * sizeof(float);

    // get the last partial scanline
    glReadPixels(0,numFullScanlines, partialScanline,1,GL_RGBA, GL_FLOAT, reinterpret_cast<void*>(writeHere));
  } // end if
 
  // unbind the buffer
  buffer.unbind();

  // unbind the framebuffer
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);
  mFramebuffer.unbind();
  restoreGLState();	
} // end GpuScattering::copyOcclusion()

void GpuScattering
  ::copyScattering(Buffer &buffer,const unsigned int n)
{
  unsigned int w = mDiscScattering.getWidth();
  unsigned int h = mDiscScattering.getHeight();

  // we may have to do two readbacks
  unsigned int numFullScanlines = n / w;
  unsigned int partialScanline = n % w;  

  // bind the framebuffer and readback
  setupGLState();
  mFramebuffer.bind();
  mFramebuffer.attachTexture(mDiscScattering.getTarget(), GL_COLOR_ATTACHMENT0_EXT, mDiscScattering);

  // bind the buffer
  buffer.setTarget(GL_PIXEL_PACK_BUFFER_ARB);
  buffer.bind();
  glReadPixels(0,0,w,numFullScanlines,GL_RGB,GL_FLOAT, 0);

  if(partialScanline > 0)
  {
    // get a pointer to the first byte to write to
    unsigned int writeHere = numFullScanlines * w * 3 * sizeof(float);

    // get the last partial scanline
    glReadPixels(0,numFullScanlines,partialScanline,1, GL_RGB, GL_FLOAT, reinterpret_cast<void*>(writeHere));
  } // end if

  // unbind the buffer
  buffer.unbind();

  // unbind the framebuffer
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);
  mFramebuffer.unbind();
  restoreGLState();	
} // end GpuScattering::copyScattering()

void GpuScattering
  ::ChangeScatteringParams(void)
{
  mScatteringParams.Next();
} // end GpuScattering::ChangeScatteringParams()

std::string GpuScattering
  ::GetScatteringParamName(void)
{
  return mScatteringParams.GetScatteringParamName();
} // end GpuScattering::GetScatteringParamName()
