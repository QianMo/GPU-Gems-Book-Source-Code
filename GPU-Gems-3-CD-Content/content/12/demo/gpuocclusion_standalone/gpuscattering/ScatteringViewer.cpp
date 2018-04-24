/*! \file ScatteringViewer.cpp
 *  \author Yuntao Jia
 *  \brief Implementation of ScatteringViewer class.
 */

#pragma warning( disable : 4996)
#include "ScatteringViewer.h"
#include <gpcpu/Vector.h>
#include <mesh/WavefrontObjUtility.h>


void ScatteringViewer
  ::init(void)
{
  Parent::init();

  mRenderMode = _RENDER_SCATTERING;  
  mEpsilon = 8.0f;
  reloadShaders();

  mLightIntensity = 5.0f;
  mLightPosition = float3(0,20,0);
  mLightTheta = 0;

  mScaleGlossyReflection = 1.0f;

  // load up yeahright
  loadMeshFile("../data/yeahright30000.obj");

  makeHelpWindow();
} // end ScatteringViewer::init()

void ScatteringViewer
  ::reloadShaders(void)
{
  Parent::reloadShaders();

  std::string source = "\n\
    uniform vec3 eye;\n\
    uniform vec3 light;\n\
    void main(void)\n\
    {\n\
      gl_FrontColor = gl_Color;\n\
      gl_TexCoord[0] = gl_MultiTexCoord0;\n\
      gl_TexCoord[1].xyz = eye - gl_Vertex.xyz;\n\
      gl_TexCoord[2].xyz = light - gl_Vertex.xyz;\n\
      gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;\n\
    }";
  mVertexShader.create(GL_VERTEX_SHADER, source.c_str());

  source = "\n\
    uniform float lightIntensity;\n\
    uniform float scaleGlossy;\n\
    void main(void)\n\
    {\n\
      vec3 n = normalize(gl_TexCoord[0].xyz);\n\
      vec3 v = normalize(gl_TexCoord[1].xyz);\n\
      vec3 l = normalize(gl_TexCoord[2].xyz);\n\
      float accessibility = gl_TexCoord[0].w;\n\
      float ndotl = saturate(dot(n,l));\n\
      gl_FragColor = lightIntensity * gl_Color;\n\
      if(ndotl > 0.0)\n\
      {\n\
        vec3 r = 2.0*ndotl*n - l;\n\
        float glossy = pow(saturate(dot(r, v)),16.0);\n\
  	    gl_FragColor += scaleGlossy * accessibility * glossy;\n\
      }\n\
      gl_FragColor.a = 1.0;\n\
    }";
  mScatteringPlusGlossyFragmentShader.create(GL_FRAGMENT_SHADER, source.c_str());

  if(!mScatteringPlusGlossyProgram.create(mVertexShader, mScatteringPlusGlossyFragmentShader))
  {
    std::cerr << "ScatteringViewer::reloadShaders(): Problem creating mScatteringPlusGlossyProgram." << std::endl;
    std::cerr << mScatteringPlusGlossyProgram << std::endl;
  } // end if

  source = "\n\
    uniform float lightIntensity;\n\
    void main(void)\n\
    {\n\
      vec3 n = normalize(gl_TexCoord[0].xyz);\n\
      vec3 l = normalize(gl_TexCoord[2].xyz);\n\
      float occlusion = gl_TexCoord[0].w;\n\
      \/\/vec3 occlusion = n;\n\
      float ndotl = saturate(dot(n,l));\n\
      \/\/gl_FragColor.rgb = abs(occlusion.xyz);\n\
      gl_FragColor.rgb = occlusion;\n\
      gl_FragColor.a = 1.0;\n\
    }";
  mAmbientOcclusionFragmentShader.create(GL_FRAGMENT_SHADER, source.c_str());

  if(!mAmbientOcclusionProgram.create(mVertexShader, mAmbientOcclusionFragmentShader))
  {
    std::cerr << "ScatteringViewer::reloadShaders(): Problem creating mAmbientOcclusionProgram." << std::endl;
    std::cerr << mAmbientOcclusionProgram << std::endl;
  } // end if
} // end ScatteringViewer::reloadShaders()

void ScatteringViewer
  ::drawMesh(void)
{
  if(mRenderMode == _RENDER_NORMAL)
  {
    Parent::draw();
  } // end if
  else
  {
    // choose which program we need
    Program &p = (mRenderMode == _RENDER_OCCLUSION) ? mAmbientOcclusionProgram : mScatteringPlusGlossyProgram;

	  // drawing
 	  p.bind();
 	  p.setUniform3fv("eye", viewPosition());	 
 	  p.setUniform3fv("light", mLightPosition);
    p.setUniform1f("lightIntensity", mLightIntensity);
    p.setUniform1f("scaleGlossy", mScaleGlossyReflection);
    drawMeshVBOs();
	  p.unbind();
  } // end else
} // end ScatteringViewer::drawMesh()

void ScatteringViewer
  ::draw(void)
{
  if(mMesh.getFaces().size() == 0) return;

  // draw the light position
  glPushAttrib(GL_POINT_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT);
  glColor3f(1,1,1);
  glDisable(GL_LIGHTING);
  glPointSize(6.0f);
  glBegin(GL_POINTS);
  glVertex3fv(mLightPosition);
  glEnd();
  glPopAttrib();

  drawMesh();
} // end ScatteringViewer::draw()

void ScatteringViewer
  ::updateScattering(void)
{
  mGpuScattering.computeScattering(mEpsilon, mLightPosition);
  mGpuScattering.copyScattering(mVertexScattering,(unsigned int)mMesh.getPositions().size());
} // end ScatteringViewer::updateScattering()

void ScatteringViewer
  ::animate(void)
{
  // update light position
  mLightTheta += PI/45;
  mLightPosition[0] = 20.0f * sin(mLightTheta);
  mLightPosition[2] = 20.0f * cos(mLightTheta);

  if(mMesh.getFaces().size() == 0) return;

  // update scattering
  updateScattering();
} // end ScatteringViewer::idle()

void ScatteringViewer
  ::drawMeshVBOs(void)
{
  // bind vbo
  glEnableClientState(GL_VERTEX_ARRAY);
  mVertexCoordinates.setTarget(GL_ARRAY_BUFFER);
  mVertexCoordinates.bind();
  glVertexPointer(3, GL_FLOAT, 0, 0);
  mVertexCoordinates.unbind();

  glEnableClientState(GL_NORMAL_ARRAY);
  mVertexNormals.setTarget(GL_ARRAY_BUFFER);
  mVertexNormals.bind();
  glNormalPointer(GL_FLOAT, 0, 0);
  mVertexNormals.unbind();

  // bind scattering to color
  glEnableClientState(GL_COLOR_ARRAY);
  mVertexScattering.setTarget(GL_ARRAY_BUFFER);
  mVertexScattering.bind();
  glColorPointer(3, GL_FLOAT, 0, 0);
  mVertexScattering.unbind();

  // bind normal and occlusion to texture coord 0
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  mVertexBentNormalAndAccessibility.setTarget(GL_ARRAY_BUFFER);
  mVertexBentNormalAndAccessibility.bind();
  glTexCoordPointer(4, GL_FLOAT, 0, 0);
  mVertexBentNormalAndAccessibility.unbind();

  unsigned int numElements = (unsigned int)(3 * mMesh.getFaces().size());
  mTriangleIndices.setTarget(GL_ELEMENT_ARRAY_BUFFER_ARB);
  mTriangleIndices.bind();

  glPushAttrib(GL_POLYGON_BIT);
  glPolygonMode(GL_FRONT_AND_BACK, mPolygonMode);

  glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_INT, 0);

  glPopAttrib();

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
} // end ScatteringViewer::drawMeshVBOs()

bool ScatteringViewer
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

    // init VBOs	
    initVBOs(mDiscsCopy);
    
    // build the hierarchy
    mOcclusionTree.build(mDiscsCopy);

    // init the gpu occlusion object
    mGpuScattering.init(mOcclusionTree);

	  // compute occlusion: one pass is enough for this application
	  mGpuScattering.computeAccessibility(mEpsilon);
	  mGpuScattering.copyAccessibility(mVertexBentNormalAndAccessibility,
                                     (unsigned int)mMesh.getPositions().size());

    // update scattering
    updateScattering();
    return true;
  } // end if

  return false;
} // end ScatteringViewer::loadMeshFile()

void ScatteringViewer
  ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 'U':
    {
      mRenderMode = _RENDER_SCATTERING;
      displayMessage("Multiple scattering");
      updateGL();
      break;
    } // end case U

    case 'P':
    {	  
      if(mRenderMode == _RENDER_SCATTERING)
      {
        mGpuScattering.ChangeScatteringParams();
        updateScattering();
        displayMessage("Change scattering material to " + mGpuScattering.GetScatteringParamName());
        updateGL();
      } // end if	  	  
      break;
    } // end case P

    case 'G':
    {
      mRenderMode = _RENDER_OCCLUSION;
      displayMessage("Ambient occlusion");
      updateGL();
      break;
    } // end case G

    case 'M':
    {
      mRenderMode = _RENDER_NORMAL;		
      updateGL();
      break;
    } // end case M

    case '+':
    {
      mScaleGlossyReflection *= 1.1f;
      char buffer[32];
      sprintf(buffer, "%f", mScaleGlossyReflection);
      displayMessage(std::string("Glossy reflection scale: ") + std::string(buffer));
      updateGL();
      break;
    } // end case +

    case '-':
    {
      mScaleGlossyReflection *= 0.9f;
      char buffer[32];
      sprintf(buffer, "%f", mScaleGlossyReflection);
      displayMessage(std::string("Glossy reflection scale: ") + std::string(buffer));
      updateGL();
      break;
    } // end case -

    case '[':
    {
      mLightIntensity *= 0.9f;
      char buffer[32];
      sprintf(buffer, "%f", mLightIntensity);		
      displayMessage(std::string("Light intensity: ") + std::string(buffer));
      updateGL();
      break;
    } // end case [

    case ']':
    {
      mLightIntensity *= 1.1f;
      char buffer[32];
      sprintf(buffer, "%f", mLightIntensity);		
      displayMessage(std::string("Light intensity: ") + std::string(buffer));
      updateGL();
      break;
    } // end case ]

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end ScatteringViewer::keyPressEvent()

std::string ScatteringViewer
  ::helpString(void) const
{
  std::string help ="\
General usage:\n\
   Press 'o' to load a mesh by typing the path to the file in the \n\
   console. Press 'p' to change the scattering materials.         \n\
   To increase or decrease light intensity, use '[' and ']'.      \n\
   Press 'enter' to toggle light spinning.                        \n\
                                                                  \n\
Keys:                                                             \n\
   esc   quit                                                     \n\
   h     toggle help                                              \n\
   o     load a Wavefront OBJ mesh                                \n\
   u     visualizes per-vertex scattering.                        \n\
   g     visualizes per-vertex ambient occlusion                  \n\
   enter toggle light animation.                                  \n\
   p     change the scattering materials                          \n\
   [(])  decrease(increase) light intensity.                      \n\
   -(+)  decrease(increase) glossy reflection.                    \n\
                                                                  \n\
Mouse:                                                            \n\
  Left button:    rotate                                          \n\
  Middle button:  zoom                                            \n\
  Right button:   pan                                             \n\
";

  return help;
} // end ScatteringViewer::helpString()

void ScatteringViewer
  ::initVBOs(std::vector<DiscTree::Disc>& discs)
{
  Mesh::FaceList vFaceList = mMesh.getFaces();
  Mesh::PositionList vPosList  = mMesh.getPositions(); 	
  
  int nVertices = (int)vPosList.size();
  int nFaces = (int)vFaceList.size();
  
  std::vector<float> fAreaList;
  std::vector<float3> vVtxNormalList;
  std::vector<int>    vVtxNormalSize;
  fAreaList.resize(nVertices);
  vVtxNormalList.resize(nVertices);
  vVtxNormalSize.resize(nVertices);
  
  int i,j;
  for(i=0; i<nVertices; ++i)
  {
    fAreaList[i] = vVtxNormalSize[i] = 0;
    vVtxNormalList[i] = float3(0,0,0);
  } // end for i
  
  for(i=0; i<nFaces; ++i)
  {
    Mesh::Face face = vFaceList[i];
    // triangulated
    assert(face.size()==3);			 	
    
    // area
    int vtx1 = face[0].mPositionIndex;
    int vtx2 = face[1].mPositionIndex;
    int vtx3 = face[2].mPositionIndex;
    
    float3 v = vPosList[vtx2] - vPosList[vtx1];
    float3 w = vPosList[vtx3] - vPosList[vtx1];
    
    float3 normal = v.cross(w);
    float length  = normal.length();
    float area = 0.5f*length;
    normal /= length;
    
    // area
    fAreaList[vtx1] +=area;
    fAreaList[vtx2] +=area;
    fAreaList[vtx3] +=area;
    
    // normal		
    vVtxNormalList[vtx1] += normal;
    vVtxNormalSize[vtx1] ++;
    vVtxNormalList[vtx2] += normal;
    vVtxNormalSize[vtx2] ++;
    vVtxNormalList[vtx3] += normal;
    vVtxNormalSize[vtx3] ++;
  } // end for i
  
  discs.resize(nVertices);
  for(i=0; i<(int)discs.size(); ++i)
  {
    discs[i].mPosition = vPosList[i];
    discs[i].mNormal = vVtxNormalList[i]/vVtxNormalSize[i];
    discs[i].mNormal = discs[i].mNormal /discs[i].mNormal.length();
    
    discs[i].mArea = fAreaList[i]/3;		
    discs[i].mVertex = i;
    discs[i].mOcclusion = 1.f;		
  } // end for i
  
  fAreaList.clear();
  vVtxNormalList.clear();
  vVtxNormalSize.clear();
  
  // create two vertex buffers for per-vertex occlusion
  std::vector<float3> tempOcclusion3(nVertices);
  std::fill(tempOcclusion3.begin(), tempOcclusion3.end(), float3(1,1,1));
  std::vector<float4> tempOcclusion4(nVertices);
  std::fill(tempOcclusion4.begin(), tempOcclusion4.end(), float4(1,1,1,1));

  mVertexBentNormalAndAccessibility.create();
  mVertexBentNormalAndAccessibility.setTarget(GL_ARRAY_BUFFER_ARB);
  mVertexBentNormalAndAccessibility.init(nVertices * 4 * sizeof(float),
                                            GL_DYNAMIC_DRAW,
                                            static_cast<float*>(tempOcclusion4[0]));
  
  mVertexScattering.create();
  mVertexScattering.setTarget(GL_ARRAY_BUFFER_ARB);
  mVertexScattering.init(nVertices * 3 * sizeof(float),
                            GL_DYNAMIC_DRAW,
                            static_cast<float*>(tempOcclusion3[0]));
  
  // do this the slow way
  std::vector<float3> vertexCoords = std::vector<float3> (nVertices);
  std::vector<float3> vertexNormals = std::vector<float3> (nVertices);
  for(i = 0; i <nVertices; ++i)
  {
    vertexCoords[i] = vPosList[i];
    vertexNormals[i] = discs[i].mNormal;
  } // end for i
  
  mVertexCoordinates.create();
  mVertexCoordinates.setTarget(GL_ARRAY_BUFFER_ARB);
  mVertexCoordinates.init(vertexCoords.size() * 3 * sizeof(float),
                          GL_STATIC_DRAW, static_cast<float*>(vertexCoords[0]));
  
  mVertexNormals.create();
  mVertexNormals.setTarget(GL_ARRAY_BUFFER_ARB);
  mVertexNormals.init(vertexNormals.size() * 3 * sizeof(float),
                      GL_STATIC_DRAW, static_cast<float*>(vertexNormals[0]));
  
  // vertex index
  std::vector<unsigned int> triangleVertexIndices(nFaces * 3);
  unsigned int v = 0;
  for(i = 0; i<(int)vFaceList.size(); ++i)
  {
    for(j = 0; j<(int)vFaceList[i].size(); ++j)		
    {
      triangleVertexIndices[v++] = vFaceList[i][j].mPositionIndex;
    } // end for j
  } // end for i
  
  mTriangleIndices.create();
  mTriangleIndices.setTarget(GL_ELEMENT_ARRAY_BUFFER_ARB);
  mTriangleIndices.init(triangleVertexIndices.size() * sizeof(unsigned int),
                        GL_STATIC_DRAW, &triangleVertexIndices[0]);
  
  // clear
  vertexCoords.clear();
  vertexNormals.clear();
  triangleVertexIndices.clear();
  tempOcclusion3.clear();
  tempOcclusion4.clear();
} // end ScatteringViewer::initVBOs()
