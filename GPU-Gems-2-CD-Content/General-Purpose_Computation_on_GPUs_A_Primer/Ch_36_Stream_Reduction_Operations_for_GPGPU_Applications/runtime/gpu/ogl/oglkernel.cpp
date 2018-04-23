
#include "oglfunc.hpp"
#include "oglcontext.hpp"
#include "oglcheckgl.hpp"
#include "ogltexture.hpp"
#include "oglwindow.hpp"
#include "nvcontext.hpp"
using namespace brook;

static const char passthrough_vertex[] = 
"not used";

static const char passthrough_pixel[] =
"!!ARBfp1.0\n"
"ATTRIB tex0 = fragment.texcoord[0];\n"
"OUTPUT oColor = result.color;\n"
"TEX oColor, tex0, texture[0], RECT;\n"
"END\n";

OGLPixelShader::OGLPixelShader(unsigned int _id, const char * program_string):
  id(_id), largest_constant(0) {
  unsigned int i;
  
  for (i=0; i<(unsigned int) MAXCONSTANTS; i++) {
    constants[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  
  if (NVContext::isVendorContext()) {//only they rely on named constants
    if (strstr(program_string,"#profile fp30")) {
      while (*program_string&&(program_string=strstr(program_string,"#semantic main."))!=NULL) {
        const char *name;
        unsigned int len=0;
        program_string += strlen("#semantic main.");         
        /* set name to the ident */
        name = program_string;
        do{program_string++; len++;}while (*program_string!='\0'&&*program_string != ' ');
        std::string const_name(name,len);
        do program_string++; while (*program_string !=':');
        do program_string++; while (*program_string ==' ');
        if (*program_string!='C') continue;
        program_string++;
        char * ptr=NULL;
        unsigned int constreg = strtol (program_string,&ptr,10);
        if(ptr){
          if (constreg > (unsigned int)MAXCONSTANTS) {
            fprintf (stderr, "NV30GL: Too many constant registers\n");
            exit(1);
          }       
          program_string=ptr;
          constant_names[constreg] = const_name;
        }             
      }
    }
  }
  
}
  


GPUContext::VertexShaderHandle 
OGLContext::getPassthroughVertexShader(void) {
#if 0
  if (!_passthroughVertexShader) {
    GLuint id;
    glGenProgramsARB(1, &id);
    glBindProgramARB(GL_VERTEX_PROGRAM_ARB, id);
    glProgramStringARB(GL_VERTEX_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                       strlen(passthrough_vertex), 
                       (GLubyte *) passthrough_vertex);
    CHECK_GL();
    _passthroughVertexShader = (GPUContext::VertexShaderHandle) id;
  }
  return _passthroughVertexShader;
#else
  return (GPUContext::VertexShaderHandle) 1;
#endif
}

GPUContext::PixelShaderHandle 
OGLContext::getPassthroughPixelShader() {
  
  //fprintf (stderr, "getPassthroughPixelShader: this=0x%p\n", this);

  if (!_passthroughPixelShader) {
    GLuint id;
    //fprintf (stderr, "Calling glGenProgramsARB...\n");
    glGenProgramsARB(1, &id);
    //fprintf (stderr, "Calling glBindProgramARB...\n");
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, id);
    //fprintf (stderr, "Loading String: \n");
    //fprintf (stderr, "%s\n", passthrough_pixel);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                          strlen(passthrough_pixel), 
                          (GLubyte *) passthrough_pixel);
    //fprintf (stderr, "Mallocing PixelShader\n");
    _passthroughPixelShader = new OGLPixelShader(id,passthrough_pixel);
    //fprintf (stderr, "Checking GL\n");
    CHECK_GL();
  }

  //fprintf (stderr, "  returning 0x%p\n ", _passthroughPixelShader);
  return (GPUContext::PixelShaderHandle) _passthroughPixelShader;
}

GPUContext::PixelShaderHandle
OGLContext::createPixelShader( const char* shader ) 
{
  unsigned int id;

  // Allocate ids
  glGenProgramsARB(1, &id);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, id);

  // Try loading the program
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, 
                     GL_PROGRAM_FORMAT_ASCII_ARB,
                     strlen(shader), (GLubyte *) shader);
  
  /* Check for program errors */
  if (glGetError() == GL_INVALID_OPERATION) {
    GLint pos;
    int i;
    int line, linestart;
    char *progcopy;

    progcopy = strdup (shader);
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &pos);
    
    line = 1;
    linestart = 0;
    for (i=0; i<pos; i++) {
      if (progcopy[i] == '\n') {
        line++;
        linestart = i+1;
      }
    }
    fprintf ( stderr, "GL: Program Error on line %d\n", line);
    for (i=linestart; progcopy[i] != '\0' && progcopy[i] != '\n'; i++);
    progcopy[i] = '\0';
    fprintf ( stderr, "%s\n", progcopy+linestart);
    for (i=linestart; i<pos; i++)
      fprintf ( stderr, " ");
    for (;progcopy[i] != '\0' && progcopy[i] != '\n'; i++)
      fprintf ( stderr, "^");
    fprintf ( stderr, "\n");
    free(progcopy);
    fprintf ( stderr, "%s\n",
              glGetString(GL_PROGRAM_ERROR_STRING_ARB));
    fflush(stderr);
    assert(0);
    exit(1);
  }

  return (GPUContext::PixelShaderHandle) new OGLPixelShader(id,shader);
}

void 
OGLContext::bindConstant( PixelShaderHandle ps,
                          unsigned int inIndex, 
                          const float4& inValue ) {
  
  OGLPixelShader *oglps = (OGLPixelShader *) ps;

  GPUAssert(oglps, "Missing shader");

  bindPixelShader(ps);
  glProgramLocalParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, inIndex,
                                (const float *) &inValue);
  CHECK_GL();
  std::string::size_type len=oglps->constant_names[inIndex].length();
  if (len){
    glProgramNamedParameter4fvNV(oglps->id,
                                 len,
                                 (const GLubyte *)oglps->constant_names[inIndex].c_str(),
                                 &inValue.x);
    GLenum err=glGetError();
    //errors come if a constant is passed into a kernel but optimized out of that kernel.
    // they are "safe" to ignore
    assert (err==GL_NO_ERROR||err==GL_INVALID_VALUE);
  }
  GPUAssert(inIndex < (unsigned int) OGLPixelShader::MAXCONSTANTS, 
            "Too many constants used in kernel");

  if (inIndex >= oglps->largest_constant)
    oglps->largest_constant = inIndex+1;

  oglps->constants[inIndex] = inValue;
}


void 
OGLContext::bindTexture( unsigned int inIndex, 
                         TextureHandle inTexture ) {
  OGLTexture *oglTexture = (OGLTexture *) inTexture;
  
  GPUAssert(oglTexture, "Null Texture");
  GPUAssert(inIndex < _slopTextureUnit, 
            "Too many bound textures");

  glActiveTextureARB(GL_TEXTURE0_ARB+inIndex);
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, oglTexture->id());

  _boundTextures[inIndex] = oglTexture;

  CHECK_GL();
}


void OGLContext::bindOutput( unsigned int inIndex, 
                             TextureHandle inTexture ) {
  OGLTexture *oglTexture = (OGLTexture *) inTexture;
  
  GPUAssert(oglTexture, "Null Texture");

  GPUAssert(inIndex <= _maxOutputCount , 
            "Backend does not support more than"
            " four shader output.");

  _outputTextures[inIndex] = oglTexture;
}

void 
OGLContext::bindPixelShader( GPUContext::PixelShaderHandle inPixelShader ) {
  
  OGLPixelShader *ps = (OGLPixelShader *) inPixelShader;
  GPUAssert(ps, "Null pixel shader");

  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, ps->id);
  CHECK_GL();
  
  _boundPixelShader = ps;
}

void 
OGLContext::bindVertexShader( GPUContext::VertexShaderHandle inVertexShader ) {
#if 0
  glBindProgramARB(GL_VERTEX_PROGRAM_ARB, 
                   (unsigned int) inVertexShader);
  CHECK_GL();
#endif
}

void OGLContext::disableOutput( unsigned int inIndex ) {
 GPUAssert(inIndex <= 4,
           "Backend does not support more than"
           " four shader outputs.");

 _outputTextures[inIndex] = NULL;
}

void
OGLContext::get1DInterpolant( const float4 &start, 
                              const float4 &end,
                              const unsigned int w,
                              GPUInterpolant &interpolant) const {

  if (w == 1) {
    interpolant.vertices[0] = start;
    interpolant.vertices[1] = start;
    interpolant.vertices[2] = start;
    return;
  }

  float4 f1, f2;
  float bias = 0.00001f;
  
  float x1 = start.x;
  float y1 = start.y;
  float z1 = start.z;
  float w1 = start.w;

  float x2 = end.x;
  float y2 = end.y;
  float z2 = end.z;
  float w2 = end.w;

  float sx = x2-x1;
  float sy = y2-y1;
  float sz = z2-z1;
  float sw = w2-w1;
  float ratiox = sx / w;
  float ratioy = sy / w;
  float ratioz = sz / w;
  float ratiow = sw / w;
  float shiftx = ratiox * 0.5f;
  float shifty = ratioy * 0.5f;
  float shiftz = ratioz * 0.5f;
  float shiftw = ratiow * 0.5f;

  f1.x = x1 - shiftx + bias;
  f1.y = y1 - shifty + bias;
  f1.z = z1 - shiftz + bias;
  f1.w = w1 - shiftw + bias;

  f2.x = (x1+2*sx) - shiftx + bias;
  f2.y = (y1+2*sy) - shifty + bias;
  f2.z = (z1+2*sz) - shiftz + bias;
  f2.w = (w1+2*sw) - shiftw + bias;

  interpolant.vertices[0] = f1;
  interpolant.vertices[1] = f2; 
  interpolant.vertices[2] = f1;
}


void
OGLContext::get2DInterpolant( const float2 &start, 
                              const float2 &end,
                              const unsigned int w,
                              const unsigned int h,
                              GPUInterpolant &interpolant) const {
  float2 f1, f2;
  
  float x1 = start.x;
  float y1 = start.y;
  float x2 = end.x;
  float y2 = end.y;
  float bias = 0.00001f;
  
  if (w==1 && h==1) {
    float4 v (start.x, start.y, 0.0f, 1.0f);
    interpolant.vertices[0] = v;
    interpolant.vertices[1] = v;
    interpolant.vertices[2] = v;
    return;
  }

  float sx = x2-x1;
  float sy = y2-y1;
  float ratiox = sx / w;
  float ratioy = sy / h;
  float shiftx = ratiox * 0.5f;
  float shifty = ratioy * 0.5f;

  f1.x = x1 - shiftx + bias;
  f1.y = y1 - shifty + bias;

  f2.x = (x1+2*sx) - shiftx + bias;
  f2.y = (y1+2*sy) - shifty + bias;

  if (h==1) {
//    interpolant.vertices[0] = float4(f1.x, f1.y, 0.0f, 1.0f);
//    interpolant.vertices[1] = float4(f2.x, f1.y, 0.0f, 1.0f);
//    interpolant.vertices[2] = interpolant.vertices[0];
    interpolant.vertices[0] = float4(f1.x, y1, 0.0f, 1.0f);
    interpolant.vertices[1] = float4(f2.x, y1, 0.0f, 1.0f);
    interpolant.vertices[2] = interpolant.vertices[0];
    return;
  }

  if (w==1) {
//    interpolant.vertices[0] = float4(f1.x, f1.y, 0.0f, 1.0f);
//    interpolant.vertices[1] = interpolant.vertices[0];
//    interpolant.vertices[2] = float4(f1.x, f2.y, 0.0f, 1.0f);
    interpolant.vertices[0] = float4(x1, f1.y, 0.0f, 1.0f);
    interpolant.vertices[1] = interpolant.vertices[0];
    interpolant.vertices[2] = float4(x1, f2.y, 0.0f, 1.0f);
    return;
  }

  interpolant.vertices[0] = float4(f1.x, f1.y, 0.0f, 1.0f);
  interpolant.vertices[1] = float4(f2.x, f1.y, 0.0f, 1.0f);
  interpolant.vertices[2] = float4(f1.x, f2.y, 0.0f, 1.0f);
}



float4 
OGLContext::getStreamIndexofConstant( TextureHandle inTexture ) const {
  return float4(1.0f, 1.0f, 0.0f, 0.0f);
}


float4
OGLContext::getStreamGatherConstant(
                                    unsigned int inRank, const unsigned int* inDomainMin,
                                    const unsigned int* inDomainMax, const unsigned int* inExtents ) const {
  float scaleX = 1.0f;
  float scaleY = 1.0f;
  float offsetX = 0.0f;
  float offsetY = 0.0f;
  if( inRank == 1 )
  {
    unsigned int base = inDomainMin[0];

    offsetX = base + 0.5f;
  }
  else
  {
    unsigned int baseX = inDomainMin[1];
    unsigned int baseY = inDomainMin[0];

    offsetX = baseX + 0.5f;
    offsetY = baseY + 0.5f;
  }
  return float4( scaleX, scaleY, offsetX, offsetY );
//  return float4(1.0f, 1.0f, 0.5f, 0.5f);
}


void
OGLContext::getStreamInterpolant( const TextureHandle texture,
                                  unsigned int rank,
                                  const unsigned int* domainMin,
                                  const unsigned int* domainMax,
                                  unsigned int w,
                                  unsigned int h,
                                  GPUInterpolant &interpolant) const {

  unsigned int minX, minY, maxX, maxY;
  if( rank == 1 )
  {
      minX = domainMin[0];
      minY = 0;
      maxX = domainMax[0];
      maxY = 0;
  }
  else
  {
      minX = domainMin[1];
      minY = domainMin[0];
      maxX = domainMax[1];
      maxY = domainMax[0];
  }

  float2 start(minX + 0.005f, minY + 0.005f);
  float2 end(maxX + 0.005f, maxY + 0.005f);

  get2DInterpolant(  start, end, w, h, interpolant); 
}

void
OGLContext::getStreamOutputRegion( const TextureHandle texture,
                                   unsigned int rank,
                                   const unsigned int* domainMin,
                                   const unsigned int* domainMax,
                                   GPURegion &region) const
{
  unsigned int minX, minY, maxX, maxY;
  if( rank == 1 )
  {
      minX = domainMin[0];
      minY = 0;
      maxX = domainMax[0];
      maxY = 1;
  }
  else
  {
      minX = domainMin[1];
      minY = domainMin[0];
      maxX = domainMax[1];
      maxY = domainMax[0];
  }

  region.vertices[0].x = -1;
  region.vertices[0].y = -1;

  region.vertices[1].x = 3;
  region.vertices[1].y = -1;

  region.vertices[2].x = -1;
  region.vertices[2].y = 3;

  region.viewport.minX = minX;
  region.viewport.minY = minY;
  region.viewport.maxX = maxX;
  region.viewport.maxY = maxY;
}

void 
OGLContext::getStreamReduceInterpolant( const TextureHandle inTexture,
                                        const unsigned int outputWidth,
                                        const unsigned int outputHeight, 
                                        const unsigned int minX,
                                        const unsigned int maxX, 
                                        const unsigned int minY,
                                        const unsigned int maxY,
                                        GPUInterpolant &interpolant) const
{
    float2 start(0.005f + minX, 0.005f + minY);
    float2 end(0.005f + maxX, 0.005f + maxY);

    get2DInterpolant( start, end, outputWidth, outputHeight, interpolant); 
}

void
OGLContext::getStreamReduceOutputRegion( const TextureHandle inTexture,
                                         const unsigned int minX,
                                         const unsigned int maxX, 
                                         const unsigned int minY,
                                         const unsigned int maxY,
                                         GPURegion &region) const
{
  region.vertices[0].x = -1;
  region.vertices[0].y = -1;

  region.vertices[1].x = 3;
  region.vertices[1].y = -1;

  region.vertices[2].x = -1;
  region.vertices[2].y = 3;

  region.viewport.minX = minX;
  region.viewport.minY = minY;
  region.viewport.maxX = maxX;
  region.viewport.maxY = maxY;
}

void 
OGLContext::drawRectangle( const GPURegion& outputRegion, 
                           const GPUInterpolant* interpolants, 
                           unsigned int numInterpolants ) {
  unsigned int w, h, i, v;
  unsigned int numOutputs, maxComponent;
  static const GLenum outputEnums[] = {GL_FRONT_LEFT, GL_AUX0,
                                       GL_AUX1, GL_AUX2};

  /* Here we assume that all of the outputs are of the same size */
  w = _outputTextures[0]->width();
  h = _outputTextures[0]->height();

  numOutputs = 0;
  maxComponent = 0;
  for (i=0; i<4; i++) {
    if (_outputTextures[i]) {
      numOutputs = i+1;
      if (_outputTextures[i]->components() > maxComponent)
        maxComponent = _outputTextures[i]->components();
    }
  }

  CHECK_GL();

  if (_wnd->bindPbuffer(w, h, numOutputs, maxComponent)) {

    // Rebind the shader
    GPUAssert(_boundPixelShader, "Missing pixel shader");
    bindPixelShader((PixelShaderHandle) _boundPixelShader);
    
    // Rebind the textures
    for (i=0; i<32; i++) 
      if (_boundTextures[i]) {
        glActiveTextureARB(GL_TEXTURE0_ARB+i);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, _boundTextures[i]->id());
      }

    // Rebind the constants
    for (i=0; i<_boundPixelShader->largest_constant; i++) {
      bindConstant((PixelShaderHandle) _boundPixelShader,
                   i, _boundPixelShader->constants[i]);
    }
        
  }
  
  CHECK_GL();

  // TIM: hacky magic magic
  if( _isUsingAddressTranslation && _isUsingOutputDomain )
  {
    // if we are writing to a domain of an address-translated
    // stream, then copy the output textures to the pbuffer
    // so that we can overwrite the proper domain

    // NOTE: this will fail if we try to optimize domain
    // handling by only drawing to a subrectangle - for
    // now we render to the whole thing, so copying in
    // the whole texture is correct
    for( i = 0; i < numOutputs; i++ )
    {
      OGLTexture* output = _outputTextures[i];
      glDrawBuffer(outputEnums[i]);
      copy_to_pbuffer(output);
    }

    // We need to rebind stuff since we messed up the state
    // of things

    // Rebind the shader
    GPUAssert(_boundPixelShader, "Missing pixel shader");
    bindPixelShader((PixelShaderHandle) _boundPixelShader);

    // Rebind the textures
    for (i=0; i<32; i++) 
    if (_boundTextures[i]) {
    glActiveTextureARB(GL_TEXTURE0_ARB+i);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _boundTextures[i]->id());
    }

    // Rebind the constants
    for (i=0; i<_boundPixelShader->largest_constant; i++) {
    bindConstant((PixelShaderHandle) _boundPixelShader,
    i, _boundPixelShader->constants[i]);
    }
  }

  if (_maxOutputCount > 1)
    glDrawBuffersATI (numOutputs, outputEnums); 
   

  if (_maxOutputCount > 1)
    glDrawBuffersATI (numOutputs, outputEnums); 

  CHECK_GL();
  
  /*
   * We execute our kernel by using it to texture a triangle that
   * has vertices (-1, 3), (-1, -1), and (3, -1) which works out
   * nicely to contain the square (-1, -1), (-1, 1), (1, 1), (1, -1).
   */

  int minX = outputRegion.viewport.minX;
  int minY = outputRegion.viewport.minY;
  int maxX = outputRegion.viewport.maxX;
  int maxY = outputRegion.viewport.maxY;
  int width = maxX - minX;
  int height = maxY - minY;

  CHECK_GL();

  glViewport( minX, minY, width, height );

  CHECK_GL();

  glBegin(GL_TRIANGLES);

  for (v=0; v<3; v++ )
  {
        GPULOG(1) << "vertex " << v;

        for (i=0; i<numInterpolants; i++) 
        {
            glMultiTexCoord4fvARB(GL_TEXTURE0_ARB+i,
                                (GLfloat *) &(interpolants[i].vertices[v]));

            GPULOG(1) << "tex" << i << " : " << interpolants[i].vertices[v].x
                << ", " << interpolants[i].vertices[v].y;
        }
        glVertex2fv((GLfloat *) &(outputRegion.vertices[v]));
        GPULOG(1) << "pos : " << outputRegion.vertices[v].x
            << ", " << outputRegion.vertices[v].y;
  }
  glEnd();
  CHECK_GL();

  /* Copy the output to the texture */
  for (i=0; i<numOutputs; i++) {
    glActiveTextureARB(GL_TEXTURE0+_slopTextureUnit);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _outputTextures[i]->id());
    glReadBuffer(outputEnums[i]);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 
                        minX, minY, 
                        minX, minY, 
                        width, height);
    CHECK_GL();
  }
  glReadBuffer(outputEnums[0]);
  glDrawBuffer(outputEnums[0]);

  for (i=0; i<4; i++)
    _outputTextures[i] = NULL;
}
