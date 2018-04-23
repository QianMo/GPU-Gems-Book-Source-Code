

#include "oglfunc.hpp"

#ifdef WIN32

#ifndef WGL_NV_float_buffer
#define WGL_FLOAT_COMPONENTS_NV        0x20B0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV 0x20B1
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV 0x20B2
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV 0x20B3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV 0x20B4
#define WGL_TEXTURE_FLOAT_R_NV         0x20B5
#define WGL_TEXTURE_FLOAT_RG_NV        0x20B6
#define WGL_TEXTURE_FLOAT_RGB_NV       0x20B7
#define WGL_TEXTURE_FLOAT_RGBA_NV      0x20B8
#endif

#ifndef WGL_NV_render_texture_rectangle
#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV 0x20A0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV 0x20A1
#define WGL_TEXTURE_RECTANGLE_NV       0x20A2
#endif

#ifndef WGL_ARB_render_texture
#define WGL_TEXTURE_FORMAT_ARB  0x2072
#define WGL_TEXTURE_TARGET_ARB  0x2073
#endif

#else

#ifndef GLX_FLOAT_COMPONENTS_NV
#define GLX_FLOAT_COMPONENTS_NV         0x20B0
#endif

#endif

#ifndef GL_NV_float_buffer
#define GL_FLOAT_R32_NV                   0x8885
#define GL_FLOAT_RG32_NV                  0x8887
#define GL_FLOAT_RGB32_NV                 0x8889
#define GL_FLOAT_RGBA32_NV                0x888B
#define GL_FLOAT_R16_NV                   0x8884
#define GL_FLOAT_RG16_NV                   0x8886
#define GL_FLOAT_RGB16_NV                   0x8888
#define GL_FLOAT_RGBA16_NV                   0x888A
#define GL_TEXTURE_FLOAT_COMPONENTS_NV    0x888C
//#define GL_LUMINANCE_ALPHA                  0x190A
#endif

#include "nvcontext.hpp"

using namespace brook;

static const unsigned int 
nvtypes[4][3] =   {{GL_FLOAT_R32_NV,GL_FLOAT_R16_NV,GL_LUMINANCE8},
                   {GL_FLOAT_RGBA32_NV,GL_FLOAT_RG16_NV,GL_LUMINANCE_ALPHA},
                   {GL_FLOAT_RGBA32_NV,GL_FLOAT_RGB16_NV,GL_RGB8},
                   {GL_FLOAT_RGBA32_NV,GL_FLOAT_RGBA16_NV,GL_RGBA8}};

static const unsigned int 
nvformats[4][3] =  { {GL_RED,GL_RED,GL_LUMINANCE},
                  {GL_RGBA,GL_LUMINANCE_ALPHA,GL_LUMINANCE_ALPHA},
                  {GL_RGBA,GL_RGB,GL_RGB},
                  {GL_RGBA,GL_RGBA,GL_RGBA} };

static const unsigned int 
sizefactor[4][3] = { {1,1,1}, {4,2,2}, {4,3,3}, {4,4,4} };
const static unsigned int 
atomSize[4][3]={{4,2,1},{4,2,1},{4,2,1},{4,2,1}};

NVTexture::NVTexture ( size_t inWidth, 
                       size_t inHeight, 
                       GPUContext::TextureFormat inFormat) :
  OGLTexture(inWidth, inHeight, inFormat, 
             nvformats, nvtypes, sizefactor, atomSize)
{
   _nativeFormat = nvformats[components()][elementType()]; 
}


static const int nviAttribList[4][64] = {
#ifdef WIN32
  {  WGL_FLOAT_COMPONENTS_NV,                     GL_TRUE, 
     WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV,    GL_TRUE,
     0,0},
  {  WGL_FLOAT_COMPONENTS_NV,                     GL_TRUE, 
     WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV,   GL_TRUE,
     0,0},
  {  WGL_FLOAT_COMPONENTS_NV,                     GL_TRUE, 
     WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV,  GL_TRUE,
     0,0},
  {  WGL_FLOAT_COMPONENTS_NV,                     GL_TRUE, 
     WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV, GL_TRUE,
     0,0}
#else
  {  GLX_FLOAT_COMPONENTS_NV, GL_TRUE,
     0,0},
  {  GLX_FLOAT_COMPONENTS_NV, GL_TRUE,
     0,0},
  {  GLX_FLOAT_COMPONENTS_NV, GL_TRUE,
     0,0},
  {  GLX_FLOAT_COMPONENTS_NV, GL_TRUE,
     0,0}
#endif
};


static const int nvpiAttribList[4][16] = {
#ifdef WIN32
  {  WGL_TEXTURE_FORMAT_ARB, WGL_TEXTURE_FLOAT_R_NV,
     WGL_TEXTURE_TARGET_ARB, WGL_TEXTURE_RECTANGLE_NV,
     0,0},
  {  WGL_TEXTURE_FORMAT_ARB, WGL_TEXTURE_FLOAT_RGBA_NV,
     WGL_TEXTURE_TARGET_ARB, WGL_TEXTURE_RECTANGLE_NV,
     0,0},
  {  WGL_TEXTURE_FORMAT_ARB, WGL_TEXTURE_FLOAT_RGBA_NV,
     WGL_TEXTURE_TARGET_ARB, WGL_TEXTURE_RECTANGLE_NV,
     0,0},
  {  WGL_TEXTURE_FORMAT_ARB, WGL_TEXTURE_FLOAT_RGBA_NV,
     WGL_TEXTURE_TARGET_ARB, WGL_TEXTURE_RECTANGLE_NV,
     0,0}
#else
  {  0,0},
  {  0,0},
  {  0,0},
  {  0,0}
#endif
};


NVContext::NVContext()
    : supportsFP40(false)
{}


NVContext *
NVContext::create() {
  NVContext *ctx = new NVContext();

  if (!ctx)
    return NULL;

  ctx->init(&nviAttribList, NULL, &nvpiAttribList);

  const char *ext = (const char *) glGetString(GL_EXTENSIONS);
  if(strstr(ext, "GL_NV_fragment_program2"))
      ctx->supportsFP40 = true;

  return ctx;
}


int 
NVContext::getShaderFormatRank (const char *name) const {
  if( strcmp(name, "arb") == 0 )
    return 1;
// TIM: fp30 uses different constant-setting interface, and thus
// isn't useful to use
  if( strcmp(name, "fp30") == 0 )
    return 3;
  if( supportsFP40 &&
      strcmp(name, "fp40") == 0 )
    return 4;
  return -1;
}


GPUContext::TextureHandle 
NVContext::createTexture2D( unsigned int inWidth,
                            unsigned int inHeight, 
                            GPUContext::TextureFormat inFormat) {
  return (GPUContext::TextureHandle) 
    new NVTexture(inWidth, inHeight, inFormat);
}


static const char nvext[][64] = {
  "GL_ARB_fragment_program",
  "GL_NV_float_buffer",
  "GL_NV_fragment_program",
  "GL_NV_texture_rectangle",
  ""};

bool
NVContext::isCompatibleContext () {
  const char *ext = (const char *) glGetString(GL_EXTENSIONS);
  int p;

  assert(ext);

  for (p = 0; *nvext[p]; p++) {
    if (!strstr(ext, nvext[p]))
      return false;
  }
  return true;
}

bool
NVContext::isVendorContext () {
  const char *vendor = (const char *) glGetString(GL_VENDOR);
  assert (vendor);
  return strstr(vendor, "NVIDIA") != NULL;
}

