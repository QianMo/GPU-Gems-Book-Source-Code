#ifdef GLEXT_PROTOTYPE
	int				ARB_texture_cube_map = 0;
	int				ARB_texture_border_clamp = 0;
	int				ARB_texture_env_add = 0;
	int				ARB_texture_env_combine = 0;
	int				ARB_texture_env_crossbar = 0;
	int				ARB_texture_env_dot3 = 0;
	int				ARB_transpose_matrix = 0;
	int				ATI_texture_mirror_once = 0;
	int				NV_texgen_reflection = 0;
	int				NV_blend_square = 0;
	int				NV_fog_distance = 0;
	int				NV_texgen_emboss = 0;
	int				NV_texture_env_combine4 = 0;
	int				NV_texture_shader = 0;
	int				NV_texture_shader2 = 0;
	int				NV_texture_rectangle = 0;
	int				NV_vertex_array_range2 = 0;
	int				SGIS_generate_mipmap = 0;
#else
	extern int		ARB_texture_cube_map;
	extern int		ARB_texture_border_clamp;
	extern int		ARB_texture_env_add;
	extern int		ARB_texture_env_combine;
	extern int		ARB_texture_env_crossbar;
	extern int		ARB_texture_env_dot3;
	extern int		ARB_transpose_matrix;
	extern int		ATI_texture_mirror_once;
	extern int		NV_texture_shader;
	extern int		NV_texture_shader2;
	extern int		NV_texture_rectangle;
	extern int		NV_texgen_reflection;
	extern int		NV_blend_square;
	extern int		NV_fog_distance;
	extern int		NV_texgen_emboss;
	extern int		NV_texture_env_combine4;
	extern int		NV_vertex_array_range2;
	extern int		SGIS_generate_mipmap;
#endif

//=================//
// ARB_multisample //
//=================//
#ifdef GLEXT_PROTOTYPE
	int								ARB_multisample = 0;
	PFNGLSAMPLECOVERAGEARBPROC		glSampleCoverageARB = NULL;
#else
	extern int							ARB_multisample;
	extern PFNGLSAMPLECOVERAGEARBPROC	glSampleCoverageARB;
#endif

//====================//
// ARB_matrix_palette //
//====================//
#ifdef GLEXT_PROTOTYPE
   int									ARB_matrix_palette = 0;
   PFNGLCURRENTPALETTEMATRIXARBPROC		glCurrentPaletteMatrixARB = NULL;
   PFNGLMATRIXINDEXUBVARBPROC			glMatrixIndexubvARB = NULL;
   PFNGLMATRIXINDEXUSVARBPROC			glMatrixIndexusvARB = NULL;
   PFNGLMATRIXINDEXUIVARBPROC			glMatrixIndexuivARB = NULL;
   PFNGLMATRIXINDEXPOINTERARBPROC		glMatrixIndexPointerARB = NULL;
#else
   extern int									ARB_matrix_palette;
   extern PFNGLCURRENTPALETTEMATRIXARBPROC		glCurrentPaletteMatrixARB;
   extern PFNGLMATRIXINDEXUBVARBPROC			glMatrixIndexubvARB;
   extern PFNGLMATRIXINDEXUSVARBPROC			glMatrixIndexusvARB;
   extern PFNGLMATRIXINDEXUIVARBPROC			glMatrixIndexuivARB;
   extern PFNGLMATRIXINDEXPOINTERARBPROC		glMatrixIndexPointerARB;
#endif

//==================//
// ARB_vertex_blend //
//==================//
#ifdef GLEXT_PROTOTYPE
   int                                 ARB_vertex_blend = 0;
   PFNGLWEIGHTBVARBPROC                glWeightbvARB = NULL;
   PFNGLWEIGHTSVARBPROC                glWeightsvARB = NULL;
   PFNGLWEIGHTIVARBPROC                glWeightivARB = NULL;
   PFNGLWEIGHTFVARBPROC                glWeightfvARB = NULL;
   PFNGLWEIGHTDVARBPROC                glWeightdvARB = NULL;
   PFNGLWEIGHTUBVARBPROC               glWeightubvARB = NULL;
   PFNGLWEIGHTUSVARBPROC               glWeightusvARB = NULL;
   PFNGLWEIGHTUIVARBPROC               glWeightuivARB = NULL;
   PFNGLWEIGHTPOINTERARBPROC           glWeightPointerARB = NULL;
   PFNGLVERTEXBLENDARBPROC             glVertexBlendARB = NULL;
#else
   extern int                          ARB_vertex_blend;
   extern PFNGLWEIGHTBVARBPROC         glWeightbvARB;
   extern PFNGLWEIGHTSVARBPROC         glWeightsvARB;
   extern PFNGLWEIGHTIVARBPROC         glWeightivARB;
   extern PFNGLWEIGHTFVARBPROC         glWeightfvARB;
   extern PFNGLWEIGHTDVARBPROC         glWeightdvARB;
   extern PFNGLWEIGHTUBVARBPROC        glWeightubvARB;
   extern PFNGLWEIGHTUSVARBPROC        glWeightusvARB;
   extern PFNGLWEIGHTUIVARBPROC        glWeightuivARB;
   extern PFNGLWEIGHTPOINTERARBPROC    glWeightPointerARB;
   extern PFNGLVERTEXBLENDARBPROC      glVertexBlendARB;
#endif

//=========================//
// ARB_texture_compression //
//=========================//
#ifdef GLEXT_PROTOTYPE
	int											ARB_texture_compression = 0;
	PFNGLCOMPRESSEDTEXIMAGE3DARBPROC			glCompressedTexImage3DARB = NULL;
	PFNGLCOMPRESSEDTEXIMAGE2DARBPROC			glCompressedTexImage2DARB = NULL;
	PFNGLCOMPRESSEDTEXIMAGE1DARBPROC			glCompressedTexImage1DARB = NULL;
	PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC			glCompressedTexSubImage3DARB = NULL;
	PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC			glCompressedTexSubImage2DARB = NULL;
	PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC			glCompressedTexSubImage1DARB = NULL;
	PFNGLGETCOMPRESSEDTEXIMAGEARBPROC			glGetCompressedTexImageARB = NULL;
#else
	extern int									ARB_texture_compression;
	extern PFNGLCOMPRESSEDTEXIMAGE3DARBPROC		glCompressedTexImage3DARB;
	extern PFNGLCOMPRESSEDTEXIMAGE2DARBPROC		glCompressedTexImage2DARB;
	extern PFNGLCOMPRESSEDTEXIMAGE1DARBPROC		glCompressedTexImage1DARB;
	extern PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC	glCompressedTexSubImage3DARB; 
	extern PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC	glCompressedTexSubImage2DARB; 
	extern PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC	glCompressedTexSubImage1DARB; 
	extern PFNGLGETCOMPRESSEDTEXIMAGEARBPROC	glGetCompressedTexImageARB;
#endif

//=====================//
// ARB_multitexture //
//=====================//
#ifdef GLEXT_PROTOTYPE
   int                                             ARB_multitexture = 0;
   PFNGLACTIVETEXTUREARBPROC                       glActiveTextureARB;
   PFNGLCLIENTACTIVETEXTUREARBPROC                 glClientActiveTextureARB;
   PFNGLMULTITEXCOORD1DARBPROC                     glMultiTexCoord1dARB;
   PFNGLMULTITEXCOORD1DVARBPROC                    glMultiTexCoord1dvARB;
   PFNGLMULTITEXCOORD1FARBPROC                     glMultiTexCoord1fARB;
   PFNGLMULTITEXCOORD1FVARBPROC                    glMultiTexCoord1fvARB;
   PFNGLMULTITEXCOORD1IARBPROC                     glMultiTexCoord1iARB;
   PFNGLMULTITEXCOORD1IVARBPROC                    glMultiTexCoord1ivARB;
   PFNGLMULTITEXCOORD1SARBPROC                     glMultiTexCoord1sARB;
   PFNGLMULTITEXCOORD1SVARBPROC                    glMultiTexCoord1svARB;
   PFNGLMULTITEXCOORD2DARBPROC                     glMultiTexCoord2dARB;
   PFNGLMULTITEXCOORD2DVARBPROC                    glMultiTexCoord2dvARB;
   PFNGLMULTITEXCOORD2FARBPROC                     glMultiTexCoord2fARB;
   PFNGLMULTITEXCOORD2FVARBPROC                    glMultiTexCoord2fvARB;
   PFNGLMULTITEXCOORD2IARBPROC                     glMultiTexCoord2iARB;
   PFNGLMULTITEXCOORD2IVARBPROC                    glMultiTexCoord2ivARB;
   PFNGLMULTITEXCOORD2SARBPROC                     glMultiTexCoord2sARB;
   PFNGLMULTITEXCOORD2SVARBPROC                    glMultiTexCoord2svARB;
   PFNGLMULTITEXCOORD3DARBPROC                     glMultiTexCoord3dARB;
   PFNGLMULTITEXCOORD3DVARBPROC                    glMultiTexCoord3dvARB;
   PFNGLMULTITEXCOORD3FARBPROC                     glMultiTexCoord3fARB;
   PFNGLMULTITEXCOORD3FVARBPROC                    glMultiTexCoord3fvARB;
   PFNGLMULTITEXCOORD3IARBPROC                     glMultiTexCoord3iARB;
   PFNGLMULTITEXCOORD3IVARBPROC                    glMultiTexCoord3ivARB;
   PFNGLMULTITEXCOORD3SARBPROC                     glMultiTexCoord3sARB;
   PFNGLMULTITEXCOORD3SVARBPROC                    glMultiTexCoord3svARB;
   PFNGLMULTITEXCOORD4DARBPROC                     glMultiTexCoord4dARB;
   PFNGLMULTITEXCOORD4DVARBPROC                    glMultiTexCoord4dvARB;
   PFNGLMULTITEXCOORD4FARBPROC                     glMultiTexCoord4fARB;
   PFNGLMULTITEXCOORD4FVARBPROC                    glMultiTexCoord4fvARB;
   PFNGLMULTITEXCOORD4IARBPROC                     glMultiTexCoord4iARB;
   PFNGLMULTITEXCOORD4IVARBPROC                    glMultiTexCoord4ivARB;
   PFNGLMULTITEXCOORD4SARBPROC                     glMultiTexCoord4sARB;
   PFNGLMULTITEXCOORD4SVARBPROC                    glMultiTexCoord4svARB;
#else
   extern int                                             ARB_multitexture;
   extern PFNGLACTIVETEXTUREARBPROC                       glActiveTextureARB;
   extern PFNGLCLIENTACTIVETEXTUREARBPROC                 glClientActiveTextureARB;
   extern PFNGLMULTITEXCOORD1DARBPROC                     glMultiTexCoord1dARB;
   extern PFNGLMULTITEXCOORD1DVARBPROC                    glMultiTexCoord1dvARB;
   extern PFNGLMULTITEXCOORD1FARBPROC                     glMultiTexCoord1fARB;
   extern PFNGLMULTITEXCOORD1FVARBPROC                    glMultiTexCoord1fvARB;
   extern PFNGLMULTITEXCOORD1IARBPROC                     glMultiTexCoord1iARB;
   extern PFNGLMULTITEXCOORD1IVARBPROC                    glMultiTexCoord1ivARB;
   extern PFNGLMULTITEXCOORD1SARBPROC                     glMultiTexCoord1sARB;
   extern PFNGLMULTITEXCOORD1SVARBPROC                    glMultiTexCoord1svARB;
   extern PFNGLMULTITEXCOORD2DARBPROC                     glMultiTexCoord2dARB;
   extern PFNGLMULTITEXCOORD2DVARBPROC                    glMultiTexCoord2dvARB;
   extern PFNGLMULTITEXCOORD2FARBPROC                     glMultiTexCoord2fARB;
   extern PFNGLMULTITEXCOORD2FVARBPROC                    glMultiTexCoord2fvARB;
   extern PFNGLMULTITEXCOORD2IARBPROC                     glMultiTexCoord2iARB;
   extern PFNGLMULTITEXCOORD2IVARBPROC                    glMultiTexCoord2ivARB;
   extern PFNGLMULTITEXCOORD2SARBPROC                     glMultiTexCoord2sARB;
   extern PFNGLMULTITEXCOORD2SVARBPROC                    glMultiTexCoord2svARB;
   extern PFNGLMULTITEXCOORD3DARBPROC                     glMultiTexCoord3dARB;
   extern PFNGLMULTITEXCOORD3DVARBPROC                    glMultiTexCoord3dvARB;
   extern PFNGLMULTITEXCOORD3FARBPROC                     glMultiTexCoord3fARB;
   extern PFNGLMULTITEXCOORD3FVARBPROC                    glMultiTexCoord3fvARB;
   extern PFNGLMULTITEXCOORD3IARBPROC                     glMultiTexCoord3iARB;
   extern PFNGLMULTITEXCOORD3IVARBPROC                    glMultiTexCoord3ivARB;
   extern PFNGLMULTITEXCOORD3SARBPROC                     glMultiTexCoord3sARB;
   extern PFNGLMULTITEXCOORD3SVARBPROC                    glMultiTexCoord3svARB;
   extern PFNGLMULTITEXCOORD4DARBPROC                     glMultiTexCoord4dARB;
   extern PFNGLMULTITEXCOORD4DVARBPROC                    glMultiTexCoord4dvARB;
   extern PFNGLMULTITEXCOORD4FARBPROC                     glMultiTexCoord4fARB;
   extern PFNGLMULTITEXCOORD4FVARBPROC                    glMultiTexCoord4fvARB;
   extern PFNGLMULTITEXCOORD4IARBPROC                     glMultiTexCoord4iARB;
   extern PFNGLMULTITEXCOORD4IVARBPROC                    glMultiTexCoord4ivARB;
   extern PFNGLMULTITEXCOORD4SARBPROC                     glMultiTexCoord4sARB;
   extern PFNGLMULTITEXCOORD4SVARBPROC                    glMultiTexCoord4svARB;
#endif

//===============//
// EXT_texture3D //
//===============//
#ifdef GLEXT_PROTOTYPE
	int								EXT_texture3D = 0;
	PFNGLTEXIMAGE3DEXTPROC			glTexImage3DEXT=NULL;
	PFNGLTEXSUBIMAGE3DPROC			glTexSubImage3DEXT=NULL;
	PFNGLCOPYTEXSUBIMAGE3DPROC		glCopyTexSubImage3DEXT=NULL;
#else
	extern int							EXT_texture3D;
	extern PFNGLTEXIMAGE3DEXTPROC		glTexImage3DEXT;
	extern PFNGLTEXSUBIMAGE3DPROC		glTexSubImage3DEXT;
	extern PFNGLCOPYTEXSUBIMAGE3DPROC	glCopyTexSubImage3DEXT;
#endif

//======================//
// EXT_stencil_two_side //
//======================//
#ifdef GLEXT_PROTOTYPE
	int								EXT_stencil_two_side = 0;
	PFNGLACTIVESTENCILFACEXT		glActiveStencilFaceEXT = NULL;
#else
	extern int							EXT_stencil_two_side;
	extern PFNGLACTIVESTENCILFACEXT		glActiveStencilFaceEXT;
#endif

//===========================//
// EXT_compiled_vertex_array //
//===========================//
#ifdef GLEXT_PROTOTYPE
   int                              EXT_compiled_vertex_array = 0;
   PFNGLLOCKARRAYSEXTPROC           glLockArraysEXT = NULL;
   PFNGLUNLOCKARRAYSEXTPROC         glUnlockArraysEXT = NULL;
#else
   extern int                       EXT_compiled_vertex_array;
   extern PFNGLLOCKARRAYSEXTPROC    glLockArraysEXT;
   extern PFNGLUNLOCKARRAYSEXTPROC  glUnlockArraysEXT;
#endif

//===============//
// EXT_fog_coord //
//===============//
#ifdef GLEXT_PROTOTYPE
	int								EXT_fog_coord = 0;
	PFNGLFOGCOORDFEXTPROC			glFogCoordfEXT = NULL;
	PFNGLFOGCOORDDEXTPROC			glFogCoorddEXT = NULL;
	PFNGLFOGCOORDFVEXTPROC			glFogCoordfvEXT = NULL;
	PFNGLFOGCOORDDVEXTPROC			glFogCoorddvEXT = NULL;
	PFNGLFOGCOORDPOINTEREXTPROC		glFogCoordPointerEXT = NULL;
#else
	extern int							EXT_fog_coord;
	extern PFNGLFOGCOORDFEXTPROC		glFogCoordfEXT;
	extern PFNGLFOGCOORDDEXTPROC		glFogCoorddEXT;
	extern PFNGLFOGCOORDFVEXTPROC		glFogCoordfvEXT; 
	extern PFNGLFOGCOORDDVEXTPROC		glFogCoorddvEXT;
	extern PFNGLFOGCOORDPOINTEREXTPROC	glFogCoordPointerEXT;
#endif

//=========================//
// ATI_vertex_array_object //
//=========================//
#ifdef GLEXT_PROTOTYPE
   int                                    ATI_vertex_array_object = 0;
   PFNGLNEWOBJECTBUFFERATIPROC            glNewObjectBufferATI = NULL;
   PFNGLISOBJECTBUFFERATIPROC             glIsObjectBufferATI = NULL;
   PFNGLUPDATEOBJECTBUFFERATIPROC         glUpdateObjectBufferATI = NULL;
   PFNGLGETOBJECTBUFFERFVATIPROC          glGetObjectBufferfvATI = NULL;
   PFNGLGETOBJECTBUFFERIVATIPROC          glGetObjectBufferivATI = NULL;
   PFNGLFREEOBJECTBUFFERATIPROC           glFreeObjectBufferATI = NULL;
   PFNGLARRAYOBJECTATIPROC                glArrayObjectATI = NULL;
   PFNGLGETARRAYOBJECTFVATIPROC           glGetArrayObjectfvATI = NULL;
   PFNGLGETARRAYOBJECTIVATIPROC           glGetArrayObjectivATI = NULL;
   PFNGLVARIANTARRAYOBJECTATIPROC         glVariantArrayObjectATI = NULL;
   PFNGLGETVARIANTARRAYOBJECTFVATIPROC    glGetVariantArrayObjectfvATI = NULL;
   PFNGLGETVARIANTARRAYOBJECTIVATIPROC    glGetVariantArrayObjectivATI = NULL;
#else
   extern int                                 ATI_vertex_array_object;
   extern PFNGLNEWOBJECTBUFFERATIPROC         glNewObjectBufferATI;
   extern PFNGLISOBJECTBUFFERATIPROC          glIsObjectBufferATI;
   extern PFNGLUPDATEOBJECTBUFFERATIPROC      glUpdateObjectBufferATI;
   extern PFNGLGETOBJECTBUFFERFVATIPROC       glGetObjectBufferfvATI;
   extern PFNGLGETOBJECTBUFFERIVATIPROC       glGetObjectBufferivATI;
   extern PFNGLFREEOBJECTBUFFERATIPROC        glFreeObjectBufferATI;
   extern PFNGLARRAYOBJECTATIPROC             glArrayObjectATI;
   extern PFNGLGETARRAYOBJECTFVATIPROC        glGetArrayObjectfvATI;
   extern PFNGLGETARRAYOBJECTIVATIPROC        glGetArrayObjectivATI;
   extern PFNGLVARIANTARRAYOBJECTATIPROC      glVariantArrayObjectATI;
   extern PFNGLGETVARIANTARRAYOBJECTFVATIPROC glGetVariantArrayObjectfvATI;
   extern PFNGLGETVARIANTARRAYOBJECTIVATIPROC glGetVariantArrayObjectivATI;
#endif

//====================//
// ATI_element_arrays //
//====================//
#ifdef GLEXT_PROTOTYPE
   int                                       ATI_element_array = 0;
   PFNGLELEMENTPOINTERATIPROC                glElementPointerATI = NULL;
   PFNGLDRAWELEMENTARRAYATIPROC              glDrawElementArrayATI = NULL;
   PFNGLDRAWRANGEELEMENTARRAYATIPROC         glDrawRangeElementArrayATI = NULL;
#else
   extern int                                ATI_element_array;
   extern PFNGLELEMENTPOINTERATIPROC         glElementPointerATI;
   extern PFNGLDRAWELEMENTARRAYATIPROC       glDrawElementArrayATI;
   extern PFNGLDRAWRANGEELEMENTARRAYATIPROC  glDrawRangeElementArrayATI;
#endif

//=========================//
// GL_ATI_separate_stencil //
//=========================//
#ifdef GLEXT_PROTOTYPE
   int                                       ATI_separate_stencil = 0;
   PFNGLSTENCILOPSEPARATEATIPROC             glStencilOpSeparateATI = NULL;
   PFNGLSTENCILFUNCSEPARATEATIPROC           glStencilFuncSeparateATI = NULL;
#else
   extern int                                ATI_separate_stencil;
   extern PFNGLSTENCILOPSEPARATEATIPROC      glStencilOpSeparateATI;
   extern PFNGLSTENCILFUNCSEPARATEATIPROC    glStencilFuncSeparateATI;
#endif

//=====================//
// GL_ATI_pn_triangles //
//=====================//
#ifdef GLEXT_PROTOTYPE
	int                                             ATI_pn_triangles = 0;
	PFNGLPNTRIANGLESIATIPROC						glPNTrianglesiATI = NULL;
	PFNGLPNTRIANGLESFATIPROC						glPNTrianglesfATI = NULL;
#else
	extern int                                      ATI_pn_triangles;
	extern PFNGLPNTRIANGLESIATIPROC					glPNTrianglesiATI;
	extern PFNGLPNTRIANGLESFATIPROC					glPNTrianglesfATI;
#endif

//====================//
// ATI_envmap_bumpmap //
//====================//
#ifdef GLEXT_PROTOTYPE
	int                                             ATI_envmap_bumpmap = 0;
	PFNGLTEXBUMPPARAMETERIVATIPROC					glTexBumpParameterivATI = NULL;
	PFNGLTEXBUMPPARAMETERFVATIPROC					glTexBumpParameterfvATI = NULL;
	PFNGLGETTEXBUMPPARAMETERIVATIPROC				glGetTexBumpParameterivATI = NULL;
	PFNGLGETTEXBUMPPARAMETERFVATIPROC				glGetTexBumpParameterfvATI = NULL;
#else
	extern int                                      ATI_envmap_bumpmap;
	extern PFNGLTEXBUMPPARAMETERIVATIPROC			glTexBumpParameterivATI;
	extern PFNGLTEXBUMPPARAMETERFVATIPROC			glTexBumpParameterfvATI;
	extern PFNGLGETTEXBUMPPARAMETERIVATIPROC		glGetTexBumpParameterivATI;
	extern PFNGLGETTEXBUMPPARAMETERFVATIPROC		glGetTexBumpParameterfvATI;
#endif

//========================//
// ATI_fragment_shader //
//========================//
#ifdef GLEXT_PROTOTYPE
   int                                             ATI_fragment_shader = 0;
   PFNGLGENFRAGMENTSHADERSATIPROC                  glGenFragmentShadersATI = NULL;
   PFNGLBINDFRAGMENTSHADERATIPROC                  glBindFragmentShaderATI = NULL;
   PFNGLDELETEFRAGMENTSHADERATIPROC                glDeleteFragmentShaderATI = NULL;
   PFNGLBEGINFRAGMENTSHADERATIPROC                 glBeginFragmentShaderATI = NULL;
   PFNGLENDFRAGMENTSHADERATIPROC                   glEndFragmentShaderATI = NULL;
   PFNGLPASSTEXCOORDATIPROC                        glPassTexCoordATI = NULL;
   PFNGLSAMPLEMAPATIPROC                           glSampleMapATI = NULL;
   PFNGLCOLORFRAGMENTOP1ATIPROC                    glColorFragmentOp1ATI = NULL;
   PFNGLCOLORFRAGMENTOP2ATIPROC                    glColorFragmentOp2ATI = NULL;
   PFNGLCOLORFRAGMENTOP3ATIPROC                    glColorFragmentOp3ATI = NULL;
   PFNGLALPHAFRAGMENTOP1ATIPROC                    glAlphaFragmentOp1ATI = NULL;
   PFNGLALPHAFRAGMENTOP2ATIPROC                    glAlphaFragmentOp2ATI = NULL;
   PFNGLALPHAFRAGMENTOP3ATIPROC                    glAlphaFragmentOp3ATI = NULL;
   PFNGLSETFRAGMENTSHADERCONSTANTATIPROC           glSetFragmentShaderConstantATI = NULL;
#else
   extern int                                      ATI_fragment_shader;
   extern PFNGLGENFRAGMENTSHADERSATIPROC           glGenFragmentShadersATI;
   extern PFNGLBINDFRAGMENTSHADERATIPROC           glBindFragmentShaderATI;
   extern PFNGLDELETEFRAGMENTSHADERATIPROC         glDeleteFragmentShaderATI;
   extern PFNGLBEGINFRAGMENTSHADERATIPROC          glBeginFragmentShaderATI;
   extern PFNGLENDFRAGMENTSHADERATIPROC            glEndFragmentShaderATI;
   extern PFNGLPASSTEXCOORDATIPROC                 glPassTexCoordATI;
   extern PFNGLSAMPLEMAPATIPROC                    glSampleMapATI;
   extern PFNGLCOLORFRAGMENTOP1ATIPROC             glColorFragmentOp1ATI;
   extern PFNGLCOLORFRAGMENTOP2ATIPROC             glColorFragmentOp2ATI;
   extern PFNGLCOLORFRAGMENTOP3ATIPROC             glColorFragmentOp3ATI;
   extern PFNGLALPHAFRAGMENTOP1ATIPROC             glAlphaFragmentOp1ATI;
   extern PFNGLALPHAFRAGMENTOP2ATIPROC             glAlphaFragmentOp2ATI;
   extern PFNGLALPHAFRAGMENTOP3ATIPROC             glAlphaFragmentOp3ATI;
   extern PFNGLSETFRAGMENTSHADERCONSTANTATIPROC    glSetFragmentShaderConstantATI;
#endif

//==========//
// NV_fence //
//==========//
#ifdef GLEXT_PROTOTYPE
	int								NV_fence = 0;
	PFNGLGENFENCESNVPROC			glGenFencesNV = NULL;
	PFNGLDELETEFENCESNVPROC			glDeleteFencesNV = NULL;
	PFNGLSETFENCENVPROC				glSetFenceNV = NULL;
	PFNGLTESTFENCENVPROC			glTestFenceNV = NULL;
	PFNGLFINISHFENCENVPROC			glFinishFenceNV = NULL;
	PFNGLISFENCENVPROC				glIsFenceNV = NULL;
	PFNGLGETFENCEIVNVPROC			glGetFenceivNV = NULL;
#else
	extern int							NV_fence;
	extern PFNGLGENFENCESNVPROC			glGenFencesNV;
	extern PFNGLDELETEFENCESNVPROC		glDeleteFencesNV;
	extern PFNGLSETFENCENVPROC			glSetFenceNV;
	extern PFNGLTESTFENCENVPROC			glTestFenceNV;
	extern PFNGLFINISHFENCENVPROC		glFinishFenceNV;
	extern PFNGLISFENCENVPROC			glIsFenceNV;
	extern PFNGLGETFENCEIVNVPROC		glGetFenceivNV;
#endif

//===============//
// NV_evaluators //
//===============//
#ifdef GLEXT_PROTOTYPE
	int										NV_evaluators = 0;
	PFNGLMAPCONTROLPOINTSNVPROC				glMapControlPointsNV = NULL;
	PFNGLMAPPARAMETERIVNVPROC				glMapParameterivNV = NULL;
	PFNGLMAPPARAMETERFVNVPROC				glMapParameterfvNV = NULL;
	PFNGLGETMAPCONTROLPOINTSNVPROC			glGetMapControlPointsNV = NULL;
	PFNGLGETMAPPARAMETERIVNVPROC			glGetMapParameterivNV = NULL;
	PFNGLGETMAPPARAMETERFVNVPROC			glGetMapParameterfvNV = NULL;
	PFNGLGETMAPATTRIBPARAMETERIVNVPROC		glGetMapAttribParameterivNV = NULL;
	PFNGLGETMAPATTRIBPARAMETERFVNVPROC		glGetMapAttribParameterfvNV = NULL;
	PFNGLEVALMAPSNVPROC						glEvalMapsNV = NULL;
#else
	extern int									NV_evaluators;
	extern PFNGLMAPCONTROLPOINTSNVPROC			glMapControlPointsNV;
	extern PFNGLMAPPARAMETERIVNVPROC			glMapParameterivNV;
	extern PFNGLMAPPARAMETERFVNVPROC			glMapParameterfvNV;
	extern PFNGLGETMAPCONTROLPOINTSNVPROC		glGetMapControlPointsNV;
	extern PFNGLGETMAPPARAMETERIVNVPROC			glGetMapParameterivNV;
	extern PFNGLGETMAPPARAMETERFVNVPROC			glGetMapParameterfvNV;
	extern PFNGLGETMAPATTRIBPARAMETERIVNVPROC	glGetMapAttribParameterivNV;
	extern PFNGLGETMAPATTRIBPARAMETERFVNVPROC	glGetMapAttribParameterfvNV;
	extern PFNGLEVALMAPSNVPROC					glEvalMapsNV;
#endif

//===================//
// NV_vertex_program //
//===================//
#ifdef GLEXT_PROTOTYPE
	int									NV_vertex_program = 0;
	PFNGLBINDPROGRAMNVPROC				glBindProgramNV = NULL;
	PFNGLDELETEPROGRAMSNVPROC			glDeleteProgramsNV = NULL;
	PFNGLEXECUTEPROGRAMNVPROC			glExecuteProgramNV = NULL;
	PFNGLGENPROGRAMSNVPROC				glGenProgramsNV = NULL;
	PFNGLAREPROGRAMSRESIDENTNVPROC		glAreProgramsResidentNV = NULL;
	PFNGLREQUESTRESIDENTPROGRAMSNVPROC	glRequestResidentProgramsNV = NULL;
	PFNGLGETPROGRAMPARAMETERDVNVPROC	glGetProgramParameterdvNV = NULL;
	PFNGLGETPROGRAMPARAMETERFVNVPROC	glGetProgramParameterfvNV = NULL;
	PFNGLGETPROGRAMIVNVPROC				glGetProgramivNV = NULL;
	PFNGLGETPROGRAMSTRINGNVPROC			glGetProgramStringNV = NULL;
	PFNGLGETVERTEXATTRIBDVNVPROC		glGetVertexAttribdvNV = NULL;
	PFNGLGETVERTEXATTRIBFVNVPROC		glGetVertexAttribfvNV = NULL;
	PFNGLGETVERTEXATTRIBIVNVPROC		glGetVertexAttribivNV = NULL;
	PFNGLGETVERTEXATTRIBPOINTERVNVPROC	glGetVertexAttribPointervNV = NULL;
	PFNGLISPROGRAMNVPROC				glIsProgramNV = NULL;
	PFNGLLOADPROGRAMNVPROC				glLoadProgramNV = NULL;
	PFNGLPROGRAMPARAMETER4DNVPROC		glProgramParameter4dNV = NULL;
	PFNGLPROGRAMPARAMETER4DVNVPROC		glProgramParameter4dvNV = NULL;
	PFNGLPROGRAMPARAMETER4FNVPROC		glProgramParameter4fNV = NULL;
	PFNGLPROGRAMPARAMETER4FVNVPROC		glProgramParameter4fvNV = NULL;
	PFNGLPROGRAMPARAMETERS4DVNVPROC		glProgramParameters4dvNV = NULL;
	PFNGLPROGRAMPARAMETERS4FVNVPROC		glProgramParameters4fvNV = NULL;
	PFNGLTRACKMATRIXNVPROC				glTrackMatrixNV = NULL;
	PFNGLGETTRACKMATRIXIVNVPROC			glGetTrackMatrixNV = NULL;
	PFNGLVERTEXATTRIBPOINTERNVPROC		glVertexAttribPointerNV = NULL;
	PFNGLVERTEXATTRIB1DNVPROC			glVertexAttrib1dNV = NULL;
	PFNGLVERTEXATTRIB1DVNVPROC			glVertexAttrib1dvNV = NULL;
	PFNGLVERTEXATTRIB1FNVPROC			glVertexAttrib1fNV = NULL;
	PFNGLVERTEXATTRIB1FVNVPROC			glVertexAttrib1fvNV = NULL;
	PFNGLVERTEXATTRIB1SNVPROC			glVertexAttrib1sNV = NULL;
	PFNGLVERTEXATTRIB1SVNVPROC			glVertexAttrib1svNV = NULL;
	PFNGLVERTEXATTRIB2DNVPROC			glVertexAttrib2dNV = NULL;
	PFNGLVERTEXATTRIB2DVNVPROC			glVertexAttrib2dvNV = NULL;
	PFNGLVERTEXATTRIB2FNVPROC			glVertexAttrib2fNV = NULL;
	PFNGLVERTEXATTRIB2FVNVPROC			glVertexAttrib2fvNV = NULL;
	PFNGLVERTEXATTRIB2SNVPROC			glVertexAttrib2sNV = NULL;
	PFNGLVERTEXATTRIB2SVNVPROC			glVertexAttrib2svNV = NULL;
	PFNGLVERTEXATTRIB3DNVPROC			glVertexAttrib3dNV = NULL;
	PFNGLVERTEXATTRIB3DVNVPROC			glVertexAttrib3dvNV = NULL;
	PFNGLVERTEXATTRIB3FNVPROC			glVertexAttrib3fNV = NULL;
	PFNGLVERTEXATTRIB3FVNVPROC			glVertexAttrib3fvNV = NULL;
	PFNGLVERTEXATTRIB3SNVPROC			glVertexAttrib3sNV = NULL;
	PFNGLVERTEXATTRIB3SVNVPROC			glVertexAttrib3svNV = NULL;
	PFNGLVERTEXATTRIB4DNVPROC			glVertexAttrib4dNV = NULL;
	PFNGLVERTEXATTRIB4DVNVPROC			glVertexAttrib4fNV = NULL;
	PFNGLVERTEXATTRIB4FNVPROC			glVertexAttrib4sNV = NULL;
	PFNGLVERTEXATTRIB4FVNVPROC			glVertexAttrib4dvNV = NULL;
	PFNGLVERTEXATTRIB4SNVPROC			glVertexAttrib4fvNV = NULL;
	PFNGLVERTEXATTRIB4SVNVPROC			glVertexAttrib4svNV = NULL;
	PFNGLVERTEXATTRIB4UBNVPROC			glVertexAttrib4ubNV = NULL;
	PFNGLVERTEXATTRIB4UBVNVPROC			glVertexAttrib4ubvNV = NULL;
	PFNGLVERTEXATTRIBS1DVNVPROC			glVertexAttribs1dvNV = NULL;
	PFNGLVERTEXATTRIBS1FVNVPROC			glVertexAttribs1fvNV = NULL;
	PFNGLVERTEXATTRIBS1SVNVPROC			glVertexAttribs1svNV = NULL;
	PFNGLVERTEXATTRIBS2DVNVPROC			glVertexAttribs2dvNV = NULL;
	PFNGLVERTEXATTRIBS2FVNVPROC			glVertexAttribs2fvNV = NULL;
	PFNGLVERTEXATTRIBS2SVNVPROC			glVertexAttribs2svNV = NULL;
	PFNGLVERTEXATTRIBS3DVNVPROC			glVertexAttribs3dvNV = NULL;
	PFNGLVERTEXATTRIBS3FVNVPROC			glVertexAttribs3fvNV = NULL;
	PFNGLVERTEXATTRIBS3SVNVPROC			glVertexAttribs3svNV = NULL;
	PFNGLVERTEXATTRIBS4DVNVPROC			glVertexAttribs4dvNV = NULL;
	PFNGLVERTEXATTRIBS4FVNVPROC			glVertexAttribs4fvNV = NULL;
	PFNGLVERTEXATTRIBS4SVNVPROC			glVertexAttribs4svNV = NULL;
	PFNGLVERTEXATTRIBS4UBVNVPROC		glVertexAttribs4ubvNV = NULL;
#else
	extern int									NV_vertex_program;
	extern PFNGLBINDPROGRAMNVPROC				glBindProgramNV;
	extern PFNGLDELETEPROGRAMSNVPROC			glDeleteProgramsNV;
	extern PFNGLEXECUTEPROGRAMNVPROC			glExecuteProgramNV;
	extern PFNGLGENPROGRAMSNVPROC				glGenProgramsNV;
	extern PFNGLAREPROGRAMSRESIDENTNVPROC		glAreProgramsResidentNV;
	extern PFNGLREQUESTRESIDENTPROGRAMSNVPROC	glRequestResidentProgramsNV;
	extern PFNGLGETPROGRAMPARAMETERDVNVPROC		glGetProgramParameterdvNV;
	extern PFNGLGETPROGRAMPARAMETERFVNVPROC		glGetProgramParameterfvNV;
	extern PFNGLGETPROGRAMIVNVPROC				glGetProgramivNV;
	extern PFNGLGETPROGRAMSTRINGNVPROC			glGetProgramStringNV;
	extern PFNGLGETVERTEXATTRIBDVNVPROC			glGetVertexAttribdvNV;
	extern PFNGLGETVERTEXATTRIBFVNVPROC			glGetVertexAttribfvNV;
	extern PFNGLGETVERTEXATTRIBIVNVPROC			glGetVertexAttribivNV;
	extern PFNGLGETVERTEXATTRIBPOINTERVNVPROC	glGetVertexAttribPointervNV;
	extern PFNGLISPROGRAMNVPROC					glIsProgramNV;
	extern PFNGLLOADPROGRAMNVPROC				glLoadProgramNV;
	extern PFNGLPROGRAMPARAMETER4DNVPROC		glProgramParameter4dNV;
	extern PFNGLPROGRAMPARAMETER4DVNVPROC		glProgramParameter4dvNV;
	extern PFNGLPROGRAMPARAMETER4FNVPROC		glProgramParameter4fNV;
	extern PFNGLPROGRAMPARAMETER4FVNVPROC		glProgramParameter4fvNV;
	extern PFNGLPROGRAMPARAMETERS4DVNVPROC		glProgramParameters4dvNV;
	extern PFNGLPROGRAMPARAMETERS4FVNVPROC		glProgramParameters4fvNV;
	extern PFNGLTRACKMATRIXNVPROC				glTrackMatrixNV;
	extern PFNGLGETTRACKMATRIXIVNVPROC			glGetTrackMatrixNV;
	extern PFNGLVERTEXATTRIBPOINTERNVPROC		glVertexAttribPointerNV;
	extern PFNGLVERTEXATTRIB1DNVPROC			glVertexAttrib1dNV;
	extern PFNGLVERTEXATTRIB1DVNVPROC			glVertexAttrib1dvNV;
	extern PFNGLVERTEXATTRIB1FNVPROC			glVertexAttrib1fNV;
	extern PFNGLVERTEXATTRIB1FVNVPROC			glVertexAttrib1fvNV;
	extern PFNGLVERTEXATTRIB1SNVPROC			glVertexAttrib1sNV;
	extern PFNGLVERTEXATTRIB1SVNVPROC			glVertexAttrib1svNV;
	extern PFNGLVERTEXATTRIB2DNVPROC			glVertexAttrib2dNV;
	extern PFNGLVERTEXATTRIB2DVNVPROC			glVertexAttrib2dvNV;
	extern PFNGLVERTEXATTRIB2FNVPROC			glVertexAttrib2fNV;
	extern PFNGLVERTEXATTRIB2FVNVPROC			glVertexAttrib2fvNV;
	extern PFNGLVERTEXATTRIB2SNVPROC			glVertexAttrib2sNV;
	extern PFNGLVERTEXATTRIB2SVNVPROC			glVertexAttrib2svNV;
	extern PFNGLVERTEXATTRIB3DNVPROC			glVertexAttrib3dNV;
	extern PFNGLVERTEXATTRIB3DVNVPROC			glVertexAttrib3dvNV;
	extern PFNGLVERTEXATTRIB3FNVPROC			glVertexAttrib3fNV;
	extern PFNGLVERTEXATTRIB3FVNVPROC			glVertexAttrib3fvNV;
	extern PFNGLVERTEXATTRIB3SNVPROC			glVertexAttrib3sNV;
	extern PFNGLVERTEXATTRIB3SVNVPROC			glVertexAttrib3svNV;
	extern PFNGLVERTEXATTRIB4DNVPROC			glVertexAttrib4dNV;
	extern PFNGLVERTEXATTRIB4DVNVPROC			glVertexAttrib4fNV;
	extern PFNGLVERTEXATTRIB4FNVPROC			glVertexAttrib4sNV;
	extern PFNGLVERTEXATTRIB4FVNVPROC			glVertexAttrib4dvNV;
	extern PFNGLVERTEXATTRIB4SNVPROC			glVertexAttrib4fvNV;
	extern PFNGLVERTEXATTRIB4SVNVPROC			glVertexAttrib4svNV;
	extern PFNGLVERTEXATTRIB4UBNVPROC			glVertexAttrib4ubNV;
	extern PFNGLVERTEXATTRIB4UBVNVPROC			glVertexAttrib4ubvNV;
	extern PFNGLVERTEXATTRIBS1DVNVPROC			glVertexAttribs1dvNV;
	extern PFNGLVERTEXATTRIBS1FVNVPROC			glVertexAttribs1fvNV;
	extern PFNGLVERTEXATTRIBS1SVNVPROC			glVertexAttribs1svNV;
	extern PFNGLVERTEXATTRIBS2DVNVPROC			glVertexAttribs2dvNV;
	extern PFNGLVERTEXATTRIBS2FVNVPROC			glVertexAttribs2fvNV;
	extern PFNGLVERTEXATTRIBS2SVNVPROC			glVertexAttribs2svNV;
	extern PFNGLVERTEXATTRIBS3DVNVPROC			glVertexAttribs3dvNV;
	extern PFNGLVERTEXATTRIBS3FVNVPROC			glVertexAttribs3fvNV;
	extern PFNGLVERTEXATTRIBS3SVNVPROC			glVertexAttribs3svNV;
	extern PFNGLVERTEXATTRIBS4DVNVPROC			glVertexAttribs4dvNV;
	extern PFNGLVERTEXATTRIBS4FVNVPROC			glVertexAttribs4fvNV;
	extern PFNGLVERTEXATTRIBS4SVNVPROC			glVertexAttribs4svNV;
	extern PFNGLVERTEXATTRIBS4UBVNVPROC			glVertexAttribs4ubvNV;
#endif

//=======================//
// NV_register_combiners //
//=======================//
#ifdef GLEXT_PROTOTYPE
	int												NV_register_combiners = 0;
	PFNGLCOMBINERPARAMETERFVNVPROC					glCombinerParameterfvNV = NULL;
	PFNGLCOMBINERPARAMETERIVNVPROC					glCombinerParameterivNV = NULL;
	PFNGLCOMBINERPARAMETERFNVPROC					glCombinerParameterfNV = NULL;
	PFNGLCOMBINERPARAMETERINVPROC					glCombinerParameteriNV = NULL;
	PFNGLCOMBINERINPUTNVPROC						glCombinerInputNV = NULL;
	PFNGLCOMBINEROUTPUTNVPROC						glCombinerOutputNV = NULL;
	PFNGLFINALCOMBINERINPUTNVPROC					glFinalCombinerInputNV = NULL;
	PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC			glGetCombinerInputParameterfvNV = NULL;
	PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC			glGetCombinerInputParameterivNV = NULL;
	PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC			glGetCombinerOutputParameterfvNV = NULL;
	PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC			glGetCombinerOutputParameterivNV = NULL;
	PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC		glGetFinalCombinerInputParameterfvNV = NULL;
	PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC		glGetFinalCombinerInputParameterivNV = NULL;
#else
	extern int												NV_register_combiners;
	extern PFNGLCOMBINERPARAMETERFVNVPROC					glCombinerParameterfvNV;
	extern PFNGLCOMBINERPARAMETERIVNVPROC					glCombinerParameterivNV;
	extern PFNGLCOMBINERPARAMETERFNVPROC					glCombinerParameterfNV;
	extern PFNGLCOMBINERPARAMETERINVPROC					glCombinerParameteriNV;
	extern PFNGLCOMBINERINPUTNVPROC							glCombinerInputNV;
	extern PFNGLCOMBINEROUTPUTNVPROC						glCombinerOutputNV;
	extern PFNGLFINALCOMBINERINPUTNVPROC					glFinalCombinerInputNV;
	extern PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC			glGetCombinerInputParameterfvNV;
	extern PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC			glGetCombinerInputParameterivNV;
	extern PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC			glGetCombinerOutputParameterfvNV;
	extern PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC			glGetCombinerOutputParameterivNV;
	extern PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC		glGetFinalCombinerInputParameterfvNV;
	extern PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC		glGetFinalCombinerInputParameterivNV;
#endif

//========================//
// NV_register_combiners2 //
//========================//
#ifdef GLEXT_PROTOTYPE
	int												NV_register_combiners2 = 0;
	PFNGLCOMBINERSTAGEPARAMETERFVNVPROC				glCombinerStageParameterfvNV = NULL;
	PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC			glGetCombinerStageParameterfvNV = NULL;
#else
	extern int										NV_register_combiners2;
	extern PFNGLCOMBINERSTAGEPARAMETERFVNVPROC		glCombinerStageParameterfvNV;
	extern PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC	glGetCombinerStageParameterfvNV;
#endif

//=======================//
// NV_vertex_array_range //
//=======================//
#ifdef GLEXT_PROTOTYPE
	int									NV_vertex_array_range = 0;
	PFNGLFLUSHVERTEXARRAYRANGENVPROC	glFlushVertexArrayRangeNV = NULL;
	PFNGLVERTEXARRAYRANGENVPROC			glVertexArrayRangeNV = NULL;
#else
	extern int								NV_vertex_array_range;
	extern PFNGLFLUSHVERTEXARRAYRANGENVPROC	glFlushVertexArrayRangeNV;
	extern PFNGLVERTEXARRAYRANGENVPROC		glVertexArrayRangeNV;
#endif

#ifdef GLEXT_PROTOTYPE
//================================================================//
// IsExtensionSupported                                           //
// checks the opengl extension string for the specified extension //
//================================================================//
int IsExtensionSupported(const char *extension)
{
	const GLubyte *extensions = NULL;
	const GLubyte *start;
	GLubyte *where, *terminator;

	// Extension names should not have spaces.
	where = (GLubyte *) strchr(extension, ' ');
	if (where || *extension == '\0')
	{
		return 0;
	}
	extensions = glGetString(GL_EXTENSIONS);
	// It takes a bit of care to be fool-proof about parsing the
	// OpenGL extensions string. Don't be fooled by sub-strings,
	// etc.
	start = extensions;
	for (;;) 
	{
		where = (GLubyte *) strstr((const char *) start, extension);
		if (!where)
		{
			break;
		}
		terminator = where + strlen(extension);
		if (where == start || *(where - 1) == ' ')
		{
			if (*terminator == ' ' || *terminator == '\0')
			{
				return 1;
			}
		}
		start = terminator;
	}
	return 0;
}

void InitGLextensions(void)
{
	ARB_texture_cube_map = IsExtensionSupported("GL_ARB_texture_cube_map");   
	ARB_texture_border_clamp = IsExtensionSupported("GL_ARB_texture_border_clamp");
	ARB_texture_env_add = IsExtensionSupported("GL_ARB_texture_env_add");
	ARB_texture_env_combine = IsExtensionSupported("GL_ARB_texture_env_combine");
	ARB_texture_env_crossbar = IsExtensionSupported("GL_ARB_texture_env_crossbar");
	ARB_texture_env_dot3 = IsExtensionSupported("GL_ARB_texture_env_dot3");
	ARB_transpose_matrix = IsExtensionSupported("GL_ARB_transpose_matrix");
	ATI_texture_mirror_once = IsExtensionSupported("GL_ATI_texture_mirror_once");
	NV_texgen_reflection = IsExtensionSupported("GL_NV_texgen_reflection");
	NV_blend_square = IsExtensionSupported("GL_NV_blend_square");
	NV_fog_distance = IsExtensionSupported("GL_NV_fog_distance");
	NV_texgen_emboss = IsExtensionSupported("GL_NV_texgen_emboss");
	NV_texture_env_combine4 = IsExtensionSupported("GL_NV_texture_env_combine4");
	NV_texture_shader = IsExtensionSupported("GL_NV_texture_shader");
	NV_texture_shader2 = IsExtensionSupported("GL_NV_texture_shader2");
	NV_texture_rectangle = IsExtensionSupported("GL_NV_texture_rectangle");
	NV_vertex_array_range2 = IsExtensionSupported("GL_NV_vertex_array_range2");
	SGIS_generate_mipmap = IsExtensionSupported("GL_SGIS_generate_mipmap");
	
	ARB_texture_compression = IsExtensionSupported("GL_ARB_texture_compression");
	if (ARB_texture_compression == TRUE)
	{
		glCompressedTexImage3DARB = (PFNGLCOMPRESSEDTEXIMAGE3DARBPROC)wglGetProcAddress("glCompressedTexImage3DARB ");
		glCompressedTexImage2DARB = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)wglGetProcAddress("glCompressedTexImage2DARB");
		glCompressedTexImage1DARB = (PFNGLCOMPRESSEDTEXIMAGE1DARBPROC)wglGetProcAddress("glCompressedTexImage1DARB");
		glCompressedTexSubImage3DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC)wglGetProcAddress("glCompressedTexSubImage3DARB");
		glCompressedTexSubImage2DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC)wglGetProcAddress("glCompressedTexSubImage2DARB");
		glCompressedTexSubImage1DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC)wglGetProcAddress("glCompressedTexSubImage1DARB");
		glGetCompressedTexImageARB = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)wglGetProcAddress("glGetCompressedTexImageARB");
	}

	ARB_multisample = IsExtensionSupported("GL_ARB_multisample");   
	if (ARB_multisample == TRUE)
	{
		glSampleCoverageARB = (PFNGLSAMPLECOVERAGEARBPROC)wglGetProcAddress("glFogCoordfEXT");
	}

	ARB_matrix_palette = IsExtensionSupported("ARB_matrix_palette");   
	if (ARB_matrix_palette == TRUE)
	{
		glCurrentPaletteMatrixARB = (PFNGLCURRENTPALETTEMATRIXARBPROC)wglGetProcAddress("glCurrentPaletteMatrixARB");
		glMatrixIndexubvARB = (PFNGLMATRIXINDEXUBVARBPROC)wglGetProcAddress("glMatrixIndexubvARB");
		glMatrixIndexusvARB = (PFNGLMATRIXINDEXUSVARBPROC)wglGetProcAddress("glMatrixIndexusvARB");
		glMatrixIndexuivARB = (PFNGLMATRIXINDEXUIVARBPROC)wglGetProcAddress("glMatrixIndexuivARB");
		glMatrixIndexPointerARB = (PFNGLMATRIXINDEXPOINTERARBPROC)wglGetProcAddress("glMatrixIndexPointerARB");
	}

	EXT_compiled_vertex_array = IsExtensionSupported("GL_EXT_compiled_vertex_array");   
	if (EXT_compiled_vertex_array == TRUE)
	{
		glLockArraysEXT = (PFNGLLOCKARRAYSEXTPROC)wglGetProcAddress("glLockArraysEXT");
		glUnlockArraysEXT = (PFNGLUNLOCKARRAYSEXTPROC)wglGetProcAddress("glUnlockArraysEXT");
	}

	EXT_fog_coord=IsExtensionSupported("GL_EXT_fog_coord");
	if (EXT_fog_coord == TRUE)
	{
		glFogCoordfEXT = (PFNGLFOGCOORDFEXTPROC)wglGetProcAddress("glFogCoordfEXT");
		glFogCoorddEXT = (PFNGLFOGCOORDDEXTPROC)wglGetProcAddress("glFogCoorddEXT");
		glFogCoordfvEXT = (PFNGLFOGCOORDFVEXTPROC)wglGetProcAddress("glFogCoordfvEXT");
		glFogCoorddvEXT = (PFNGLFOGCOORDDVEXTPROC)wglGetProcAddress("glFogCoorddvEXT");
		glFogCoordPointerEXT = (PFNGLFOGCOORDPOINTEREXTPROC)wglGetProcAddress("glFogCoordPointerEXT");
	}

	ATI_pn_triangles = IsExtensionSupported("GL_ATI_pn_triangles");
	{
		glPNTrianglesiATI = (PFNGLPNTRIANGLESIATIPROC)wglGetProcAddress("glPNTrianglesiATI");
		glPNTrianglesfATI = (PFNGLPNTRIANGLESFATIPROC)wglGetProcAddress("glPNTrianglesfATI");
	}

	ATI_envmap_bumpmap = IsExtensionSupported("GL_ATI_envmap_bumpmap");
	if (ATI_envmap_bumpmap == TRUE)
	{
		glTexBumpParameterivATI = (PFNGLTEXBUMPPARAMETERIVATIPROC)wglGetProcAddress("glTexBumpParameterivATI");
		glTexBumpParameterfvATI = (PFNGLTEXBUMPPARAMETERFVATIPROC)wglGetProcAddress("glTexBumpParameterfvATI");
		glGetTexBumpParameterivATI = (PFNGLGETTEXBUMPPARAMETERIVATIPROC)wglGetProcAddress("glGetTexBumpParameterivATI");
		glGetTexBumpParameterfvATI = (PFNGLGETTEXBUMPPARAMETERFVATIPROC)wglGetProcAddress("glGetTexBumpParameterfvATI");
	}

	ATI_vertex_array_object = IsExtensionSupported("GL_ATI_vertex_array_object");
	if (ATI_vertex_array_object == TRUE)
	{
		glNewObjectBufferATI = (PFNGLNEWOBJECTBUFFERATIPROC)wglGetProcAddress("glNewObjectBufferATI");
		glIsObjectBufferATI = (PFNGLISOBJECTBUFFERATIPROC)wglGetProcAddress("glIsObjectBufferATI");
		glUpdateObjectBufferATI = (PFNGLUPDATEOBJECTBUFFERATIPROC)wglGetProcAddress("glUpdateObjectBufferATI");
		glGetObjectBufferfvATI = (PFNGLGETOBJECTBUFFERFVATIPROC)wglGetProcAddress("glGetObjectBufferfvATI");
		glGetObjectBufferivATI = (PFNGLGETOBJECTBUFFERIVATIPROC)wglGetProcAddress("glGetObjectBufferivATI");
		glFreeObjectBufferATI = (PFNGLFREEOBJECTBUFFERATIPROC)wglGetProcAddress("glFreeObjectBufferATI");
		glArrayObjectATI = (PFNGLARRAYOBJECTATIPROC)wglGetProcAddress("glArrayObjectATI");
		glGetArrayObjectfvATI = (PFNGLGETARRAYOBJECTFVATIPROC)wglGetProcAddress("glGetArrayObjectfvATI");
		glGetArrayObjectivATI = (PFNGLGETARRAYOBJECTIVATIPROC)wglGetProcAddress("glGetArrayObjectivATI");
		glVariantArrayObjectATI = (PFNGLVARIANTARRAYOBJECTATIPROC)wglGetProcAddress("glVariantArrayObjectATI");
		glGetVariantArrayObjectfvATI = (PFNGLGETVARIANTARRAYOBJECTFVATIPROC)wglGetProcAddress("glGetVariantArrayObjectfvATI");
		glGetVariantArrayObjectivATI = (PFNGLGETVARIANTARRAYOBJECTIVATIPROC)wglGetProcAddress("glGetVariantArrayObjectivATI");
	}

	ATI_element_array = IsExtensionSupported("GL_ATI_element_array");
	if (ATI_element_array == TRUE)
	{
		glElementPointerATI = (PFNGLELEMENTPOINTERATIPROC)wglGetProcAddress("glElementArrayPointerATI");
		glDrawElementArrayATI = (PFNGLDRAWELEMENTARRAYATIPROC)wglGetProcAddress("glDrawElementArrayATI");
		glDrawRangeElementArrayATI = (PFNGLDRAWRANGEELEMENTARRAYATIPROC)wglGetProcAddress("glDrawRangeElementArrayATI");
	}

	ATI_separate_stencil = IsExtensionSupported("GL_ATI_separate_stencil");
	if (ATI_separate_stencil == TRUE)
	{
		glStencilOpSeparateATI = (PFNGLSTENCILOPSEPARATEATIPROC)wglGetProcAddress("glStencilOpSeparateATI");
		glStencilFuncSeparateATI = (PFNGLSTENCILFUNCSEPARATEATIPROC)wglGetProcAddress("glStencilFuncSeparateATI");
	}

	ARB_vertex_blend = IsExtensionSupported("GL_ARB_vertex_blend");
	if (ARB_vertex_blend == TRUE) 
	{
		glVertexBlendARB = (PFNGLVERTEXBLENDARBPROC)wglGetProcAddress("glVertexBlendARB");
		glWeightPointerARB = (PFNGLWEIGHTPOINTERARBPROC)wglGetProcAddress("glWeightPointerARB");
		glWeightbvARB = (PFNGLWEIGHTBVARBPROC)wglGetProcAddress("glWeightbvARB");
		glWeightsvARB = (PFNGLWEIGHTSVARBPROC)wglGetProcAddress("glWeightsvARB");
		glWeightivARB = (PFNGLWEIGHTIVARBPROC)wglGetProcAddress("glWeightivARB");
		glWeightfvARB = (PFNGLWEIGHTFVARBPROC)wglGetProcAddress("glWeightfvARB");
		glWeightdvARB = (PFNGLWEIGHTDVARBPROC)wglGetProcAddress("glWeightdvARB");
		glWeightubvARB = (PFNGLWEIGHTUBVARBPROC)wglGetProcAddress("glWeightubvARB");
		glWeightusvARB = (PFNGLWEIGHTUSVARBPROC)wglGetProcAddress("glWeightusvARB");
		glWeightuivARB = (PFNGLWEIGHTUIVARBPROC)wglGetProcAddress("glWeightuivARB");
	}

	EXT_texture3D = IsExtensionSupported("GL_EXT_texture3D");
	if (EXT_texture3D == TRUE)
	{
		glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC)wglGetProcAddress("glTexImage3DEXT");
		glTexSubImage3DEXT = (PFNGLTEXSUBIMAGE3DPROC)wglGetProcAddress("glTexSubImage3DEXT");
		glCopyTexSubImage3DEXT = (PFNGLCOPYTEXSUBIMAGE3DPROC)wglGetProcAddress("glCopyTexSubImage3DEXT");
	}

	ATI_fragment_shader = IsExtensionSupported("GL_ATI_fragment_shader");
	if (ATI_fragment_shader == TRUE)
	{
		glGenFragmentShadersATI = (PFNGLGENFRAGMENTSHADERSATIPROC)wglGetProcAddress("glGenFragmentShadersATI");
		glBindFragmentShaderATI = (PFNGLBINDFRAGMENTSHADERATIPROC)wglGetProcAddress("glBindFragmentShaderATI");
		glDeleteFragmentShaderATI = (PFNGLDELETEFRAGMENTSHADERATIPROC)wglGetProcAddress("glDeleteFragmentShaderATI");
		glBeginFragmentShaderATI = (PFNGLBEGINFRAGMENTSHADERATIPROC)wglGetProcAddress("glBeginFragmentShaderATI");
		glEndFragmentShaderATI = (PFNGLENDFRAGMENTSHADERATIPROC)wglGetProcAddress("glEndFragmentShaderATI");
		glPassTexCoordATI = (PFNGLPASSTEXCOORDATIPROC)wglGetProcAddress("glPassTexCoordATI");
		glSampleMapATI = (PFNGLSAMPLEMAPATIPROC)wglGetProcAddress("glSampleMapATI");
		glColorFragmentOp1ATI = (PFNGLCOLORFRAGMENTOP1ATIPROC)wglGetProcAddress("glColorFragmentOp1ATI");
		glColorFragmentOp2ATI = (PFNGLCOLORFRAGMENTOP2ATIPROC)wglGetProcAddress("glColorFragmentOp2ATI");
		glColorFragmentOp3ATI = (PFNGLCOLORFRAGMENTOP3ATIPROC)wglGetProcAddress("glColorFragmentOp3ATI");
		glAlphaFragmentOp1ATI = (PFNGLALPHAFRAGMENTOP1ATIPROC)wglGetProcAddress("glAlphaFragmentOp1ATI");
		glAlphaFragmentOp2ATI = (PFNGLALPHAFRAGMENTOP2ATIPROC)wglGetProcAddress("glAlphaFragmentOp2ATI");
		glAlphaFragmentOp3ATI = (PFNGLALPHAFRAGMENTOP3ATIPROC)wglGetProcAddress("glAlphaFragmentOp3ATI");
		glSetFragmentShaderConstantATI = (PFNGLSETFRAGMENTSHADERCONSTANTATIPROC)wglGetProcAddress("glSetFragmentShaderConstantATI");
	}

	ARB_multitexture = IsExtensionSupported("GL_ARB_multitexture");
	if (ARB_multitexture == TRUE)
	{
		glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)wglGetProcAddress("glActiveTextureARB");
		glClientActiveTextureARB = (PFNGLCLIENTACTIVETEXTUREARBPROC)wglGetProcAddress("glClientActiveTextureARB");
		glMultiTexCoord1dARB = (PFNGLMULTITEXCOORD1DARBPROC)wglGetProcAddress("glMultiTexCoord1dARB");
		glMultiTexCoord1dvARB = (PFNGLMULTITEXCOORD1DVARBPROC)wglGetProcAddress("glMultiTexCoord1dvARB");
		glMultiTexCoord1fARB = (PFNGLMULTITEXCOORD1FARBPROC)wglGetProcAddress("glMultiTexCoord1fARB");
		glMultiTexCoord1fvARB = (PFNGLMULTITEXCOORD1FVARBPROC)wglGetProcAddress("glMultiTexCoord1fvARB");
		glMultiTexCoord1iARB = (PFNGLMULTITEXCOORD1IARBPROC)wglGetProcAddress("glMultiTexCoord1iARB");
		glMultiTexCoord1ivARB = (PFNGLMULTITEXCOORD1IVARBPROC)wglGetProcAddress("glMultiTexCoord1ivARB");
		glMultiTexCoord1sARB = (PFNGLMULTITEXCOORD1SARBPROC)wglGetProcAddress("glMultiTexCoord1sARB");
		glMultiTexCoord1svARB = (PFNGLMULTITEXCOORD1SVARBPROC)wglGetProcAddress("glMultiTexCoord1svARB");
		glMultiTexCoord2dARB = (PFNGLMULTITEXCOORD2DARBPROC)wglGetProcAddress("glMultiTexCoord2dARB");
		glMultiTexCoord2dvARB = (PFNGLMULTITEXCOORD2DVARBPROC)wglGetProcAddress("glMultiTexCoord2dvARB");
		glMultiTexCoord2fARB = (PFNGLMULTITEXCOORD2FARBPROC)wglGetProcAddress("glMultiTexCoord2fARB");
		glMultiTexCoord2fvARB = (PFNGLMULTITEXCOORD2FVARBPROC)wglGetProcAddress("glMultiTexCoord2fvARB");
		glMultiTexCoord2iARB = (PFNGLMULTITEXCOORD2IARBPROC)wglGetProcAddress("glMultiTexCoord2iARB");
		glMultiTexCoord2ivARB = (PFNGLMULTITEXCOORD2IVARBPROC)wglGetProcAddress("glMultiTexCoord2ivARB");
		glMultiTexCoord2sARB = (PFNGLMULTITEXCOORD2SARBPROC)wglGetProcAddress("glMultiTexCoord2sARB");
		glMultiTexCoord2svARB = (PFNGLMULTITEXCOORD2SVARBPROC)wglGetProcAddress("glMultiTexCoord2svARB");
		glMultiTexCoord3dARB = (PFNGLMULTITEXCOORD3DARBPROC)wglGetProcAddress("glMultiTexCoord3dARB");
		glMultiTexCoord3dvARB = (PFNGLMULTITEXCOORD3DVARBPROC)wglGetProcAddress("glMultiTexCoord3dvARB");
		glMultiTexCoord3fARB = (PFNGLMULTITEXCOORD3FARBPROC)wglGetProcAddress("glMultiTexCoord3fARB");
		glMultiTexCoord3fvARB = (PFNGLMULTITEXCOORD3FVARBPROC)wglGetProcAddress("glMultiTexCoord3fvARB");
		glMultiTexCoord3iARB = (PFNGLMULTITEXCOORD3IARBPROC)wglGetProcAddress("glMultiTexCoord3iARB");
		glMultiTexCoord3ivARB = (PFNGLMULTITEXCOORD3IVARBPROC)wglGetProcAddress("glMultiTexCoord3ivARB");
		glMultiTexCoord3sARB = (PFNGLMULTITEXCOORD3SARBPROC)wglGetProcAddress("glMultiTexCoord3sARB");
		glMultiTexCoord3svARB = (PFNGLMULTITEXCOORD3SVARBPROC)wglGetProcAddress("glMultiTexCoord3svARB");
		glMultiTexCoord4dARB = (PFNGLMULTITEXCOORD4DARBPROC)wglGetProcAddress("glMultiTexCoord4dARB");
		glMultiTexCoord4dvARB = (PFNGLMULTITEXCOORD4DVARBPROC)wglGetProcAddress("glMultiTexCoord4dvARB");
		glMultiTexCoord4fARB = (PFNGLMULTITEXCOORD4FARBPROC)wglGetProcAddress("glMultiTexCoord4fARB");
		glMultiTexCoord4fvARB = (PFNGLMULTITEXCOORD4FVARBPROC)wglGetProcAddress("glMultiTexCoord4fvARB");
		glMultiTexCoord4iARB = (PFNGLMULTITEXCOORD4IARBPROC)wglGetProcAddress("glMultiTexCoord4iARB");
		glMultiTexCoord4ivARB = (PFNGLMULTITEXCOORD4IVARBPROC)wglGetProcAddress("glMultiTexCoord4ivARB");
		glMultiTexCoord4sARB = (PFNGLMULTITEXCOORD4SARBPROC)wglGetProcAddress("glMultiTexCoord4sARB");
		glMultiTexCoord4svARB = (PFNGLMULTITEXCOORD4SVARBPROC)wglGetProcAddress("glMultiTexCoord4svARB");
	}

	NV_fence = IsExtensionSupported("GL_NV_fence");
	if (NV_fence == TRUE)
	{
		glGenFencesNV = (PFNGLGENFENCESNVPROC)wglGetProcAddress("glGenFencesNV");
		glDeleteFencesNV = (PFNGLDELETEFENCESNVPROC)wglGetProcAddress("glDeleteFencesNV");
		glSetFenceNV = (PFNGLSETFENCENVPROC)wglGetProcAddress("glSetFenceNV");
		glTestFenceNV = (PFNGLTESTFENCENVPROC)wglGetProcAddress("glTestFenceNV");
		glFinishFenceNV = (PFNGLFINISHFENCENVPROC)wglGetProcAddress("glFinishFenceNV");
		glIsFenceNV = (PFNGLISFENCENVPROC)wglGetProcAddress("glIsFenceNV");
		glGetFenceivNV = (PFNGLGETFENCEIVNVPROC)wglGetProcAddress("glGetFenceivNV");
	}
	NV_evaluators = IsExtensionSupported("GL_NV_evaluators");
	if (NV_evaluators == TRUE)
	{
		glMapControlPointsNV = (PFNGLMAPCONTROLPOINTSNVPROC)wglGetProcAddress("glMapControlPointsNV");
		glMapParameterivNV = (PFNGLMAPPARAMETERIVNVPROC)wglGetProcAddress("glMapParameterivNV");
		glMapParameterfvNV = (PFNGLMAPPARAMETERFVNVPROC)wglGetProcAddress("glMapParameterfvNV");
		glGetMapControlPointsNV = (PFNGLGETMAPCONTROLPOINTSNVPROC)wglGetProcAddress("glGetMapControlPointsNV");
		glGetMapParameterivNV = (PFNGLGETMAPPARAMETERIVNVPROC)wglGetProcAddress("glGetMapParameterivNV");
		glGetMapParameterfvNV = (PFNGLGETMAPPARAMETERFVNVPROC)wglGetProcAddress("glGetMapParameterfvNV");
		glGetMapAttribParameterivNV = (PFNGLGETMAPATTRIBPARAMETERIVNVPROC)wglGetProcAddress("glGetMapAttribParameterivNV");
		glGetMapAttribParameterfvNV = (PFNGLGETMAPATTRIBPARAMETERFVNVPROC)wglGetProcAddress("glGetMapAttribParameterfvNV");
		glEvalMapsNV = (PFNGLEVALMAPSNVPROC)wglGetProcAddress("glEvalMapsNV");
	}

	NV_vertex_program = IsExtensionSupported("GL_NV_vertex_program");
	if (NV_vertex_program == TRUE)
	{
		glBindProgramNV = (PFNGLBINDPROGRAMNVPROC)wglGetProcAddress("glBindProgramNV");
		glDeleteProgramsNV = (PFNGLDELETEPROGRAMSNVPROC)wglGetProcAddress("glDeleteProgramsNV");
		glExecuteProgramNV = (PFNGLEXECUTEPROGRAMNVPROC)wglGetProcAddress("glExecuteProgramNV");
		glGenProgramsNV = (PFNGLGENPROGRAMSNVPROC)wglGetProcAddress("glGenProgramsNV");
		glAreProgramsResidentNV = (PFNGLAREPROGRAMSRESIDENTNVPROC)wglGetProcAddress("glAreProgramsResidentNV");
		glRequestResidentProgramsNV = (PFNGLREQUESTRESIDENTPROGRAMSNVPROC)wglGetProcAddress("glRequestResidentProgramsNV");
		glGetProgramParameterdvNV = (PFNGLGETPROGRAMPARAMETERDVNVPROC)wglGetProcAddress("glGetProgramParameterdvNV");
		glGetProgramParameterfvNV = (PFNGLGETPROGRAMPARAMETERFVNVPROC)wglGetProcAddress("glGetProgramParameterfvNV");
		glGetProgramivNV = (PFNGLGETPROGRAMIVNVPROC)wglGetProcAddress("glGetProgramivNV");
		glGetProgramStringNV = (PFNGLGETPROGRAMSTRINGNVPROC)wglGetProcAddress("glGetProgramStringNV");
		glGetVertexAttribdvNV = (PFNGLGETVERTEXATTRIBDVNVPROC)wglGetProcAddress("glGetVertexAttribdvNV");
		glGetVertexAttribfvNV = (PFNGLGETVERTEXATTRIBFVNVPROC)wglGetProcAddress("glGetVertexAttribfvNV");
		glGetVertexAttribivNV = (PFNGLGETVERTEXATTRIBIVNVPROC)wglGetProcAddress("glGetVertexAttribivNV");
		glGetVertexAttribPointervNV = (PFNGLGETVERTEXATTRIBPOINTERVNVPROC)wglGetProcAddress("glGetVertexAttribPointervNV");
		glIsProgramNV = (PFNGLISPROGRAMNVPROC)wglGetProcAddress("glIsProgramNV");
		glLoadProgramNV = (PFNGLLOADPROGRAMNVPROC)wglGetProcAddress("glLoadProgramNV");
		glProgramParameter4dNV = (PFNGLPROGRAMPARAMETER4DNVPROC)wglGetProcAddress("glProgramParameter4dNV");
		glProgramParameter4dvNV = (PFNGLPROGRAMPARAMETER4DVNVPROC)wglGetProcAddress("glProgramParameter4dvNV");
		glProgramParameter4fNV = (PFNGLPROGRAMPARAMETER4FNVPROC)wglGetProcAddress("glProgramParameter4fNV");
		glProgramParameter4fvNV = (PFNGLPROGRAMPARAMETER4FVNVPROC)wglGetProcAddress("glProgramParameter4fvNV");
		glProgramParameters4dvNV = (PFNGLPROGRAMPARAMETERS4DVNVPROC)wglGetProcAddress("glProgramParameters4dvNV");
		glProgramParameters4fvNV = (PFNGLPROGRAMPARAMETERS4FVNVPROC)wglGetProcAddress("glProgramParameters4fvNV");
		glTrackMatrixNV = (PFNGLTRACKMATRIXNVPROC)wglGetProcAddress("glTrackMatrixNV");
		glGetTrackMatrixNV = (PFNGLGETTRACKMATRIXIVNVPROC)wglGetProcAddress("glGetTrackMatrixNV");
		glVertexAttribPointerNV = (PFNGLVERTEXATTRIBPOINTERNVPROC)wglGetProcAddress("glVertexAttribPointerNV");
		glVertexAttrib1dNV = (PFNGLVERTEXATTRIB1DNVPROC)wglGetProcAddress("glVertexAttrib1dNV");
		glVertexAttrib1dvNV = (PFNGLVERTEXATTRIB1DVNVPROC)wglGetProcAddress("glVertexAttrib1dvNV");
		glVertexAttrib1fNV = (PFNGLVERTEXATTRIB1FNVPROC)wglGetProcAddress("glVertexAttrib1fNV");
		glVertexAttrib1fvNV = (PFNGLVERTEXATTRIB1FVNVPROC)wglGetProcAddress("glVertexAttrib1fvNV");
		glVertexAttrib1sNV = (PFNGLVERTEXATTRIB1SNVPROC)wglGetProcAddress("glVertexAttrib1sNV");
		glVertexAttrib1svNV = (PFNGLVERTEXATTRIB1SVNVPROC)wglGetProcAddress("glVertexAttrib1svNV");
		glVertexAttrib2dNV = (PFNGLVERTEXATTRIB2DNVPROC)wglGetProcAddress("glVertexAttrib2dNV");
		glVertexAttrib2dvNV = (PFNGLVERTEXATTRIB2DVNVPROC)wglGetProcAddress("glVertexAttrib2dvNV");
		glVertexAttrib2fNV = (PFNGLVERTEXATTRIB2FNVPROC)wglGetProcAddress("glVertexAttrib2fNV");
		glVertexAttrib2fvNV = (PFNGLVERTEXATTRIB2FVNVPROC)wglGetProcAddress("glVertexAttrib2fvNV");
		glVertexAttrib2sNV = (PFNGLVERTEXATTRIB2SNVPROC)wglGetProcAddress("glVertexAttrib2sNV");
		glVertexAttrib2svNV = (PFNGLVERTEXATTRIB2SVNVPROC)wglGetProcAddress("glVertexAttrib2svNV");
		glVertexAttrib3dNV = (PFNGLVERTEXATTRIB3DNVPROC)wglGetProcAddress("glVertexAttrib3dNV");
		glVertexAttrib3dvNV = (PFNGLVERTEXATTRIB3DVNVPROC)wglGetProcAddress("glVertexAttrib3dvNV");
		glVertexAttrib3fNV = (PFNGLVERTEXATTRIB3FNVPROC)wglGetProcAddress("glVertexAttrib3fNV");
		glVertexAttrib3fvNV = (PFNGLVERTEXATTRIB3FVNVPROC)wglGetProcAddress("glVertexAttrib3fvNV");
		glVertexAttrib3sNV = (PFNGLVERTEXATTRIB3SNVPROC)wglGetProcAddress("glVertexAttrib3sNV");
		glVertexAttrib3svNV = (PFNGLVERTEXATTRIB3SVNVPROC)wglGetProcAddress("glVertexAttrib3svNV");
		glVertexAttrib4dNV = (PFNGLVERTEXATTRIB4DNVPROC)wglGetProcAddress("glVertexAttrib4dNV");
		glVertexAttrib4fNV = (PFNGLVERTEXATTRIB4DVNVPROC)wglGetProcAddress("glVertexAttrib4fNV");
		glVertexAttrib4sNV = (PFNGLVERTEXATTRIB4FNVPROC)wglGetProcAddress("glVertexAttrib4sNV");
		glVertexAttrib4dvNV = (PFNGLVERTEXATTRIB4FVNVPROC)wglGetProcAddress("glVertexAttrib4dvNV");
		glVertexAttrib4fvNV = (PFNGLVERTEXATTRIB4SNVPROC)wglGetProcAddress("glVertexAttrib4fvNV");
		glVertexAttrib4svNV = (PFNGLVERTEXATTRIB4SVNVPROC)wglGetProcAddress("glVertexAttrib4svNV");
		glVertexAttrib4ubNV = (PFNGLVERTEXATTRIB4UBNVPROC)wglGetProcAddress("glVertexAttrib4ubNV");
		glVertexAttrib4ubvNV = (PFNGLVERTEXATTRIB4UBVNVPROC)wglGetProcAddress("glVertexAttrib4ubvNV");
		glVertexAttribs1dvNV = (PFNGLVERTEXATTRIBS1DVNVPROC)wglGetProcAddress("glVertexAttribs1dvNV");
		glVertexAttribs1fvNV = (PFNGLVERTEXATTRIBS1FVNVPROC)wglGetProcAddress("glVertexAttribs1fvNV");
		glVertexAttribs1svNV = (PFNGLVERTEXATTRIBS1SVNVPROC)wglGetProcAddress("glVertexAttribs1svNV");
		glVertexAttribs2dvNV = (PFNGLVERTEXATTRIBS2DVNVPROC)wglGetProcAddress("glVertexAttribs2dvNV");
		glVertexAttribs2fvNV = (PFNGLVERTEXATTRIBS2FVNVPROC)wglGetProcAddress("glVertexAttribs2fvNV");
		glVertexAttribs2svNV = (PFNGLVERTEXATTRIBS2SVNVPROC)wglGetProcAddress("glVertexAttribs2svNV");
		glVertexAttribs3dvNV = (PFNGLVERTEXATTRIBS3DVNVPROC)wglGetProcAddress("glVertexAttribs3dvNV"); 
		glVertexAttribs3fvNV = (PFNGLVERTEXATTRIBS3FVNVPROC)wglGetProcAddress("glVertexAttribs3fvNV");
		glVertexAttribs3svNV = (PFNGLVERTEXATTRIBS3SVNVPROC)wglGetProcAddress("glVertexAttribs3svNV");
		glVertexAttribs4dvNV = (PFNGLVERTEXATTRIBS4DVNVPROC)wglGetProcAddress("glVertexAttribs4dvNV");
		glVertexAttribs4fvNV = (PFNGLVERTEXATTRIBS4FVNVPROC)wglGetProcAddress("glVertexAttribs4fvNV");
		glVertexAttribs4svNV = (PFNGLVERTEXATTRIBS4SVNVPROC)wglGetProcAddress("glVertexAttribs4svNV");
		glVertexAttribs4ubvNV = (PFNGLVERTEXATTRIBS4UBVNVPROC)wglGetProcAddress("glVertexAttribs4ubvNV");
	}

	NV_register_combiners = IsExtensionSupported("GL_NV_register_combiners");
	if (NV_register_combiners == TRUE)
	{
		glCombinerParameterfvNV = (PFNGLCOMBINERPARAMETERFVNVPROC)wglGetProcAddress("glCombinerParameterfvNV");
		glCombinerParameterivNV = (PFNGLCOMBINERPARAMETERIVNVPROC)wglGetProcAddress("glCombinerParameterivNV");
		glCombinerParameterfNV = (PFNGLCOMBINERPARAMETERFNVPROC)wglGetProcAddress("glCombinerParameterfNV");
		glCombinerParameteriNV = (PFNGLCOMBINERPARAMETERINVPROC)wglGetProcAddress("glCombinerParameteriNV");
		glCombinerInputNV = (PFNGLCOMBINERINPUTNVPROC)wglGetProcAddress("glCombinerInputNV");
		glCombinerOutputNV = (PFNGLCOMBINEROUTPUTNVPROC)wglGetProcAddress("glCombinerOutputNV");
		glFinalCombinerInputNV = (PFNGLFINALCOMBINERINPUTNVPROC)wglGetProcAddress("glFinalCombinerInputNV");
		glGetCombinerInputParameterfvNV = (PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC)wglGetProcAddress("glGetCombinerInputParameterfvNV");
		glGetCombinerInputParameterivNV = (PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC)wglGetProcAddress("glGetCombinerInputParameterivNV");
		glGetCombinerOutputParameterfvNV = (PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC)wglGetProcAddress("glGetCombinerOutputParameterfvNV");
		glGetCombinerOutputParameterivNV = (PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC)wglGetProcAddress("glGetCombinerOutputParameterivNV");
		glGetFinalCombinerInputParameterfvNV = (PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC)wglGetProcAddress("glGetFinalCombinerInputParameterfvNV");
		glGetFinalCombinerInputParameterivNV = (PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC)wglGetProcAddress("glGetFinalCombinerInputParameterivNV");
	}

	NV_register_combiners2 = IsExtensionSupported("GL_NV_register_combiners2");
	if (NV_register_combiners2 == TRUE)
	{
		glCombinerStageParameterfvNV = (PFNGLCOMBINERSTAGEPARAMETERFVNVPROC)wglGetProcAddress("glCombinerStageParameterfvNV");
		glGetCombinerStageParameterfvNV = (PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC)wglGetProcAddress("glGetCombinerStageParameterfvNV");;
	}

	NV_vertex_array_range = IsExtensionSupported("GL_NV_vertex_array_range");
	if (NV_vertex_array_range == TRUE)
	{
		glFlushVertexArrayRangeNV = (PFNGLFLUSHVERTEXARRAYRANGENVPROC)wglGetProcAddress("glFlushVertexArrayRangeNV");
		glVertexArrayRangeNV = (PFNGLVERTEXARRAYRANGENVPROC)wglGetProcAddress("glVertexArrayRangeNV");
	}

	EXT_stencil_two_side = IsExtensionSupported("GL_EXT_stencil_two_side");
	if (EXT_stencil_two_side == TRUE)
	{
		glActiveStencilFaceEXT = (PFNGLACTIVESTENCILFACEXT)wglGetProcAddress("glActiveStencilFaceEXT");
	}
}

#endif

#ifdef GLEXT_PROTOTYPE
#undef GLEXT_PROTOTYPE
#endif
