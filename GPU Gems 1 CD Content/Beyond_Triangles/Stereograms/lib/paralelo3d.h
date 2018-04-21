#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>

#include <gl\gl.h>
#include <gl\glu.h>

#include "wglATI.h"
#include "glATI.h"
#include "glNVidia.h"
#include "glExtension.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#define P_SUPPORT_3DS

int CgCheckError(void);
char *CgLoadProgramText(const char *file);

#define P3D_FILEID				0x9171
#define P3D_FILEVER				0x0002

#define P_MAX_THREADS			4

#define P_FACEFLAGS_EDGE_A		(1<<0)
#define P_FACEFLAGS_EDGE_B		(1<<1)
#define P_FACEFLAGS_EDGE_C		(1<<2)
#define P_FACEFLAGS_EDGE_ALL	(P_FACEFLAGS_EDGE_A|P_FACEFLAGS_EDGE_B|P_FACEFLAGS_EDGE_C)
#define P_FACEFLAGS_TWOSIDE		(1<<3)
#define P_FACEFLAGS_NOSHADOW	(1<<4)

#define P_MATFLAGS_TWOSIDE		(1<<3)
#define P_MATFLAGS_NOSHADOW		(1<<4)
#define P_MATFLAGS_ADDBLEND		(1<<5)
#define P_MATFLAGS_ENVMAP		(1<<6)

#define P_LIGHTFLAGS_ENABLED	(1<<0)
#define P_LIGHTFLAGS_NOSHADOW	(1<<1)
#define P_LIGHTFLAGS_SELECTED	(1<<2)

#define P_COMPUTE_FACENORM	(1<<0)
#define P_COMPUTE_VERTNORM	(1<<1)
#define P_COMPUTE_TANGENTS	(1<<2)
#define P_COMPUTE_BBOX		(1<<3)

#define P_FONTS_NUM			16
#define P_FONTS_FACTOR		0.0625f

#define P_PACK_FLOAT_TO_BYTE(in) ((unsigned char)(((in)+1.0f)*127.5f));

enum 
{ 
	P_ANIMTYPE_VALUE_LINEAR=0,
	P_ANIMTYPE_VALUE_SMOOTH,
	P_ANIMTYPE_POINT_LINEAR,
	P_ANIMTYPE_POINT_SMOOTH,
	P_ANIMTYPE_QUAT_LINEAR,
	P_ANIMTYPE_QUAT_SMOOTH
};

class pVector;
class pPlane;
class pVertex;
class pQuaternion;
class pBoundBox;
class pMatrix;
class pString;
class pMaterial;
class pCamera;
class pLight;
class pOcTreeNode;
class pOcTree;
class pMaterial;
class pFace;
class pFaceLight;
class pMesh;
class pPicture;
class pRender;

#define D3D_VERTEX_DATA (D3DFVF_XYZ|D3DFVF_NORMAL|D3DFVF_TEX1)

#include "pMath.h"
#include "pAnimation.h"
#include "pArray.h"
#include "pString.h"
#include "pVertex.h"
#include "pBoundBox.h"
#include "pFrustum.h"
#include "pCamera.h"
#include "pLight.h"
#include "pOcTreeNode.h"
#include "pOcTree.h"
#include "pMaterial.h"
#include "pFace.h"
#include "pFaceLight.h"
#include "pMesh.h"
#include "pPicture.h"
#include "pRenderProfile.h"
#include "pRender.h"
