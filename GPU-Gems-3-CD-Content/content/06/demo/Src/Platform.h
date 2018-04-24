#ifndef __PLATFORM_H
#define __PLATFORM_H

#include "Cfg.h"

#if defined(PLATFORM_D3D9)
#	include <d3d9.h>
#	include <d3dx9.h>
	typedef ID3DXMesh	IMesh;
	typedef IDirect3DDevice9 IDirect3DDevice;
#else
#	include <d3d10.h>
#	include <d3dx10.h>
#
	typedef ID3D10Device IDirect3DDevice;
	typedef ID3DX10Mesh IMesh;
#endif

#endif
