#ifndef __BUILDTEXTURES_H__included_
#define __BUILDTEXTURES_H__included_

//  BuildLambertIrradianceTextures computes Al * Ylm(theta,phi) for dwSize*dwSize*6 points on the
//  sphere (cubemap), with l=0..dwOrder-1, and a lambert diffuse function
HRESULT BuildLambertIrradianceTextures(LPDIRECT3DTEXTURE9 *weightTextures, 
                                       LPDIRECT3DDEVICE9 lpDevice, 
                                       DWORD dwOrder, 
                                       DWORD dwSize);

//  BuildPhongIrradianceTextures computes Al * Ylm(theta,phi) for dwSize*dwSize*6 points on the
//  sphere (cubemap), with l=0..dwOrder-1, and a phong specular function
HRESULT BuildPhongIrradianceTextures(LPDIRECT3DTEXTURE9 *weightTextures, 
                                     LPDIRECT3DDEVICE9 lpDevice, 
                                     DWORD dwOrder, 
                                     DWORD dwSize,
                                     FLOAT specular);


//  BuildDualParaboloidWeightTextures computes the SH basis functions*solid angle for each texel
//  in a dwSize*dwSize dual-paraboloid map.
HRESULT BuildDualParaboloidWeightTextures(LPDIRECT3DTEXTURE9 *weightTextures, 
                                          LPDIRECT3DDEVICE9 lpDevice, 
                                          DWORD dwOrder, 
                                          DWORD dwSize);

//  BuildCubemapWeightTextures works exactly like BuildDualParaboloidWeightTextures, but for
//  cubemaps rather than dual-paraboloid maps.
HRESULT BuildCubemapWeightTextures(LPDIRECT3DTEXTURE9 *weightTextures, 
                                   LPDIRECT3DDEVICE9 lpDevice, 
                                   DWORD dwOrder, 
                                   DWORD dwSize);

#endif