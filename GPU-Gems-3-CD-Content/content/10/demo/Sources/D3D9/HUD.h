#pragma once

#include <TCHAR.h>
//
// Drawing text and 2D stuff on screen is done here
//
//

// Draws a shadowmap to the screen
//
void RenderSplitOnHUD(int iSplit)
{
  if(!g_bHUDTextures) return;
  GetApp()->GetDevice()->SetPixelShader(NULL);
  GetApp()->GetDevice()->SetVertexShader(NULL);
  GetApp()->GetDevice()->SetFVF(D3DFVF_XYZ|D3DFVF_TEX1|D3DFVF_DIFFUSE);

  Matrix mIdentity;
  mIdentity.SetIdentity();

  GetApp()->GetDevice()->SetTransform(D3DTS_PROJECTION, (D3DXMATRIX*)&mIdentity);
  GetApp()->GetDevice()->SetTransform(D3DTS_WORLD, (D3DXMATRIX*)&mIdentity);
  GetApp()->GetDevice()->SetTransform(D3DTS_VIEW, (D3DXMATRIX*)&mIdentity);
  GetApp()->GetDevice()->SetRenderState(D3DRS_ZENABLE,TRUE);
  GetApp()->GetDevice()->SetRenderState(D3DRS_ZFUNC, D3DCMP_ALWAYS);
  GetApp()->GetDevice()->SetRenderState(D3DRS_ZWRITEENABLE,TRUE);
  GetApp()->GetDevice()->SetRenderState(D3DRS_CULLMODE,D3DCULL_NONE);
  GetApp()->GetDevice()->SetRenderState(D3DRS_LIGHTING,FALSE);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);

  // make sure its drawn on front of everything
  //
  D3DVIEWPORT9 Viewport;
  GetApp()->GetDevice()->GetViewport(&Viewport);
  Viewport.MinZ=0;
  Viewport.MaxZ=0;
  GetApp()->GetDevice()->SetViewport(&Viewport);

  float fAspect=Viewport.Width/(float)Viewport.Height;

  float fSizeX=(1.8f-0.1f*g_iNumSplits)/g_iNumSplits;
  if(fSizeX>0.25f) fSizeX=0.25f;
  float fSizeY=fSizeX;

  fSizeX/=fAspect;

  float fOffset=fSizeX*iSplit+iSplit*0.1f;

  float fStartX=-0.9f+fOffset;
  float fStartY=-0.9f+fSizeY;
  float fEndX=-0.9f+fOffset+fSizeX;
  float fEndY=-0.9f;
  DWORD iColor=0xFFFFFFFF;
  float pVertices[4][6]=
  {
    { fStartX , fStartY, 0, *(float*)&iColor, 0,0},
    { fEndX   , fStartY, 0, *(float*)&iColor, 1,0},
    { fStartX , fEndY  , 0, *(float*)&iColor, 0,1},
    { fEndX   , fEndY  , 0, *(float*)&iColor, 1,1}
  };

  GetApp()->GetDevice()->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, pVertices, sizeof(float)*6);

  // reset states
  GetApp()->GetDevice()->SetRenderState(D3DRS_ZENABLE,TRUE);
  GetApp()->GetDevice()->SetRenderState(D3DRS_ZWRITEENABLE,TRUE);
  GetApp()->GetDevice()->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESS);
  GetApp()->GetDevice()->SetTexture(0,NULL);
}


// Draws other HUD stuff
//
void RenderHUD(void)
{
  // Build stats string
  //
  char strText[1024];
  PrintStats(strText);
  if(strText == 0) return;

  D3DVIEWPORT9 Viewport;
  GetApp()->GetDevice()->GetViewport(&Viewport);
  RECT destRect;
  SetRect(&destRect, 10, 10, Viewport.Width - 10, 400);

  // draw stats
  g_pFont->DrawTextA(NULL, strText, -1, &destRect, DT_NOCLIP, 0xFFFFFFFF);
}
