#pragma once

// Draws other HUD stuff
//
void RenderHUD(void)
{
  // Build stats string
  //
  char strText[1024];
  PrintStats(strText);
  if(strText == 0) return;

  D3D10_VIEWPORT Viewport;
  unsigned int iNumVP = 1;
  GetApp()->GetDevice()->RSGetViewports(&iNumVP, &Viewport);
  RECT destRect;
  SetRect(&destRect, 10, 10, Viewport.Width - 10, 400);

  // draw stats
  g_pFont->DrawTextA(NULL, strText, -1, &destRect, DT_NOCLIP, 0xFFFFFFFF);
}