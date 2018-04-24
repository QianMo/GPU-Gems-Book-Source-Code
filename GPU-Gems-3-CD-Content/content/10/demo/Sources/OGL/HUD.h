#pragma once

//
// Drawing text and 2D stuff on screen is done here
//
//

// Draws other HUD stuff
//
void RenderHUD(void)
{
  // Build stats string
  //
  char strText[1024];
  PrintStats(strText);
  if(strText == 0) return;

  // disable shaders
  if(g_pActiveShader != NULL) g_pActiveShader->Deactivate();

  // get view port
  int pViewPort[4] = {0, 0, 800, 600};
  glGetIntegerv(GL_VIEWPORT, pViewPort);

  // reset matrices
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // draw stats line by line
  glListBase (1000);
  glLoadIdentity();
  glTranslatef(0.0f,0.0f,-1.0f);
  unsigned int iOffset = 0;
  unsigned int iStart = 0;
  unsigned int iLine = 0;
  while(strText[iOffset] != 0)
  {
    if(strText[iOffset] == '\n')
    {
      glRasterPos2f(-1.0f + 10.0f*2.0f/(float)pViewPort[2], 1.0f - (iLine+1)*20.0f*2.0f/(float)pViewPort[3]);
      glCallLists(iOffset - iStart, GL_UNSIGNED_BYTE, strText + iStart);
      iStart = iOffset + 1;
      iLine++;
    }
    iOffset++;
  }
}
