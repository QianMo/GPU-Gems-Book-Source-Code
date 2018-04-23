// this class comes from the NVidia SDK

#ifndef _MOVIEMAKER_H
#define _MOVIEMAKER_H

//  ===============================================
//  MovieMaker class definition.
//  ===============================================

#include <windows.h>
//#include <windef.h>
#include <vfw.h>

//#include <vfw.h>
//#include <windows.h>


#define TEXT_HEIGHT	20
#define AVIIF_KEYFRAME	0x00000010L // this frame is a key frame.
#define BUFSIZE 260

class MovieMaker {
private:
  //CString FName;
  char fname[64];
  int width;
  int height;

  HWND m_hWnd;
  AVISTREAMINFO strhdr;
  PAVIFILE pfile;
  PAVISTREAM ps;
  PAVISTREAM psCompressed;
  PAVISTREAM psText;
  AVICOMPRESSOPTIONS opts;
  AVICOMPRESSOPTIONS FAR * aopts[1];
  DWORD dwTextFormat;
  char szText[BUFSIZE];
  int nFrames;
  bool bOK;


public:
  MovieMaker();
  ~MovieMaker();

  inline bool IsOK() const { return bOK; };
  void StartCapture(HWND,int,int,const char *name );
  void EndCapture();
  bool Snap();
};

#endif
