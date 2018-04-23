/*
s_p_oneil@hotmail.com
Copyright (c) 2000, Sean O'Neil
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of this project nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __GameApp_h__
#define __GameApp_h__

class CGameEngine;


/*******************************************************************************
* Class: CGameApp
********************************************************************************
* This class is similar to MFC's CWinApp, providing a lot of the same functions
* without the extra baggage of MFC.
*******************************************************************************/
class CGameApp : public CWinApp
{
// Attributes
protected:
	HDC m_hDC;
	HGLRC m_hGLRC;
	bool m_bActive;
	int m_nTimer;
	int m_nWidth, m_nHeight;

	CGameEngine *m_pGameEngine;

// Operations
protected:
	static LRESULT CALLBACK WindowProc(HWND hWnd, UINT nMsg, WPARAM wParam, LPARAM lParam);
	int OnCreate(HWND hWnd);
	void OnDestroy();
	void OnSize(int nType, int nWidth, int nHeight);

public:
	CGameApp(HINSTANCE hInstance, HINSTANCE hPrevInstance=NULL, char *pszCmdLine="", int nShowCmd=SW_SHOWNORMAL)
		: CWinApp(hInstance, hPrevInstance, pszCmdLine, nShowCmd)
	{
		m_hDC = NULL;
		m_hGLRC = NULL;
		m_bActive = false;
		m_pGameEngine = NULL;
	}

	virtual bool InitInstance();
	virtual int ExitInstance();
	virtual bool OnIdle();
	virtual void Pause();
	virtual void Restore();
	virtual bool InitMode(bool bFullScreen, int nWidth, int nHeight);
	bool IsActive()								{ return m_bActive; }
	void MakeCurrent()							{ wglMakeCurrent(m_hDC, m_hGLRC); }
	HGLRC GetHGLRC()							{ return m_hGLRC; }
	HDC GetHDC()								{ return m_hDC; }
	CGameEngine *GetGameEngine()				{ return m_pGameEngine; }

	int GetWidth()	{ return m_nWidth; }
	int GetHeight()	{ return m_nHeight; }
};

inline CGameApp *GetGameApp()		{ return (CGameApp *)GetApp(); }
inline CGameEngine *GetGameEngine()	{ return GetGameApp()->GetGameEngine(); }

#endif // __GameApp_h__
