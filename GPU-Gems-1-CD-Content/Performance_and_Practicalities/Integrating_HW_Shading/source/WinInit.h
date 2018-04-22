// from C4Dfx by Jörn Loviscach, www.l7h.cn
// functions to build a dummy window and to initialize OpenGL extensions
// and to call an external editor software

#if !defined(WIN_INIT_H)
#define WIN_INIT_H

namespace WinInit
{
	void StartWin(void); // never call twice
	bool IsOpenGLOK(void);
	bool HasMultisampleFilterHintNV(void);

	void StartEditor(const char* file, int line);
}

#endif
