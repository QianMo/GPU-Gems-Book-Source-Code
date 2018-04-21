// from C4Dfx by Jörn Loviscach, www.l7h.cn
// declaration of a function to retrieve the instance number from Windows

#if !defined(WIN_HACK_H)
#define WIN_HACK_H

#include "windows.h"

// Edit c4d_pmain.cpp to make this work!
// add as line 33: static HINSTANCE g_hinstDLL;
// add as line 37: g_hinstDLL = (HINSTANCE)hModule;
// comment line 50: // static HINSTANCE g_hinstDLL;
// add the following function at the end:
// HINSTANCE GetInstance(void)
// {
//     return g_hinstDLL;
// }

extern HINSTANCE GetInstance(void);

#endif