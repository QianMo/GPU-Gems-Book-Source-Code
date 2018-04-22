////////////////////////////////////////////////////////////
//
// NVEBGlutAPI.c
//
// This file defines the entry points needed to initialize
// and deinitialize GLUT programs that are to be loaded as
// DLLs.
//
////////////////////////////////////////////////////////////


#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <string>


////////////////////////////////////////////////////////////
// Internal types and namespaces

using namespace std;

typedef void (*LPATEXITNOTIFIER)(void*);
typedef void (*LPNVPARSENOTIFIER)(const char*, void*);
typedef void (*LPCGADDPROGRAMNOTIFIER)(const char*, int, void*);
typedef void (*LPCGADDPROGRAMFROMFILENOTIFIER)(const char*, int, void*);

////////////////////////////////////////////////////////////
// Internal private helper functions

static bool nvebGlutInit();
static bool nvebGlutExit();
static void __cdecl nvebAtExit();


////////////////////////////////////////////////////////////
// Internal private data

static bool                     nvebInstrumented                = false;
static LPATEXITNOTIFIER         nvebAtExitNotifierFunc          = NULL;
static void*                    nvebAtExitNotifierCbData        = NULL;
static LPNVPARSENOTIFIER        nvebNvParseNotifierFunc         = NULL;
static void*                    nvebNvParseNotifierCbData       = NULL;
static LPCGADDPROGRAMNOTIFIER   nvebCgAddProgramNotifierFunc    = NULL;
static void*                    nvebCgAddProgramNotifierCbData  = NULL;
static int                      nvebCgAddProgramNotifierProfile = 0;

static LPCGADDPROGRAMFROMFILENOTIFIER   nvebCgAddProgramFromFileNotifierFunc    = NULL;
static void*                            nvebCgAddProgramFromFileNotifierCbData  = NULL;
static int                              nvebCgAddProgramFromFileNotifierProfile = 0;


////////////////////////////////////////////////////////////
// Effect entry point (to initialize)

extern "C"
__declspec(dllexport) bool __nvebEffectEntryInit(HINSTANCE hInst)
{
    // Do the C/C++ RT initializations
	if (!nvebGlutInit())
		return false;

    // Record that the program is running in instrumented mode
	nvebInstrumented = true;

    // Setup an atexit() handler to catch accidental calls to exit()
    atexit(&nvebAtExit);

    // Call main with our made up parameters...
	extern int main(int, char**);

	int argc = 1;
	char *argv[2];
	char *argv0 = "NVEffectsBrowser";
	char *argv1 = NULL;

	argv[0] = argv0;
	argv[1] = argv1;

	main(argc, argv);

	return true;
}


////////////////////////////////////////////////////////////
// Effect entry point (at exit)

extern "C"
__declspec(dllexport) bool __nvebEffectEntryExit(HINSTANCE hInst)
{
	if (!nvebGlutExit())
		return false;

	return true;
}


////////////////////////////////////////////////////////////
// Effect entry point (register effect notifier callbacks)

extern "C"
__declspec(dllexport) bool __nvebEffectRegisterAtExitNotifier(void *func, void *cbdata)
{
    nvebAtExitNotifierFunc   = (LPATEXITNOTIFIER) func;
    nvebAtExitNotifierCbData = cbdata;
    return true;
}

extern "C"
__declspec(dllexport) bool __nvebEffectRegisterNvParseNotifier(void *func, void *cbdata)
{
    nvebNvParseNotifierFunc   = (LPNVPARSENOTIFIER) func;
    nvebNvParseNotifierCbData = cbdata;
    return true;
}

#ifdef NVEB_USING_CGGL

extern "C"
__declspec(dllexport) bool __nvebEffectRegisterCgAddProgramNotifier(void *func, void *cbdata, int profile)
{
    nvebCgAddProgramNotifierFunc    = (LPCGADDPROGRAMNOTIFIER) func;
    nvebCgAddProgramNotifierCbData  = cbdata;
    nvebCgAddProgramNotifierProfile = profile;
    
    return true;
}

extern "C"
__declspec(dllexport) bool __nvebEffectRegisterCgAddProgramFromFileNotifier(void *func, void *cbdata, int profile)
{
    nvebCgAddProgramFromFileNotifierFunc    = (LPCGADDPROGRAMFROMFILENOTIFIER) func;
    nvebCgAddProgramFromFileNotifierCbData  = cbdata;
    nvebCgAddProgramFromFileNotifierProfile = profile;
    
    return true;
}

#endif

////////////////////////////////////////////////////////////
// Effect custom exit() routine

extern "C"
void __cdecl __nvebExit(int res)
{
    // I've changed the behavior here.
    //
    // Just ignoring calls to exit() certainly allows the app
    // to keep running, but that isn't necessarily a good thing.
    // Many apps are coded to reasaonably expect exit() to never
    // return.  When it does, lots of things may break.
    //
    // So instead of ignoring exit() calls, we catch them and
    // the Glut filter tries to do something reasonable (either
    // reset the DLL (if the exit() happens while running) or 
    // tell the browser that we can't run at all (if the exit()
    // occurs during initialization.))
    //
    // So we always, actually call exit().
    //
    exit(res);
}


////////////////////////////////////////////////////////////
// Effect custom nvparse() routine

#ifdef NVEB_USING_NVPARSE

extern "C"
void __cdecl __nvebNvParse(const char *lpProg, int argc, ...)
{
    extern void nvparse(const char*, int, ...);

    if (nvebInstrumented)
        if (nvebNvParseNotifierFunc)
            nvebNvParseNotifierFunc(lpProg, nvebNvParseNotifierCbData);

	if(!strncmp(lpProg, "!!VSP1.0", 8))	{

        va_list ap;
		va_start(ap, lpProg);
		int vpsid = va_arg(ap,int);
		va_end(ap);
		nvparse(lpProg,vpsid);

    } else {

        nvparse(lpProg,argc);
    }
}

#endif

////////////////////////////////////////////////////////////
// Effect custom cgAddText() routine

#ifdef NVEB_USING_CGGL

extern "C"
cgError __cdecl __nvebCgAddProgram(void *context, const char *lpProg, int profile, char *entry)
{
    extern cgError cgAddProgram(void *, const char *, int, char *);

    if (nvebInstrumented)
        if (nvebCgAddProgramNotifierFunc)
            nvebCgAddProgramNotifierFunc(lpProg, profile, nvebCgAddProgramNotifierCbData);

    return cgAddProgram(context, lpProg, profile, entry);
}

extern "C"
cgError __cdecl __nvebCgAddProgramFromFile(void *context, const char *lpFilename, int profile, char *entry)
{
    extern cgError cgAddProgramFromFile(void *, const char *, int, char *);

    if (nvebInstrumented)
        if (nvebCgAddProgramFromFileNotifierFunc)
            nvebCgAddProgramFromFileNotifierFunc(lpFilename, profile, nvebCgAddProgramFromFileNotifierCbData);

    return cgAddProgramFromFile(context, lpFilename, profile, entry);
}

#endif


////////////////////////////////////////////////////////////
// Effect internal routines

static bool nvebGlutInit()
{
	return true;
}

static bool nvebGlutExit()
{
	return true;
}

static void __cdecl nvebAtExit()
{
    if (nvebAtExitNotifierFunc)
        nvebAtExitNotifierFunc(nvebAtExitNotifierCbData);
}