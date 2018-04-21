////////////////////////////////////////////////////////////
//
// NVEBGlutAPI.h
//
// This file defines the entry points needed to initialize
// and deinitialize GLUT programs that are to be loaded as
// DLLs.
//
////////////////////////////////////////////////////////////


#ifndef __NVEB_GLUT_API__
#define __NVEB_GLUT_API__


////////////////////////////////////////////////////////////
// Macros to simplify the declarations of the public macros
// below
//

#ifdef __cplusplus
	// Make sure the functions are C-typed...
#define NVEB_EFFECT_PROLOG  extern "C" {
#define NVEB_EFFECT_EPILOG  }
#else
	// Already in C, do nothing...
#define NVEB_EFFECT_PROLOG  
#define NVEB_EFFECT_EPILOG  
#endif

    // All configuration functions should have this signature.
    // The integer iConfig is the configuration to set up for.
    // This function should return true if it is successful.
    // If it returns false, then the effect will unload and
    // not run.  All configuration functions should be declared
    // with C scoping.  That is, from inside C++, be sure to
    // place 'extern "C"' before your function definition.
    //
typedef bool (*NVEBEffectConfigFunc)(unsigned int iConfig);

	// Declare a function to return a string (const char*)
#define NVEB_EFFECT_STRING_FUNC(Func,Value)					    	\
NVEB_EFFECT_PROLOG											    	\
__declspec(dllexport) const char* __nvebEffectGet##Func()	    	\
	{ return Value; }										    	\
NVEB_EFFECT_EPILOG

	// Declare a function to return a number (UINT)
#define NVEB_EFFECT_UINT_FUNC(Func,Value)				    		\
NVEB_EFFECT_PROLOG										    		\
__declspec(dllexport) UINT __nvebEffectGet##Func()		    		\
	{ return Value; }									    		\
NVEB_EFFECT_EPILOG

	// Declare a function to return a config function (NVEBEffectConfigFunc)
#define NVEB_EFFECT_PFN_FUNC(Func,Value)				    		\
NVEB_EFFECT_PROLOG										    		\
__declspec(dllexport) NVEBEffectConfigFunc __nvebEffectGet##Func()	\
	{ extern bool Value(unsigned int); return &Value; }	   		\
NVEB_EFFECT_EPILOG


////////////////////////////////////////////////////////////
// Define the public effects browser macros
//

    // Use this macro to declare the number of configurations
    // that your Glut application supports.  If you don't declare
    // this macro, then one configuration will be assumed.
    //
#define NVEB_EFFECT_CONFIGURATIONS(nNum)                            \
    NVEB_EFFECT_UINT_FUNC(Configs,nNum)

    // The prefunc and postfunc functions will be called before
    // and after main() respectively, if they are not NULL.
    //
#define NVEB_EFFECT_CONFIG_PREFUNC(pfnPreFunc)                      \
    NVEB_EFFECT_PFN_FUNC(ConfigPreFunc,pfnPreFunc)
#define NVEB_EFFECT_CONFIG_POSTFUNC(pfnPostFunc)                    \
    NVEB_EFFECT_PFN_FUNC(ConfigPostFunc,pfnPostFunc)

	// Declare the name of this configuration of this effect
    //
#define NVEB_EFFECT_NAME(strName)						    		\
	NVEB_EFFECT_STRING_FUNC(Name,strName)
#define NVEB_EFFECT_NAME_CONFIG(nNum,strName)						\
	NVEB_EFFECT_STRING_FUNC(Name##nNum,strName)

	// Declare the version of this configuration of this effect
	//
#define NVEB_EFFECT_VERSION(strVersion)						    	\
	NVEB_EFFECT_STRING_FUNC(Version,strVersion)
#define NVEB_EFFECT_VERSION_CONFIG(nNum,strVersion)			    	\
	NVEB_EFFECT_STRING_FUNC(Version##nNum,strVersion)

	// Declare the location of this configuration of this effect
	//
#define NVEB_EFFECT_LOCATION(strLocation)				    		\
	NVEB_EFFECT_STRING_FUNC(Location,strLocation)
#define NVEB_EFFECT_LOCATION_CONFIG(nNum,strLocation)	    		\
	NVEB_EFFECT_STRING_FUNC(Location##nNum,strLocation)

	// Declare the icon of this configuration of this effect
	//
#define NVEB_EFFECT_ICON(nIconID)							    	\
	NVEB_EFFECT_UINT_FUNC(IconID,nIconID)
#define NVEB_EFFECT_ICON_CONFIG(nNum,nIconID)					    \
	NVEB_EFFECT_UINT_FUNC(IconID##nNum,nIconID)

	// Declare the strings defining the nLine'th line of this
    // configuration of this effect
    //
#define NVEB_EFFECT_ABOUT(nLine,strLabel,strText,strURL)	    	\
	NVEB_EFFECT_STRING_FUNC(AboutLabel##nLine, strLabel)	    	\
	NVEB_EFFECT_STRING_FUNC(AboutText##nLine, strText)		    	\
	NVEB_EFFECT_STRING_FUNC(AboutURL##nLine, strURL)
#define NVEB_EFFECT_ABOUT_CONFIG(nNum,nLine,strLabel,strText,strURL)\
	NVEB_EFFECT_STRING_FUNC(AboutLabel##nLine##nNum,strLabel)	   	\
	NVEB_EFFECT_STRING_FUNC(AboutText##nLine##nNum,strText)		   	\
	NVEB_EFFECT_STRING_FUNC(AboutURL##nLine##nNum,strURL)


////////////////////////////////////////////////////////////
// Override the standard exit() function to use our own...
//
#define exit  __nvebExit

NVEB_EFFECT_PROLOG
	extern void __cdecl __nvebExit(int);
NVEB_EFFECT_EPILOG


////////////////////////////////////////////////////////////
// Override the standard nvparse() function to use our own...
//
#ifdef NVEB_USING_NVPARSE

#define nvparse  __nvebNvParse

NVEB_EFFECT_PROLOG
	extern void __nvebNvParse(const char*,...);
NVEB_EFFECT_EPILOG

#endif


#endif
