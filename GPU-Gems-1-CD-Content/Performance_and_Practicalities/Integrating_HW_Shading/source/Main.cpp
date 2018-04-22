// from C4Dfx by Jörn Loviscach, www.l7h.cn
// the functions called by Cinema 4D to start and end the plug-in

#include "c4d.h"
#include "WinInit.h"

Bool RegisterFXCommand(void);
Bool RegisterFXMaterial(void);
Bool RegisterPrefs(void);
Bool RegisterLightTag(void);
void FreeFXCommand(void);
void FreePrefs(void);

Bool PluginStart(void)
{
	if(GetC4DVersion() < 8500)
	{
		MessageDialog("This version of C4Dfx needs Cinema 4D 8.5 or later.");
		return FALSE;
	}

	WinInit::StartWin();

	if (!RegisterLightTag()) return FALSE;
	if (!RegisterPrefs()) return FALSE;
	if (!RegisterFXMaterial()) return FALSE;
	if (!RegisterFXCommand()) return FALSE;
	return TRUE;
}

void PluginEnd(void)
{
	FreeFXCommand();
	FreePrefs();
}

Bool PluginMessage(LONG id, void *data)
{
	switch (id)
	{
		case C4DPL_INIT_SYS:
			if (!resource.Init()) return FALSE; // don't start plug-in without resource
			return TRUE;

		case C4DMSG_PRIORITY: 
			return TRUE;
	}
	return FALSE;
}
