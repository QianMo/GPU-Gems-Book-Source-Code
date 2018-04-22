// from C4Dfx by Jörn Loviscach, www.l7h.cn
// the class for the menu function to call C4Dfx

#include "c4d.h"
#include "c4d_symbols.h"
#include "IDnumbers.h"
#include "FXDialog.h"

class FXCommand : public CommandData
{
	public:
		virtual Bool Execute(BaseDocument *doc);
};

static FXDialog* theDialog = NULL;

Bool FXCommand::Execute(BaseDocument *doc)
{
	if(theDialog == NULL)
		theDialog = gNew FXDialog;

	if(theDialog == NULL)
		return FALSE;

	if(!theDialog->Open(TRUE, NULL))
		return FALSE;

	return TRUE;
}

Bool RegisterFXCommand(void)
{
	String name = GeLoadString(IDS_FXCOMMAND);
	if (!name.Content()) return FALSE;
	return RegisterCommandPlugin(ID_COMMAND_PLUGIN, name + "...", 0, "fx32.tif", name, gNew FXCommand);
}

void FreeFXCommand(void)
{
	gDelete(theDialog);
}
