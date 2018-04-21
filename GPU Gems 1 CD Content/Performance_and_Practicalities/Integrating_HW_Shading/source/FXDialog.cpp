// from C4Dfx by Jörn Loviscach, www.l7h.cn
// the class for the menu function to call C4Dfx

#include "c4d.h"
#include "c4d_symbols.h"
#include "IDnumbers.h"
#include "FXDialog.h"
#include <assert.h>

FXDialog::FXDialog(void)
{}

FXDialog::~FXDialog(void)
{
	assert(! rww.IsOpen()); // interactive renderer already closed
	assert(windowSet.GetIndexId(0L) == NOTOK); // list of file renderer windows already empty
}

Bool FXDialog::CreateLayout(void)
{
	return LoadDialogResource(DLG_FXRENDERER, NULL, 0);
}

Bool FXDialog::InitValues(void)
{
	autoUpdate = TRUE;
	antialiasing = IDC_AA0;
	mapSize = IDC_MS256;

	SetTimer(1000); // time interval for looking after to-be-closed file renderer windows

	Bool result;
	result = SetBool(IDC_RENDER_RT, autoUpdate);
	result = SetLong(IDC_ANTIALIASING_RADIO, antialiasing) && result;
	result = SetLong(IDC_MAPSIZE_RADIO, mapSize) && result;
	return result;
}

LONG FXDialog::ComputeSize(LONG button)
{
	switch(button)
	{
		case IDC_MS64: return 64;
		case IDC_MS128: return 128;
		case IDC_MS256: return 256;
		case IDC_MS512: return 512;
		case IDC_MS1024: return 1024;
		default: assert(FALSE); return 256;
	}
}

Bool FXDialog::Command(LONG id, const BaseContainer& msg)
{
	if(!GetBool(IDC_RENDER_RT, autoUpdate) )
		return FALSE;
	if(!GetLong(IDC_ANTIALIASING_RADIO, antialiasing) )
		return FALSE;
	if(!GetLong(IDC_MAPSIZE_RADIO, mapSize) )
		return FALSE;

	switch(id)
	{
		case IDC_RENDER_FILE:
			{
				RenderFileWindow* rfw = gNew RenderFileWindow();
				if(rfw == NULL)
				{
					MessageDialog("Could not allocate window.");
					return FALSE;
				}
				else
				{
					if(! rfw->StartRender(GetActiveDocument(), antialiasing-IDC_AA0, ComputeSize(mapSize)))
					{
						MessageDialog("Could not open window.");
						rfw->Close();
						gDelete(rfw);
						return FALSE;
					}
					else
					{
						assert(windowSet.FindIndex((LONG)rfw) == NOTOK); // not already there
						windowSet.SetLong((LONG)rfw, 42L); // abusing BaseContainer to store addresses; this will not work on a 64 bit system!
					}
				}

				return TRUE;
			}
			break;
		case IDC_RENDER_WINDOW:
			return !rww.StartRender(GetActiveDocument(), antialiasing-IDC_AA0, ComputeSize(mapSize));
			// error message already handled by StartRender
			break;
		case IDC_RENDER_RT:
			if(autoUpdate && rww.IsOpen()) // has been switched on with the current mouse click
				return !rww.StartRender(GetActiveDocument(), antialiasing-IDC_AA0, ComputeSize(mapSize));
				// error message already handled by StartRender
			break;
		case IDC_ANTIALIASING_RADIO:
		case IDC_MAPSIZE_RADIO:
			if(autoUpdate && rww.IsOpen())
				return !rww.StartRender(GetActiveDocument(), antialiasing-IDC_AA0, ComputeSize(mapSize));
				// error message already handled by StartRender
			break;
		default: ; // some uninteresting command
	}

	return TRUE;
}

Bool FXDialog::CoreMessage(LONG id, const BaseContainer& msg)
{
	switch(id)
	{
		case EVMSG_CHANGE:
			if(autoUpdate && rww.IsOpen())
				return !rww.StartRender(GetActiveDocument(), antialiasing-IDC_AA0, ComputeSize(mapSize));
				// error message already handled by StartRender
			break;
		default: ; // some uninteresting core message
	}
	return TRUE;
}

void FXDialog::DestroyWindow(void)
{
	rww.Close();

	LONG i;
	LONG id;
	RenderFileWindow* rfw = NULL;
	for(i = 0; (id = windowSet.GetIndexId(i)) != NOTOK; ++i)
	{
		RenderFileWindow* rfw = (RenderFileWindow*)id;
		rfw->Close();
		gDelete(rfw);
	}
	windowSet.FlushAll();
}

void FXDialog::Timer(const BaseContainer& msg) // "garbage collection" concerning to-be-closed windows
{
	LONG i = 0;
	LONG id;
	RenderFileWindow* rfw = NULL;
	while( (id = windowSet.GetIndexId(i)) != NOTOK )
	{
		RenderFileWindow* rfw = (RenderFileWindow*)id;

		if(rfw->IsClosing()) // is to be closed
		{
			rfw->Close();
			++i;
		}
		else
		{
			if(rfw->HasBeenStarted() && !rfw->IsOpen()) // has been closed
			{
				gDelete(rfw);
				windowSet.RemoveIndex(i); // next index is advanced to position i, so don't increment i
			}
			else
				++i;
		}
	}
}