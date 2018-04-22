// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes to display windows for interactive and file rendering

#include "c4d.h"
#include "RenderDialog.h"
#include "IDnumbers.h"
#include "Render.h"
#include "assert.h"

RenderWindowUserArea::RenderWindowUserArea(void)
: w(1), h(1), doubleBuffer(0), rw(0), totalFrame(-1), jobNumber(-2), lastJobNumber(-1)
{
	bitmapSemaphore = Semaphore::Alloc(); // may be NULL, check on use
}

RenderWindowUserArea::~RenderWindowUserArea(void)
{
	if(bitmapSemaphore != NULL)
	{
		if(bitmapSemaphore->LockAndWait((BaseThread*)NULL))
		{
			doubleBuffer->Free(doubleBuffer);
		}
		bitmapSemaphore->Free(bitmapSemaphore);
	}
}

void RenderWindowUserArea::Draw(LONG x1, LONG y1, LONG x2, LONG y2)
{
	OffScreenOn();

	DrawSetPen(COLOR_BG);
	DrawRectangle(0, 0, w, h);

	if(totalFrame == -1)
	{
		DrawSetPen(COLOR_TEXT);
		DrawText("Rendering ...", 5, 5);
	}

	if(bitmapSemaphore != NULL && bitmapSemaphore->Lock())
	// in rare cases, this will yield false due to conflict; we'll live with that instead of risking complex deadlocks
	{
		if(doubleBuffer != 0)
		{
			LONG width = doubleBuffer->GetBw();
			LONG height = doubleBuffer->GetBh();
			if(width!=0 && height!=0 && w!=0 && h!= 0) // something to draw
			{
				// Which is the largest rectangle with the ratio width/height that fits into the window?
				Real ratioWin = h/(Real)w;
				Real ratioRender = height/(Real)width;

				LONG widthSpace = 0;
				LONG heightSpace = 0;
				
				if(ratioRender > ratioWin) // align top and bottom
				{
					widthSpace = (LONG)(0.5*(w - h/ratioRender));
				}
				else // align left and right
				{
					heightSpace = (LONG)(0.5*(h - w*ratioRender));
				}

				DrawBitmap(doubleBuffer, widthSpace, heightSpace, w-2*widthSpace, h-2*heightSpace, 0, 0,
					doubleBuffer->GetBw(), doubleBuffer->GetBh(), BMP_NORMAL);
			}
		}
		bitmapSemaphore->UnLock();
	}
	if(totalFrame == -2 && jobNumber != lastJobNumber)
	// cross through dirty image, but only in interactive window (totalFrame == -2)
	{
		DrawSetPen(Vector(1.0, 0.0, 0.0));
		DrawLine(0, 0, GetWidth(), GetHeight());
		DrawLine(GetWidth(), 0, 0, GetHeight());
	}

	if(totalFrame >= 0)
	{
		if(frame != totalFrame)
			rw->SetTitle("C4dfx: frame "+LongToString(frame)+" of "+LongToString(totalFrame));
		else
			rw->SetTitle("C4dfx: "+LongToString(totalFrame)+ " frames finished");
	}
}

void RenderWindowUserArea::Sized(LONG w, LONG h)
{
	this->w = w;
	this->h = h;
}

void RenderWindowUserArea::ReadNewBitmap(BaseBitmap* b, LONG frame, LONG totalFrame, LONG jobNumber)
// Caller loses possession of b! Don't invoke Free in caller!
// totalFrame < 0: no frame number written into title bar
// totalFrame == -1: "Rendering ..." written into client area
// Of course, this should not be done with special values but with additional boolean variables
{
	this->frame = frame;
	this->totalFrame = totalFrame;
	if(b != NULL && bitmapSemaphore != NULL)
	{
		if(bitmapSemaphore->Lock())
		// in rare cases, this will yield false due to conflict; we'll live with that instead of risking complex deadlocks
		{
			doubleBuffer->Free(doubleBuffer);
			doubleBuffer = b;
			this->jobNumber = jobNumber;
			Redraw(TRUE);
			bitmapSemaphore->UnLock();
		}
		else
		{
			b->Free(b);
		}
	}
}

void RenderWindowUserArea::SetLastJobNumber(LONG n)
{
	lastJobNumber = n;
}

RenderWindow::RenderWindow(void)
: rt(NULL), started(FALSE), closing(FALSE)
{
	wa.rw = this;
	ua.rw = this;
}

RenderWindow::~RenderWindow(void)
{
	assert(rt == NULL); // already destroyed
}

void RenderWindow::DestroyWindow(void)
{
	if(rt != NULL && rt->IsRunning())
		rt->End();
	gDelete(rt);
}

Bool RenderWindow::HasBeenStarted(void)
{
	return started;
}

Bool RenderWindow::CreateLayout(void)
{
	SetTitle("C4Dfx");
	C4DGadget* g = AddUserArea(20000, BFH_SCALEFIT | BFV_SCALEFIT, 120, 90);
	if(g == NULL)
	{
		MessageDialog("Couldn't create preview area.");
		return FALSE;
	}
	if(! AttachUserArea(ua, GadgetPtr(g), USERAREA_COREMESSAGE))
	{
		MessageDialog("Couldn't attach preview area.");
		return false;
	}
	return true;
}

Bool RenderWindow::IsClosing(void)
{
	return closing;
}

void RenderWindow::MarkAsClosing(void)
// to let the window close itself via the garbage collector of the FX dialog
{
	closing = TRUE;
}

RenderFileWindow::RenderFileWindow(void)
{}

RenderFileWindow::~RenderFileWindow(void)
{}

Bool RenderFileWindow::StartRender(BaseDocument* doc, int antialiasing, int mapSize)
{
	assert(doc != 0);
	BaseDocument* doc_ = (BaseDocument*)doc->GetClone(0L, NULL); // in the regular case, freeing is done in thread
	if(doc_ == NULL)
	{
		MessageDialog("Couldn't clone document for rendering.");
	}
	else
	{
		rt = gNew RenderFileThread(&wa, doc_, antialiasing, mapSize);
		if(rt == NULL)
		{
			MessageDialog("Couldn't create rendering thread.");
		}
		else
		{
			if(! rt->Start())
			{
				MessageDialog("Couldn't start rendering thread.");
			}
			else
			{
				if(! Open(TRUE, ID_COMMAND_PLUGIN, 400, 300, 300, 180))
				{
					MessageDialog("Couldn't open preview window.");
				}
				else
				{
					started = TRUE;
				}
			}
		}
	}

	if(!started)
	{	
		if(rt != NULL && rt->IsRunning())
			rt->End();
		gDelete(rt); // includes setting rt = 0;

		BaseDocument::Free(doc_);
	}

	return started;
}

RenderWinWindow::RenderWinWindow(void)
: doc(NULL), jobNumber(0)
{}

RenderWinWindow::~RenderWinWindow(void)
{}

Bool RenderWinWindow::InitValues(void)
{
	SetTimer(100); // to look after an update
	return RenderWindow::InitValues();
}

Bool RenderWinWindow::StartRender(BaseDocument* doc, int antialiasing, int mapSize)
{
	this->doc = doc;
	this->antialiasing = antialiasing;
	this->mapSize = mapSize;

	ua.SetLastJobNumber(++jobNumber);
	ua.Redraw();

	if(IsOpen())
	{
		return TRUE;
	}
	else
	{
		if(! Open(TRUE, ID_COMMAND_PLUGIN, 500, 400, 450, 300))
		{
			MessageDialog("Couldn't open renderer window.");
			return FALSE;
		}
		return TRUE;
	}
}

void RenderWinWindow::Timer(const BaseContainer& msg)
{
	// is there no update waiting?
	if(doc == NULL)
		return;

	// wait further if render thread is still running
	if(rt != NULL && rt->IsRunning())
		return;

	gDelete(rt);

	assert(doc != NULL);
	BaseDocument* doc_ = (BaseDocument*)doc->GetClone(0L, NULL); // in the regular case, freeing is done in thread
	doc = NULL;

	if(doc_ == NULL)
	{
		MessageDialog("Couldn't clone document for rendering.");
	}
	else
	{
		rt = gNew RenderWinThread(&wa, doc_, antialiasing, mapSize, jobNumber);
		if(rt == NULL)
		{
			MessageDialog("Couldn't create rendering thread.");
		}
		else
		{
			if(! rt->Start())
			{
				MessageDialog("Couldn't start rendering thread.");
			}
			else
			{
				started = TRUE;
			}
		}
	}

	if(!started)
	{	
		if(rt != NULL && rt->IsRunning())
			rt->End();
		gDelete(rt); // includes rt = 0;

		BaseDocument::Free(doc_);
	}
}