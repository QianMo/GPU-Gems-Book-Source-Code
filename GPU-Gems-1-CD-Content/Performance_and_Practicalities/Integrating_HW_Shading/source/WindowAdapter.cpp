// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a structure to let the thread communicate with its window, hiding the #includes to avoid conflicts

#include "c4d.h"
#include "WindowAdapter.h"
#include "RenderDialog.h"

WindowAdapter::WindowAdapter(void)
: rw(0)
{}

void WindowAdapter::ReadNewBitmap(BaseBitmap* b, LONG frame, LONG totalFrame, long jobNumber)
{
	rw->ua.ReadNewBitmap(b, frame, totalFrame, jobNumber);
}

void WindowAdapter::MarkAsClosing(void)
{
	rw->MarkAsClosing();
}

long WindowAdapter::GetWidth(void)
{
	return rw->ua.GetWidth();
}

long WindowAdapter::GetHeight(void)
{
	return rw->ua.GetHeight();
}
