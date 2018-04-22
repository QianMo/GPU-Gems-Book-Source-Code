// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a structure to let the thread communicate with its window, hiding the #includes to avoid conflicts

#if !defined(WINDOW_ADAPTER_H)
#define WINDOW_ADAPTER_H

class BaseBitmap;
class RenderWindow;

struct WindowAdapter
{
		WindowAdapter(void);
		void ReadNewBitmap(BaseBitmap* b, long frame, long totalFrame, long jobNumber);
		void MarkAsClosing(void);
		long GetWidth(void);
		long GetHeight(void);
		RenderWindow* rw;
};

#endif