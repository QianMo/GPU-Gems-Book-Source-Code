// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes to display windows for interactive and file rendering

#if !defined(RENDER_DIALOG_H)
#define RENDER_DIALOG_H

#include "c4d.h"
#include "WindowAdapter.h"

class RenderThread;
class RenderWindow;

class RenderWindowUserArea : public GeUserArea
{
	friend RenderWindow;
	friend WindowAdapter;

	public:
		RenderWindowUserArea(void);
		~RenderWindowUserArea(void); // do not inherit
		virtual void Draw(LONG x1, LONG y1, LONG x2, LONG y2);
		virtual void Sized(LONG w, LONG h);
		void SetLastJobNumber(LONG n);
	private:
		void ReadNewBitmap(BaseBitmap* b, LONG frame, LONG totalFrame, LONG jobNumber);
		LONG w;
		LONG h;
		BaseBitmap* doubleBuffer;
		Semaphore* bitmapSemaphore;
		RenderWindow* rw;
		LONG frame;
		LONG totalFrame;
		LONG jobNumber;
		LONG lastJobNumber;
};

class RenderWindow : public GeDialog
{
	friend WindowAdapter;

	public:
		RenderWindow(void);
		virtual ~RenderWindow(void);
		virtual Bool StartRender(BaseDocument* doc, int antialiasing, int mapSize) = 0;
		virtual Bool CreateLayout(void);
		virtual void DestroyWindow(void);
		Bool HasBeenStarted(void);
		Bool IsClosing(void);
	protected:
		RenderWindowUserArea ua;
		WindowAdapter wa;
		Bool started;
		void MarkAsClosing(void);
		Bool closing;
		RenderThread* rt;
};

class RenderFileWindow : public RenderWindow
{
	public:
		RenderFileWindow(void);
		~RenderFileWindow(void); // do not inherit
		virtual Bool StartRender(BaseDocument* doc, int antialiasing, int mapSize);
};

class RenderWinWindow : public RenderWindow
{
	public:
		RenderWinWindow(void);
		~RenderWinWindow(void); // do not inherit
		virtual Bool InitValues(void);
		virtual Bool StartRender(BaseDocument* doc, int antialiasing, int mapSize);
		virtual void Timer(const BaseContainer& msg);
	private:
		BaseDocument* doc;
		int antialiasing;
		int mapSize;
		LONG jobNumber;
};

#endif