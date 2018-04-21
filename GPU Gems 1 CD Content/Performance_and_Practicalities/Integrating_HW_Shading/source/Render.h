// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes to control threads for rendering

#if !defined(RENDER_H)
#define RENDER_H

#include "BitmapWrapper.h"

class BaseDocument;
class BaseBitmap;
class Lights;
class Materials;
class ShadowMaps;
struct WindowAdapter;
struct FileRendererOpenGLStuff; // to hide the Windows/OpenGL data types

class FileRenderer
{
	public:
		FileRenderer(BaseDocument* doc, int antialiasing, int mapSize);
		~FileRenderer(void);
		bool Initialize(void* hDC); // cast to void* to hide type from C4D in include file
		int GetFrameTotal(void) {return numFramesMinusOne+1;}
		int GetFrameCurrent(void) {return frame-startFrame+1;}
		BaseBitmap* Draw(void);
		int GetWidth(void) {return width;}
		int GetHeight(void) {return height;}
		enum State {FX_HAS_MORE, FX_ERROR, FX_END};
		State NextFrame(void);
	private:
		FileRendererOpenGLStuff* froglst;
		bool valid;
		int antialiasing;
		int mapSize;
		long width;
		long height;
		long fpsDoc;
		long fpsRender;
		long from;
		long to;
		long format;
		int startFrame;
		int numFramesMinusOne;
		int endFrame;
		int frame;
		BaseDocument* doc;
		long lineWidth;
		unsigned char* pic;
		MovieSaverWrapper movie;
		BitmapWrapper* b;
		Lights* lights;
		Materials* materials;
		ShadowMaps* shadows;
};

class RenderThread
{
	public:
		RenderThread(WindowAdapter* wa, BaseDocument* doc, int antialiasing, int mapSize);
		virtual ~RenderThread(void);
		virtual bool Start(void) = 0;
		void End(void);
		bool IsRunning(void);
	// The following should be private, but the Win32 thread needs access.
		bool running;
		BaseDocument* doc;
		int antialiasing;
		int mapSize;
		WindowAdapter* wa;
		bool ending;
};

class RenderFileThread : public RenderThread
{
	public:
		RenderFileThread(WindowAdapter* wa, BaseDocument* doc, int antialiasing, int mapSize);
		~RenderFileThread(void); // do not inherit
		virtual bool Start(void);
};

class RenderWinThread : public RenderThread
{
	public:
		RenderWinThread(WindowAdapter* wa, BaseDocument* doc, int antialiasing, int mapSize, long jobNumber);
		~RenderWinThread(void); // do not inherit
		virtual bool Start(void);
	// The following should be private, but the Win32 thread needs access.
		long jobNumber;
};

#endif