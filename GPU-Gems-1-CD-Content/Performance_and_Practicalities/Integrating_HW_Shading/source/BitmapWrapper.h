// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes to wrap Cinema 4D's bitmap and movie file saving functions

#if !defined(BITMAP_WRAPPER_H)
#define BITMAP_WRAPPER_H

class MovieSaverWrapper;
class BaseDocument;
class BaseBitmap;
class MovieSaver;
class RenderFileWindowUserArea;
class BaseContainer;

class BitmapWrapper
{
friend MovieSaverWrapper;

public:
	BitmapWrapper(long width, long height);
	~BitmapWrapper(void);
	void SetLine(long y, void* line);
	BaseBitmap* ScaledClone(long w, long h);
	BaseBitmap* ExtractBitmap(void);
private:
	long w, h;
	BaseBitmap* bmp;
};

class MovieSaverWrapper
{
public:
	MovieSaverWrapper(void);
	~MovieSaverWrapper(void);
	bool Open(BaseDocument* doc, BitmapWrapper* b);
	bool Write(BitmapWrapper* b);
	void Close(void);
private:
	MovieSaver* ms;
};

#endif