// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes to wrap Cinema 4D's bitmap and movie file saving functions

#include "c4d.h"
#include "BitmapWrapper.h"

BitmapWrapper::BitmapWrapper(long width, long height)
: w(width), h(height)
{
	bmp = BaseBitmap::Alloc(); // may be NULL
	if(bmp != NULL)
	{
		if(bmp->Init(w, h) != IMAGE_OK)
		{
			BaseBitmap::Free(bmp); // sets bmp = NULL	
		}
	}
}

BitmapWrapper::~BitmapWrapper(void)
{
	BaseBitmap::Free(bmp);
}

void BitmapWrapper::SetLine(long y, void* line)
{
	if(bmp != NULL)
		bmp->SetLine(h-y, line, 24);
}

BaseBitmap* BitmapWrapper::ScaledClone(long w, long h)
// may return NULL
{
	BaseBitmap* b = BaseBitmap::Alloc();
	if(b != NULL)
	{
		if(b->Init(w, h) == IMAGE_OK)
		{
			bmp->ScaleIt(b, 256, FALSE, TRUE);
		}
		else
		{
			b->Free(b);
		}
	}
	return b;
}

BaseBitmap* BitmapWrapper::ExtractBitmap(void)
// caller has to dispose BaseBitmap
{
	BaseBitmap* b = bmp;
	bmp = NULL;
	return b;
}

MovieSaverWrapper::MovieSaverWrapper(void)
{
	ms = MovieSaver::Alloc(); // may be NULL
}

MovieSaverWrapper::~MovieSaverWrapper(void)
{
	MovieSaver::Free(ms);
}

bool MovieSaverWrapper::Open(BaseDocument* doc, BitmapWrapper* b)
{
	if(ms == NULL)
		return false;

	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	if(c == NULL)
		return false;

	BaseContainer* settings = gNew BaseContainer();
	if(settings == NULL)
		return false;

	if(	   ! settings->SetData(AVISAVER_FCCTYPE, c->GetData(AVISAVER_FCCTYPE))
		|| ! settings->SetData(AVISAVER_FCCHANDLER, c->GetData(AVISAVER_FCCHANDLER))
		|| ! settings->SetData(AVISAVER_LKEY, c->GetData(AVISAVER_LKEY))
		|| ! settings->SetData(AVISAVER_LDATARATE, c->GetData(AVISAVER_LDATARATE))
		|| ! settings->SetData(AVISAVER_LQ, c->GetData(AVISAVER_LQ))
	)
	{
		gDelete(settings);
		return false;
	}

	bool result = (IMAGE_OK == ms->Open(c->GetFilename(RDATA_PATH), b->bmp,
		c->GetLong(RDATA_FRAMERATE), FILTER_AVI_USER, settings, 0L));

	gDelete(settings);
	return result;
}

bool MovieSaverWrapper::Write(BitmapWrapper* b)
{
	if(ms == NULL)
		return false;

	return ms != NULL && ms->Write(b->bmp) == IMAGE_OK;
}

void MovieSaverWrapper::Close(void)
{
	if(ms != NULL)
		ms->Close();
}