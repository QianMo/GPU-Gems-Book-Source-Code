// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes to control threads for rendering

#include <windows.h>
#include <glh/glh_extensions.h>
#include <GL/glu.h>
#include <GL/gl_extensions.h>
#include "CgFX/ICgFXEffect.h"
#include "C4DWrapper.h"
#include "LightsMaterialsObjects.h"
#include "FXWrapper.h"
#include "WinInit.h"
#include "WindowAdapter.h"
#include "WinHack.h"
#include "Render.h"
#include "Paint.h"
#include <stdio.h>
#include <assert.h>

struct FileRendererOpenGLStuff
{
	HDC hDC;
	HPBUFFERARB renderBuf;
	HDC renderhDC;
	HGLRC renderhRC;
	DWORD tStart;

	FileRendererOpenGLStuff(void);
};

FileRendererOpenGLStuff::FileRendererOpenGLStuff(void)
: hDC(0), renderBuf(0), renderhDC(0), renderhRC(0)
{
	tStart = timeGetTime();
}

FileRenderer::FileRenderer(BaseDocument* doc, int antialiasing, int mapSize)
: valid(false), antialiasing(antialiasing), mapSize(mapSize),
doc(doc), lineWidth(-1000), pic(0), lights(0), materials(0), shadows(0), froglst(0)
{}

FileRenderer::~FileRenderer(void)
{
	movie.Close();

	delete b;
	b = 0;

	delete[] pic;
	pic = 0;

	delete materials;
	materials = 0;

	delete shadows;
	shadows = 0;

	delete lights;
	lights = 0;

	if(froglst != 0)
	{
		if(froglst->renderBuf != 0)
		{
			if(froglst->renderhDC != 0)
			{
				wglMakeCurrent(froglst->renderhDC, NULL);
				if(froglst->renderhRC != 0)
				{
					wglDeleteContext(froglst->renderhRC);
					wglReleasePbufferDCARB(froglst->renderBuf, froglst->renderhDC);
				}
			}
			wglDestroyPbufferARB(froglst->renderBuf);
		}
	}

	DWORD tEnd = timeGetTime();
	char ts[16];
	sprintf(ts, "%8.3f s", (tEnd - froglst->tStart)*0.001);
	C4DWrapper::Print(ts);
}

bool FileRenderer::Initialize(void* hDC)
{
	froglst = new FileRendererOpenGLStuff();
	if(froglst == 0)
	{
		C4DWrapper::MsgBox("Couldn't allocate memory for renderer.");
		return false;
	}
	froglst->hDC = (HDC)hDC;

	width = C4DWrapper::GetRenderWidth(doc);
	height = C4DWrapper::GetRenderHeight(doc);

	lineWidth = 4*(int)(0.9 + (3*width)/4.0);
	pic = new unsigned char[lineWidth*height];
	if(pic == 0)
	{
		C4DWrapper::MsgBox("Couldn't allocate memory for double buffering.");
		return false;
	}

	fpsDoc = C4DWrapper::GetFps(doc);
	fpsRender = C4DWrapper::GetFpsRender(doc);
	from = C4DWrapper::GetStartFrame(doc);
	to = C4DWrapper::GetEndFrame(doc);

	frame = startFrame = (int)((from*fpsRender)/(double)fpsDoc);
	numFramesMinusOne = (int)(((to-from)*fpsRender)/(double)fpsDoc);
	endFrame = startFrame + numFramesMinusOne;
		
	if(!C4DWrapper::CheckFormat(doc))
	{
		C4DWrapper::MsgBox("Only user-defined AVI can be used as output format.");
		return false;
	}

	long pathLength = C4DWrapper::GetRenderPathLength(doc);
	if(pathLength == 0)
	{
		C4DWrapper::MsgBox("No path specified for saving output.");
		return false;
	}

	b = new BitmapWrapper(width, height);
	if(b == 0)
	{
		C4DWrapper::MsgBox("Couldn't allocate output bitmap.");
		return false;
	}

	if(!movie.Open(doc, b))
	{
		C4DWrapper::MsgBox("Couldn't open movie file.");
		return false;
	}

	BOOL status;
	int pixelFormat;
	unsigned int numFormats;
	int iAttributes[30];
	float fAttributes[] = {0, 0};
	iAttributes[0] = WGL_DRAW_TO_PBUFFER_ARB;
	iAttributes[1] = GL_TRUE;
	iAttributes[2] = WGL_ACCELERATION_ARB;
	iAttributes[3] = WGL_FULL_ACCELERATION_ARB;
	iAttributes[4] = WGL_COLOR_BITS_ARB;
	iAttributes[5] = 24;
	iAttributes[6] = WGL_ALPHA_BITS_ARB;
	iAttributes[7] = 8;
	iAttributes[8] = WGL_DEPTH_BITS_ARB;
	iAttributes[9] = 24;

	iAttributes[10] = (antialiasing==0 ? 0 : WGL_SAMPLE_BUFFERS_ARB); // stop here if no antialiasing
	iAttributes[11] = TRUE;
	iAttributes[12] = WGL_SAMPLES_ARB;
	iAttributes[13] = (1<<antialiasing);
	iAttributes[14] = 0;

	status = wglChoosePixelFormatARB(froglst->hDC, iAttributes,
					fAttributes, 1, &pixelFormat, &numFormats);
	if (status != GL_TRUE || numFormats == 0)
	{
		C4DWrapper::MsgBox("No suitable OpenGL format found.");
		return false;
	}

	const int pRenderBuf[] = { 0 };

	froglst->renderBuf = wglCreatePbufferARB(froglst->hDC, pixelFormat, width, height, pRenderBuf);
	if(froglst->renderBuf == 0)
	{
		C4DWrapper::MsgBox("Couldn't create offscreen buffer.");
		return false;
	}

	froglst->renderhDC = wglGetPbufferDCARB(froglst->renderBuf);
	if(froglst->renderhDC == 0)
	{
		C4DWrapper::MsgBox("Couldn't create device context.");
		return false;
	}
	froglst->renderhRC = wglCreateContext(froglst->renderhDC);
	if(froglst->renderhRC == 0)
	{
		C4DWrapper::MsgBox("Couldn't create offscreen render context.");
		return false;
	}

	if(!wglMakeCurrent(froglst->renderhDC, froglst->renderhRC))
	{
		C4DWrapper::MsgBox("Couldn't switch to offscreen render context.");
		return false;
	}
	glViewport(0, 0, width, height);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	if(antialiasing != 0 && WinInit::HasMultisampleFilterHintNV())
		glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

	if(FAILED(CgFXSetDevice("OpenGL", 0)))
	{
		C4DWrapper::MsgBox("Switching CgFX to OpenGL device failed.");
		return false;
	}
	lights = new Lights(doc);
	if(lights == 0)
	{
		C4DWrapper::MsgBox("Couldn't allocate lights.");
		return false;
	}
	shadows = new ShadowMaps(froglst->hDC, lights, mapSize);
	if(shadows == 0)
	{
		C4DWrapper::MsgBox("Couldn't allocate shadows.");
		return false;
	}
	materials = new Materials(doc, lights, mapSize, shadows);
	if(materials == 0)
	{
		C4DWrapper::MsgBox("Couldn't allocate materials.");
		return false;
	}

	valid = true; // if we return before, valid is false

	return valid;
}

BaseBitmap* FileRenderer::Draw(void)
{
	if(!valid)
		return 0;

	double w = width;
	double h = height;

	 // limit target size to 400 x 300
	if(w > 400.0)
	{
		h *= 400.0/w;
		w = 400.0;
	}
	if(h > 300.0)
	{
		w *= 300.0/h;
		h = 300.0;
	}

	return b->ScaledClone((long)(w+0.5), (long)(h+0.5));
}

// true if last frame
FileRenderer::State FileRenderer::NextFrame(void)
{
	if(!valid)
		return FX_ERROR; // error message already given

	assert(frame <= endFrame);

	if(numFramesMinusOne != 0)
		C4DWrapper::SetFrame(doc, (from + (to-from)*(frame-startFrame)/(double)numFramesMinusOne)/(double)fpsDoc);
	else
		C4DWrapper::SetFrame(doc, to/(double)fpsDoc);

	shadows->Render(doc, materials);

	if(! wglMakeCurrent(froglst->renderhDC, froglst->renderhRC))
	{
		C4DWrapper::MsgBox("Couldn't switch back to main render context.");
		valid = false;
	}
	else
	{
		materials->BeginRendering();
		Paint(doc, materials, shadows);
		glFinish();
		materials->EndRendering();

		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pic);

		long y;
		for(y=0; y<height; ++y)
		{
			b->SetLine(y, pic + lineWidth*y);
		}

		if(!movie.Write(b))
		{
			C4DWrapper::MsgBox("Couldn't write frame to movie.");
			valid = false;
		}
	}

	++frame;

	if(!valid)
		return FX_ERROR;

	if(frame > endFrame)
		return FX_END;
	else
		return FX_HAS_MORE;
}

RenderThread::RenderThread(WindowAdapter* wa, BaseDocument* doc, int antialiasing, int mapSize)
: wa(wa), doc(doc), antialiasing(antialiasing), mapSize(mapSize), running(false), ending(false)
{}

RenderThread::~RenderThread(void)
{}

bool RenderThread::IsRunning(void)
{
	return running;
}

void RenderThread::End(void)
{
	ending = true;
	while(running)
		Sleep(100);
}

RenderFileThread::RenderFileThread(WindowAdapter* wa, BaseDocument* doc, int antialiasing, int mapSize)
: RenderThread(wa, doc, antialiasing, mapSize)
{}

RenderFileThread::~RenderFileThread(void)
{}

static DWORD WINAPI FileRenderThreadFunc(LPVOID rft)
{
	RenderFileThread* rft_ = (RenderFileThread*) rft;
	 
	if(!WinInit::IsOpenGLOK())
	{
		C4DWrapper::MsgBox("Failing due to error on program startup.");
		rft_->wa->MarkAsClosing();
	}
	else
	{
		HINSTANCE hInstance = GetInstance();
		HWND hWndRend = CreateWindowEx(WS_EX_TOPMOST, "C4Dfx", "C4Dfx",
			WS_OVERLAPPEDWINDOW, 0, 0, 200, 150, NULL, NULL, hInstance, NULL);
		if(hWndRend == NULL)
		{
			C4DWrapper::MsgBox("Could not create renderer window.");
			rft_->wa->MarkAsClosing();
		}
		else
		{
			FileRenderer* fr = new FileRenderer(rft_->doc, rft_->antialiasing, rft_->mapSize);	
			if(fr == NULL)
			{
				C4DWrapper::MsgBox("Could not allocate renderer.");
				rft_->wa->MarkAsClosing();
			}
			else
			{
				HDC hDC = GetDC(hWndRend);
				if(hDC == NULL)
				{
					C4DWrapper::MsgBox("Could not initialize renderer context.");
					rft_->wa->MarkAsClosing();
				}
				else			
				{
					if(! fr->Initialize((void*)hDC))
					{
						// error message already given in Initialize
						rft_->wa->MarkAsClosing();
					}
					else
					{
						long result = FileRenderer::FX_HAS_MORE;
						while(!rft_->ending && result == FileRenderer::FX_HAS_MORE)
						{
							result = fr->NextFrame();
							if(result == FileRenderer::FX_ERROR)
							{
								rft_->wa->MarkAsClosing();
								break; // error message already given
							}
							rft_->wa->ReadNewBitmap(fr->Draw(), fr->GetFrameCurrent()-1, fr->GetFrameTotal(), 0);
							// minus one because we show the frame before, not the one that is rendered next
						}
					}
				}
				delete fr;
			}
			BOOL result = DestroyWindow(hWndRend);
			assert(result);
		}
	}

	C4DWrapper::FreeDoc(rft_->doc);

	rft_->running = false;
	return 0;
}

bool RenderFileThread::Start(void)
{
	DWORD dwFileRenderThreadId = 0;
	HANDLE hFileRenderThread = NULL; 

	running = true;
	hFileRenderThread = CreateThread(NULL, 0, FileRenderThreadFunc, this, 0, &dwFileRenderThreadId);
	if(hFileRenderThread == NULL)
	{
		running = false;
		C4DWrapper::MsgBox("Could not create renderer thread.");
		return false;
	}
	CloseHandle(hFileRenderThread);
	return true;
}

RenderWinThread::RenderWinThread(WindowAdapter* wa, BaseDocument* doc, int antialiasing, int mapSize, long jobNumber)
: RenderThread(wa, doc, antialiasing, mapSize), jobNumber(jobNumber)
{}

RenderWinThread::~RenderWinThread(void)
{}

static DWORD WINAPI WinRenderThreadFunc(LPVOID rwt)
{
	RenderWinThread* rwt_ = (RenderWinThread*) rwt;

	HINSTANCE hInstance = GetInstance();
	HDC hDC = 0;
	HPBUFFERARB renderBuf = 0;
	HDC renderhDC = 0;
	HGLRC renderhRC = 0;
	DWORD tStart = timeGetTime();

	long width;
	long height;
	long lineWidth;

	unsigned char* pic;
	BitmapWrapper* b;
	Lights* lights;
	Materials* materials;
	ShadowMaps* shadows;
	 
	if(!WinInit::IsOpenGLOK())
	{
		C4DWrapper::MsgBox("Failing due to error on program startup.");
	}
	else
	{
		HWND hWndRend = CreateWindowEx(WS_EX_TOPMOST, "C4Dfx", "C4Dfx",
			WS_OVERLAPPEDWINDOW, 0, 0, 200, 150, NULL, NULL, hInstance, NULL);
		if(hWndRend ==NULL)
		{
			C4DWrapper::MsgBox("Could not create renderer window.");
		}
		else
		{
			hDC = GetDC(hWndRend);
			if(hDC == NULL)
			{
				C4DWrapper::MsgBox("Could not initialize renderer context.");
				goto error;
			}

			{
				// window dimensions
				LONG w = rwt_->wa->GetWidth();
				LONG h = rwt_->wa->GetHeight();
				// render dimensions, later contains the actual rendering dimensions (maybe letterbox etc.)
				LONG rw = C4DWrapper::GetRenderWidth(rwt_->doc);
				LONG rh = C4DWrapper::GetRenderHeight(rwt_->doc);
				if(rw==0 && rh==0 && w==0 && h== 0) // nothing to draw
					goto error;

				// Which is the largest rectangle with the ratio width/height that fits into the window?
				float ratioWin = h/(float)w;
				float ratioRender = rh/(float)rw;

				int widthSpace = 0;
				int heightSpace = 0;
				
				if(ratioRender > ratioWin) // align top and bottom
				{
					widthSpace = (LONG)(0.5*(w - h/ratioRender));
				}
				else // align left and right
				{
					heightSpace = (LONG)(0.5*(h - w*ratioRender));
				}

				width = w - 2*widthSpace;
				height = h - 2*heightSpace;

				lineWidth = 4*(int)(0.9 + (3*width)/4.0);
				pic = new unsigned char[lineWidth*height];
			}
			if(pic == 0)
			{
				C4DWrapper::MsgBox("Couldn't allocate memory for double buffering.");
				goto error;
			}

			b = new BitmapWrapper(width, height);
			if(b == 0)
			{
				C4DWrapper::MsgBox("Couldn't allocate output bitmap.");
				goto error;
				return false;
			}

			{
				BOOL status;
				int pixelFormat;
				unsigned int numFormats;
				int iAttributes[30];
				float fAttributes[] = {0, 0};
				iAttributes[0] = WGL_DRAW_TO_PBUFFER_ARB;
				iAttributes[1] = GL_TRUE;
				iAttributes[2] = WGL_ACCELERATION_ARB;
				iAttributes[3] = WGL_FULL_ACCELERATION_ARB;
				iAttributes[4] = WGL_COLOR_BITS_ARB;
				iAttributes[5] = 24;
				iAttributes[6] = WGL_ALPHA_BITS_ARB;
				iAttributes[7] = 8;
				iAttributes[8] = WGL_DEPTH_BITS_ARB;
				iAttributes[9] = 24;

				iAttributes[10] = (rwt_->antialiasing==0 ? 0 : WGL_SAMPLE_BUFFERS_ARB); // stop here if no antialiasing
				iAttributes[11] = TRUE;
				iAttributes[12] = WGL_SAMPLES_ARB;
				iAttributes[13] = (1<<rwt_->antialiasing);
				iAttributes[14] = 0;

				status = wglChoosePixelFormatARB(hDC, iAttributes,
								fAttributes, 1, &pixelFormat, &numFormats);
				if (status != GL_TRUE || numFormats == 0)
				{
					C4DWrapper::MsgBox("No suitable OpenGL format found.");
					goto error;
				}

				const int pRenderBuf[] = { 0 };

				renderBuf = wglCreatePbufferARB(hDC, pixelFormat, width, height, pRenderBuf);
			}
			if(renderBuf == 0)
			{
				C4DWrapper::MsgBox("Couldn't create offscreen buffer.");
				goto error;
			}

			renderhDC = wglGetPbufferDCARB(renderBuf);
			if(renderhDC == 0)
			{
				C4DWrapper::MsgBox("Couldn't create device context.");
				goto error;
			}
			renderhRC = wglCreateContext(renderhDC);
			if(renderhRC == 0)
			{
				C4DWrapper::MsgBox("Couldn't create offscreen render context.");
				goto error;
			}

			if(!wglMakeCurrent(renderhDC, renderhRC))
			{
				C4DWrapper::MsgBox("Couldn't switch to offscreen render context.");
				goto error;
			}
			glViewport(0, 0, width, height);
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			if(rwt_->antialiasing != 0 && WinInit::HasMultisampleFilterHintNV())
				glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

			if(FAILED(CgFXSetDevice("OpenGL", 0)))
			{
				C4DWrapper::MsgBox("Switching CgFX to OpenGL device failed.");
				goto error;
			}
			lights = new Lights(rwt_->doc);
			if(lights == 0)
			{
				C4DWrapper::MsgBox("Couldn't allocate lights.");
				goto error;
			}
			shadows = new ShadowMaps(hDC, lights, rwt_->mapSize);
			if(shadows == 0)
			{
				C4DWrapper::MsgBox("Couldn't allocate shadows.");
				goto error;
			}
			materials = new Materials(rwt_->doc, lights, rwt_->mapSize, shadows);
			if(materials == 0)
			{
				C4DWrapper::MsgBox("Couldn't allocate materials.");
				goto error;
			}

			shadows->Render(rwt_->doc, materials);

			if(! wglMakeCurrent(renderhDC, renderhRC))
			{
				C4DWrapper::MsgBox("Couldn't switch back to offscreen render context.");
				goto error;
			}
			materials->BeginRendering();
			Paint(rwt_->doc, materials, shadows);
			glFinish();
			materials->EndRendering();

			glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pic);

			long y;
			for(y=0; y<height; ++y)
			{
				b->SetLine(y, pic + lineWidth*y);
			}
			rwt_->wa->ReadNewBitmap(b->ExtractBitmap(), 42, -2, rwt_->jobNumber);

		error:
			delete b;
			b = 0;

			delete[] pic;
			pic = 0;

			delete materials;
			materials = 0;

			delete shadows;
			shadows = 0;

			delete lights;
			lights = 0;

			if(renderBuf != 0)
			{
				if(renderhDC != 0)
				{
					wglMakeCurrent(renderhDC, NULL);
					if(renderhRC != 0)
					{
						wglDeleteContext(renderhRC);
						wglReleasePbufferDCARB(renderBuf, renderhDC);
					}
				}
				wglDestroyPbufferARB(renderBuf);
			}

			BOOL result = DestroyWindow(hWndRend);
			assert(result);
		}
	}

	C4DWrapper::FreeDoc(rwt_->doc);

	DWORD tEnd = timeGetTime();
	char ts[16];
	sprintf(ts, "%8.3f s", (tEnd - tStart)*0.001);
	C4DWrapper::Print(ts);

	rwt_->running = false;
	return 0;
}

bool RenderWinThread::Start(void)
{
	DWORD dwWinRenderThreadId = 0;
	HANDLE hWinRenderThread = NULL; 

	running = true;
	hWinRenderThread = CreateThread(NULL, 0, WinRenderThreadFunc, this, 0, &dwWinRenderThreadId);
	if(hWinRenderThread == NULL)
	{
		running = false;
		C4DWrapper::MsgBox("Could not create renderer thread.");
		return false;
	}
	CloseHandle(hWinRenderThread);
	return true;
}