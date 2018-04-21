// Disable warning for loss of data
#pragma warning( disable : 4244 )
#pragma warning( disable : 4305 )

// Thin out unneeded Win32 baggage
#define WIN32_LEAN_AND_MEAN

//system libraries include files
#include <windows.h>            // Standard Windows include file
#include <GL/gl.h>				// OpenGL headers
#include <GL/glu.h>
#include <mmsystem.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include "point.h"
#include "texmanager.h"
#include "plane.h"

int tris=0;

int quit=0;

HWND mainwindow;
HDC GLOBhDC;
BOOL g_bActive=1;
HINSTANCE g_hInst;

int WINDOWX=640;
int WINDOWY=480;

int base;
int running;
int width,height;

#define XFIELD 50
#define ZFIELD 50

#define QUADSIZE 1
#define VTXSIZE 0.8
#define TEXDIVIDER 30
#define SPEED 2
#define WAVESIZE 0.003

float speed=15;
float rotspeed=1;


long time1,time2;

double fps;
int tps;

point ppos(0,0,0);
double pyaw=0;

float *vtx;
float *tex;
GLubyte *col;

int start=1;
long lasttime;

int mode=0;

texmanager tm;


LRESULT CALLBACK WndProc(HWND hWnd,UINT message,WPARAM wParam,LPARAM lParam);

int frames;
int XSIZE=50;
int ZSIZE=50;

// Set rendering options (lighting, quality, etc)
void SetRenderingOptions()
{
glClearColor(0.7f, 0.6f, 0.5, 1.0f);
glEnable(GL_DEPTH_TEST);
glShadeModel(GL_SMOOTH);
glEnable(GL_CULL_FACE);
glCullFace(GL_BACK);
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
glEnable(GL_TEXTURE_2D);
glTexEnvi(GL_TEXTURE,GL_TEXTURE_ENV_MODE,GL_MODULATE);
}


GLvoid glPrint(const char *fmt, ...)					// Custom GL "Print" Routine
{
char		text[256];								// Holds Our String
va_list		ap;										// Pointer To List Of Arguments

if (fmt == NULL)									// If There's No Text
	return;											// Do Nothing
va_start(ap, fmt);									// Parses The String For Variables
vsprintf(text, fmt, ap);						// And Converts Symbols To Actual Numbers
va_end(ap);											// Results Are Stored In Text

glPushAttrib(GL_LIST_BIT);							// Pushes The Display List Bits
glListBase(base - 32);								// Sets The Base Character to 32
glCallLists(strlen(text), GL_UNSIGNED_BYTE, text);	// Draws The Display List Text
glPopAttrib();										// Pops The Display List Bits
}

int octaves=5;

float elapsed;

float timer;

float func(float x,float z)
{
float y=0;

float factor=1;
float d=sqrt(x*x+z*z);
d=d/40;
if (d>1.5) d=1.5;
if (d<0) d=0;
for (int i=0;i<octaves;i++)
	{
	y-=	(factor)*VTXSIZE*d*cos(((float)timer*SPEED)+(1/factor)*x*z*WAVESIZE)+
			(factor)*VTXSIZE*d*cos(((float)timer*SPEED)+(1/factor)*x*z*WAVESIZE) ;
	factor=factor/2;		
	}
return y;
}




void compute_sea()
{
int i=0;
int j=0;

timer=(float)timeGetTime()/1000;
// first pass: render the sea bottom as a textured plane
tm.usetexture(0);
float size=4;
glBegin(GL_QUADS);
glColor3f(1,1,1);
glTexCoord2f(0,0);glVertex3f(0,-10.1,0);
glTexCoord2f(size,0);glVertex3f(XFIELD*QUADSIZE,-10.1,0);
glTexCoord2f(size,size);glVertex3f(XFIELD*QUADSIZE,-10.1,ZFIELD*QUADSIZE);
glTexCoord2f(0,size);glVertex3f(0,-10.1,ZFIELD*QUADSIZE);
glEnd();

// second pass: caustic on top of the floor as an additive blend
glEnable(GL_BLEND);
glDepthMask(GL_FALSE);
glBlendFunc(GL_ONE,GL_ONE);
glDepthFunc(GL_LEQUAL);
	
for (int xi=0;xi<XFIELD;xi++)
	{
	tm.usetexture(1);
	glBegin(GL_TRIANGLE_STRIP);

	for (int zi=0;zi<ZFIELD;zi++)
		{
		// compute caustic environment mapping for point 1 in the strip
		point p(xi*QUADSIZE,0,zi*QUADSIZE);p.y=func(xi,zi);
		point q((xi+1)*QUADSIZE,0,zi*QUADSIZE);q.y=func(xi+1,zi);
		point r((xi)*QUADSIZE,0,(zi+1)*QUADSIZE);r.y=func(xi,zi+1);
		
		point e1=q-p;
		point e2=r-p;
		point e3=e1^e2;	// e3 is the normal of the sea above the sampling point #1


		plane pl(0,-1,0,30);
		point res;
		pl.testline(p,e3,res);
		// compute the collision to the lightmap

		glTexCoord2f(res.x/TEXDIVIDER,res.z/TEXDIVIDER);
		glVertex3f(p.x,-10,p.z);

		// compute caustic environment mapping for point 2 in the strip (shift 1 in xi)
		p=q;
		q.create((xi+2)*QUADSIZE,0,zi*QUADSIZE);q.y=func(xi+2,zi);
		r.create((xi+1)*QUADSIZE,0,(zi+1)*QUADSIZE);r.y=func(xi+1,zi+1);
		
		e1=q-p;
		e2=r-p;
		e3=e1^e2;
		// e3 is the normal of the sea above the sampling point #2

		pl.testline(p,e3,res);
		// compute the collision to the lightmap

		
		glTexCoord2f(res.x/TEXDIVIDER,res.z/TEXDIVIDER);
		glVertex3f(p.x,-10,p.z);

		}
	glEnd();
	}
glDepthMask(GL_TRUE);
glEnable(GL_BLEND);

// third pass: environment mapping. I could use automatic generation, but I felt like coding this
glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

for (xi=0;xi<XFIELD;xi++)
	{	

	tm.usetexture(2);
	glBegin(GL_TRIANGLE_STRIP);

	for (int zi=0;zi<ZFIELD;zi++)
		{
		point p(xi*QUADSIZE,0,zi*QUADSIZE);p.y=func(xi,zi);
		point q((xi+1)*QUADSIZE,0,zi*QUADSIZE);q.y=func(xi+1,zi);
		point r((xi)*QUADSIZE,0,(zi+1)*QUADSIZE);r.y=func(xi,zi+1);


		point e1=q-p;
		point e2=r-p;
		point e3;
		e3=e1^e2;

		plane pl(0,-1,0,10);
		point res;
		pl.testline(p,e3,res);
		float col=p.y*0.1+0.8;
		glColor4f(col,col,col,0.8);
		glTexCoord2f(res.x/TEXDIVIDER,res.z/TEXDIVIDER);
		glVertex3f(p.x,p.y,p.z);

		point pold=p;
		p=q;
		q.create((xi+2)*QUADSIZE,0,zi*QUADSIZE);q.y=func(xi+2,zi);
		r.create((xi+1)*QUADSIZE,0,(zi+1)*QUADSIZE);r.y=func(xi+1,zi+1);
		
		e1=q-p;
		e2=r-p;
		e3=e1^e2;

		pl.testline(p,e3,res);
		col=p.y*0.1+0.8;
		glColor4f(col,col,col,0.8);
	
		glTexCoord2f(res.x/TEXDIVIDER,res.z/TEXDIVIDER);
		glVertex3f(p.x,p.y,p.z);
		}
	glEnd();
	}
}


long last;

float pheight=0;
float zoom=20;

void HandleIdle()
{
tris=0;
elapsed=(float)(timeGetTime()-last)/1000;
last=timeGetTime();

if (!running) return;

if (GetAsyncKeyState(VK_ESCAPE)!=0) quit=1;
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
// draw UI

if (timeGetTime()-lasttime>10)
	{
	if (GetAsyncKeyState('W')!=0) mode=1;
	if (GetAsyncKeyState('S')!=0) mode=0;
	}

if (mode==0) 
	{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	}
else
	{
	glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glColor3f(0,0,0);
	glEnable(GL_TEXTURE_2D);
	}


glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45,1.3,1,50000);

pyaw+=rotspeed*elapsed*((GetAsyncKeyState(VK_RIGHT)!=0)-(GetAsyncKeyState(VK_LEFT)!=0));
zoom+=elapsed*speed*(GetAsyncKeyState(VK_DOWN)!=0)-(GetAsyncKeyState(VK_UP)!=0);
if (zoom<2) zoom=2;
pheight+=elapsed*speed*(GetAsyncKeyState('Q')!=0)-(GetAsyncKeyState('A')!=0);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
gluLookAt(zoom*cos(pyaw)+XFIELD/2,pheight,zoom*sin(pyaw)+ZFIELD/2,XFIELD/2,0,ZFIELD/2,0,1,0);


glEnable(GL_DEPTH_TEST);
glDisable(GL_CULL_FACE);

if (timeGetTime()-lasttime>300)
	{
	int oldoctaves=octaves;
	octaves+=(GetAsyncKeyState('P')!=0)-(GetAsyncKeyState('O')!=0);
	if (octaves<0) octaves=0;
	if (oldoctaves!=octaves) lasttime=timeGetTime();
	}


compute_sea();

glDisable(GL_DEPTH_TEST);
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluOrtho2D (0, width, 0, height);
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
glTranslatef (0.375, 0.375, 0.);
glRasterPos2f(20,20);

frames++;

if (frames%5==0)
	{
	fps=1000.0*5/(timeGetTime()-time1);
	time1=timeGetTime();
	tps=(int)(tris*fps);
	}
glPrint("fps: %f  tps:%d\n",fps,tps);
SwapBuffers(GLOBhDC);
}




//////////////////////////////////////////////////////////////
// Main entry point for the program
//////////////////////////////////////////////////////////////
int APIENTRY WinMain(HINSTANCE hInstance,HINSTANCE hPrevInstance,
					 LPSTR lpCmdLine,int nCmdShow)
{
MSG msg;            // Message Structure
WNDCLASS wc;        // Window Class structure

// Register Window style
wc.style        = CS_HREDRAW | CS_VREDRAW ;
wc.lpfnWndProc  = (WNDPROC) WndProc;
wc.cbClsExtra   = 0;
wc.cbWndExtra   = 0;
wc.hInstance    = hInstance;
wc.hIcon        = LoadIcon(hInstance, NULL);
wc.hCursor      = LoadCursor(NULL, NULL);
wc.hbrBackground= NULL;
wc.lpszMenuName = NULL;

wc.lpszClassName= "ANIMATE";

g_hInst=hInstance;

if (RegisterClass(&wc) == 0) return FALSE;

mainwindow = CreateWindow("animate","animate",WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_POPUP | WS_VISIBLE ,
							0,0,1024,768,	NULL, NULL, hInstance,NULL);

DWORD error;
if (mainwindow==NULL)
	error=GetLastError();

// create bitmaps for the device context font's first 256 glyphs 
wglUseFontBitmaps(GLOBhDC,0,256,1000); 
HFONT font;										// Windows Font ID
base = glGenLists(96);								// Storage For 96 Characters
font = CreateFont(	-10,							// Height Of Font
					0,								// Width Of Font
					0,								// Angle Of Escapement
					0,								// Orientation Angle
					FW_BOLD,						// Font Weight
					FALSE,							// Italic
					FALSE,							// Underline
					FALSE,							// Strikeout
					ANSI_CHARSET,					// Character Set Identifier
					OUT_TT_PRECIS,					// Output Precision
					CLIP_DEFAULT_PRECIS,			// Clipping Precision
					NONANTIALIASED_QUALITY,			// Output Quality
					FF_DONTCARE|DEFAULT_PITCH,		// Family And Pitch
					"MS Sans Serif");					// Font Name
SelectObject(GLOBhDC,font);							// Selects The Font We Want
wglUseFontBitmaps(GLOBhDC,32,96,base);				// Builds 96 Characters Starting At Character 32


running=true;
time1=timeGetTime();
tm.load("data/textures.dat");

vtx=new float[XFIELD*ZFIELD*4*3];
col=new unsigned char[XFIELD*ZFIELD*4*3];
tex=new float[XFIELD*ZFIELD*4*2];
ppos.x=100;
ppos.z=100;
ppos.y=10;
pyaw=1;

while(!quit)
	{
	// Any new message ?
	if (PeekMessage(&msg, NULL, 0, 0,PM_REMOVE))
		{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
		}
	 //well, if nothing happens we do as usual
	HandleIdle();
	}
glDeleteLists(1000,256);
return msg.wParam;
}

// Select the pixel format for a given device context
void SetDCPixelFormat(HDC hDC)
{
int nPixelFormat;

static PIXELFORMATDESCRIPTOR pfd;

pfd.nSize                 = sizeof(PIXELFORMATDESCRIPTOR);
pfd.nVersion              = 1;
pfd.dwFlags               = PFD_DRAW_TO_WINDOW | 
                            PFD_SUPPORT_OPENGL | 
                            PFD_DOUBLEBUFFER ;
pfd.iPixelType            = PFD_TYPE_RGBA;
pfd.cColorBits            = 32;
pfd.cRedBits              = 0;
pfd.cRedShift             = 0;
pfd.cGreenBits            = 0;
pfd.cGreenShift           = 0;
pfd.cBlueBits             = 0;
pfd.cBlueShift            = 0;
pfd.cAlphaBits            = 0;
pfd.cAlphaShift           = 0;
pfd.cAccumBits            = 0;   
pfd.cAccumRedBits         = 0;
pfd.cAccumGreenBits       = 0;
pfd.cAccumBlueBits        = 0;
pfd.cAccumAlphaBits       = 0;
pfd.cDepthBits            = 32;
pfd.cStencilBits          = 0;
pfd.cAuxBuffers           = 0;
pfd.iLayerType            = PFD_MAIN_PLANE;
pfd.bReserved             = 0;
pfd.dwLayerMask           = 0;
pfd.dwVisibleMask         = 0;
pfd.dwDamageMask          = 0;

// Choose a pixel format that best matches that described in pfd
nPixelFormat = ChoosePixelFormat(hDC, &pfd);
// Set the pixel format for the device context
SetPixelFormat(hDC, nPixelFormat, &pfd);
}



// If necessary, creates a 3-3-2 palette for the device context listed.
HPALETTE GetOpenGLPalette(HDC hDC)
{
HPALETTE hRetPal = NULL;	// Handle to palette to be created
PIXELFORMATDESCRIPTOR pfd;	// Pixel Format Descriptor
LOGPALETTE *pPal;			// Pointer to memory for logical palette
int nPixelFormat;			// Pixel format index
int nColors;				// Number of entries in palette
int i;						// Counting variable
BYTE RedRange,GreenRange,BlueRange;
							// Range for each color entry (7,7,and 3)

// Get the pixel format index and retrieve the pixel format description
nPixelFormat = GetPixelFormat(hDC);
DescribePixelFormat(hDC, nPixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd);

// Does this pixel format require a palette?  If not, do not create a
// palette and just return NULL
if(!(pfd.dwFlags & PFD_NEED_PALETTE))
	return NULL;
// Number of entries in palette.  8 bits yeilds 256 entries
nColors = 1 << pfd.cColorBits;	

// Allocate space for a logical palette structure plus all the palette entries
pPal = (LOGPALETTE*)malloc(sizeof(LOGPALETTE) +nColors*sizeof(PALETTEENTRY));
// Fill in palette header 
pPal->palVersion = 0x300;		// Windows 3.0
pPal->palNumEntries = nColors; // table size

// Build mask of all 1's.  This creates a number represented by having
// the low order x bits set, where x = pfd.cRedBits, pfd.cGreenBits, and
// pfd.cBlueBits.  
RedRange = (1 << pfd.cRedBits) -1;
GreenRange = (1 << pfd.cGreenBits) - 1;
BlueRange = (1 << pfd.cBlueBits) -1;

// Loop through all the palette entries
for(i = 0; i < nColors; i++)
	{
	// Fill in the 8-bit equivalents for each component
	pPal->palPalEntry[i].peRed = (i >> pfd.cRedShift) & RedRange;
	pPal->palPalEntry[i].peRed = (unsigned char)(
		(double) pPal->palPalEntry[i].peRed * 255.0 / RedRange);
		pPal->palPalEntry[i].peGreen = (i >> pfd.cGreenShift) & GreenRange;
	pPal->palPalEntry[i].peGreen = (unsigned char)(
		(double)pPal->palPalEntry[i].peGreen * 255.0 / GreenRange);

	pPal->palPalEntry[i].peBlue = (i >> pfd.cBlueShift) & BlueRange;
	pPal->palPalEntry[i].peBlue = (unsigned char)(
		(double)pPal->palPalEntry[i].peBlue * 255.0 / BlueRange);
		pPal->palPalEntry[i].peFlags = (unsigned char) NULL;
	}
		
// Create the palette
hRetPal = CreatePalette(pPal);

// Go ahead and select and realize the palette for this device context
SelectPalette(hDC,hRetPal,FALSE);
RealizePalette(hDC);

// Free the memory used for the logical palette structure
free(pPal);

// Return the handle to the new palette
return hRetPal;
}


// Viewport resizing
void HandleResize(int w,int h)
{
// Prevent a divide by zero
if(h==0) h=1;

// Set Viewport to window dimensions
glViewport(0,0,w,h);

// Calculate aspect ratio of the window
GLfloat fAspect = (GLfloat)w/(GLfloat)h;

// Set the perspective coordinate system
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
width=w;
height=h;
// Field of view of 45 degrees, near and far planes 1.0 and 425
gluPerspective(45.0f, fAspect, 1.0,50000.0);
// Modelview matrix reset
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
WINDOWX=w;
WINDOWY=h;

}



// Window procedure, handles all messages for the window
LRESULT CALLBACK WndProc(HWND hWnd,	UINT message, WPARAM wParam,
						 LPARAM lParam)
{
static HGLRC hRC = NULL;            // OpenGL Rendering context
static HPALETTE hPalette = NULL;	// 3-3-2 palette (used in 8-bit modes)

switch (message)
	{
	case WM_CREATE:
		{
		// Now, setup all the game stuff
		GLOBhDC = GetDC(hWnd);
		SetDCPixelFormat(GLOBhDC);
		hPalette = GetOpenGLPalette(GLOBhDC);
		// Create the rendering context and make it current
		hRC = wglCreateContext(GLOBhDC);
		wglMakeCurrent(GLOBhDC, hRC);
		// Hide the cursor
		ShowCursor(FALSE);
		// Setup rendering options first
		SetRenderingOptions();
		break;
		}
	case WM_SIZE:
		{
		HandleResize(LOWORD(lParam),HIWORD(lParam));
		break;
		}
	case WM_DESTROY:
		{
		// Deselect the current rendering context and delete it
		quit=1;
		wglMakeCurrent(GLOBhDC,NULL);
		wglDeleteContext(hRC);
		if(hPalette != NULL) DeleteObject(hPalette);
		ReleaseDC(hWnd,GLOBhDC);
		ChangeDisplaySettings(NULL,0);

		// Restores the display settings
		ShowCursor(TRUE);
		PostQuitMessage(0);
		break;
		}
	// Windows is telling the application that it may modify
	// the system palette.  This message in essance asks the
	// application for a new palette.
	case WM_QUERYNEWPALETTE:
		{
		// If the palette was created.
		if(hPalette)
			{
			int nRet;
			// Selects the palette into the current device context
			SelectPalette(GLOBhDC, hPalette, FALSE);
			// Map entries from the currently selected palette to
			// the system palette.  The return value is the number
			// of palette entries modified.
			nRet = RealizePalette(GLOBhDC);
			// Repaint, forces remap of palette in current window
			InvalidateRect(hWnd,NULL,FALSE);
			return nRet;
			}
		break;
		}

	// This window may set the palette, even though it is not the
	// currently active window.
	case WM_PALETTECHANGED:
		{
		// Don't do anything if the palette does not exist, or if
		// this is the window that changed the palette.
		if((hPalette != NULL) && ((HWND)wParam != hWnd))
			{
			// Select the palette into the device context
			SelectPalette(GLOBhDC,hPalette,FALSE);
			// Map entries to system palette
			RealizePalette(GLOBhDC);
			// Remap the current colors to the newly realized palette
			UpdateColors(GLOBhDC);
			return 0;
			}
		break;
		}
	default:          // Passes message on if unproccessed
		    return (DefWindowProc(hWnd, message, wParam, lParam));
	}
return (0L);
}


