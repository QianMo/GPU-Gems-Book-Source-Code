/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)

#include <iostream>

#include "gpu.h"

#ifdef WINNT
#include <windowsx.h>
#endif


extern GpuContext GpuCreationContext;


void (*GpuOGL::set_root_context) (GpuContext ctx) = NULL;


struct GpuUserSelect {
    int fd;
    void (*callback)(void *user_data);
    void *user_data;
    bool readfd;
    GpuUserSelect *next;
    bool added;
};

GpuUserSelect *GpuSelectList = NULL;

static long GpuTimeoutSec = 10;
static long GpuTimeoutUSec = 0;
static void (*GpuTimeoutCallback) (void *user_data) = NULL;
static void *GpuTimeoutUserData = NULL;
extern std::vector<GpuWindow *> GpuWindowList; 

#define IDT_TIMER1 100


#ifdef XWINDOWS


//
// This XWINDOWS region contains functions and globals unique
// to the X/GLX implementation.
//



#define XK_LATIN1
#define XK_MISCELLANY
#include <X11/keysymdef.h>
#include <X11/cursorfont.h>

#include "hand.csr"
#include "zoomin.csr"
#include "zoomout.csr"
#include "wipe.csr"
#include "cross.csr"


// Global Display, supplied by canvas.cpp
extern Display *GpuDisplay;

// Code from the motif header files that control the decorations
// on standard windows.  It seems as if all window managers respect
// these settings, even if they are not motif.

typedef struct
{
    /* 32-bit property items are stored as long on the client (whether
     * that means 32 bits or 64).  XChangeProperty handles the conversion
     * to the actual 32-bit quantities sent to the server.
     */
    unsigned long	flags;
    unsigned long	functions;
    unsigned long	decorations;
    long 	        inputMode;
    unsigned long	status;
} PropMotifWmHints;

typedef PropMotifWmHints	PropMwmHints;
#define MWM_HINTS_DECORATIONS	(1L << 1)
#define _XA_MOTIF_WM_HINTS	"_MOTIF_WM_HINTS"
#define _XA_MWM_HINTS		_XA_MOTIF_WM_HINTS
#define PROP_MOTIF_WM_HINTS_ELEMENTS	5
#define PROP_MWM_HINTS_ELEMENTS		PROP_MOTIF_WM_HINTS_ELEMENTS



static Cursor GpuCursor[GpuWindow::LastCursor];



inline Bool
wait_for_notify(Display *display, XEvent *event, char *arg)
{
    return (event->type == MapNotify) && (event->xmap.window == (Window)arg);
}

// Create an X window and a GLX connection and make the window visible
inline Window
create_ogl_x_window (char *title,Window parent, int x, int y, int w, int h,
    GLXContext *glx_context, bool doublebuffered)
{
#ifndef DISABLE_GL    
    int n;
    int screen;
    Window window;
    int glx_attrib_list[16];
    XSetWindowAttributes win_attr;
    Colormap colormap;
    XVisualInfo *visual_info;
    XSizeHints sizehints;

    screen = DefaultScreen (GpuDisplay);

    n = 0;
    glx_attrib_list[n++] = GLX_RGBA;
    if (doublebuffered)
        glx_attrib_list[n++] = GLX_DOUBLEBUFFER;
    glx_attrib_list[n++] = GLX_RED_SIZE;
    glx_attrib_list[n++] = 1;
    glx_attrib_list[n++] = GLX_GREEN_SIZE;
    glx_attrib_list[n++] = 1;
    glx_attrib_list[n++] = GLX_BLUE_SIZE;
    glx_attrib_list[n++] = 1;
    glx_attrib_list[n++] = GLX_DEPTH_SIZE;
    glx_attrib_list[n++] = 1;
    glx_attrib_list[n++] = None;

    visual_info = glXChooseVisual (GpuDisplay, screen, glx_attrib_list);
    if (visual_info == NULL) {
        std::cerr << "Cannot choose GLX visual.\n";
        exit (EXIT_FAILURE);
    }

    extern GLXContext GpuCreationContext;
    *glx_context = glXCreateContext (GpuDisplay, visual_info,
                                     GpuCreationContext, True);
    if (*glx_context == NULL) {
        std::cerr << "Cannot create GLX context.\n";
        exit (EXIT_FAILURE);
    }
    if (GpuCreationContext == NULL) {
        GpuCreationContext = *glx_context;
        if (GpuOGL::set_root_context != NULL)
            GpuOGL::set_root_context (GpuCreationContext);
    }
    
    colormap = XCreateColormap (GpuDisplay,
        RootWindow (GpuDisplay, screen), visual_info->visual, AllocNone);

    win_attr.border_pixel = 0;
    win_attr.colormap = colormap;

    window = XCreateWindow (GpuDisplay, parent, x, y, w, h, 0,
        visual_info->depth, InputOutput, visual_info->visual,
        CWBorderPixel | CWColormap, &win_attr);

    sizehints.flags = USPosition | USSize | PMinSize | PMaxSize;
    sizehints.x = x;
    sizehints.y = y;
    sizehints.width = w;
    sizehints.height = h;
    sizehints.min_width = 10;
    sizehints.min_height = 10;
    GpuOGL::screen_size (sizehints.max_width, sizehints.max_height);
    
    XSetStandardProperties (GpuDisplay, window, title, title, None,
        NULL, 0, &sizehints);
    XSelectInput (GpuDisplay, window, StructureNotifyMask |
        ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
        ButtonPressMask | ButtonReleaseMask | PropertyChangeMask);

    XMapWindow (GpuDisplay, window);

    // Insure window is mapped and ready for OpenGL drawing
    XEvent event;
    XIfEvent (GpuDisplay, &event, wait_for_notify, (char *)window);

    return window;
#endif    
}

inline GpuEvent::Button
xbutton_to_button (unsigned int button)
{
    if (button == Button1) {
        return GpuEvent::LeftMouse;
    } else if (button == Button2) {
        return GpuEvent::MiddleMouse;
    } else if (button == Button3) {
        return GpuEvent::RightMouse;
    } else if (button == Button4) {
        return GpuEvent::WheelUpMouse;
    } else if (button == Button5) {
        return GpuEvent::WheelDownMouse;
    }

    return GpuEvent::NoMouse;
}    


inline GpuEvent::Button
state_to_button (unsigned int state)
{
    if (state & Button1Mask) {
        return GpuEvent::LeftMouse;
    } else if (state & Button2Mask) {
        return GpuEvent::MiddleMouse;
    } else if (state & Button3Mask) {
        return GpuEvent::RightMouse;
    }

    return GpuEvent::NoMouse;
}    



inline void
xstate_to_modifiers (unsigned int state, GpuEvent &event)
{
    event.alt = (state & Mod1Mask) != 0;        // not sure if this is right
    event.shift = (state & ShiftMask) != 0;
    event.control = (state & ControlMask) != 0;
}



GpuWindow *
gpu_find_by_xwindow (Window window)
{
    for (std::vector<GpuWindow *>::const_iterator i =
             GpuWindowList.begin(); i != GpuWindowList.end(); i++) {
        if ((*i)->window == window) {
            return *i;
        }
    }

    return NULL;
}


Cursor
create_cursor (int w, int h, unsigned char *bits, int xhot, int yhot)
{
    Window root = RootWindow (GpuDisplay, DefaultScreen (GpuDisplay));
    Pixmap source = XCreatePixmap (GpuDisplay, root, w, h, 1);
    Pixmap mask = XCreatePixmap (GpuDisplay, root, w, h, 1);

    static GC fgc = NULL, bgc = NULL;
    if (fgc == NULL) {
        XGCValues values;
        values.foreground = 0;
        fgc = XCreateGC(GpuDisplay, source, GCForeground, &values);
        values.foreground = 1;
        bgc = XCreateGC(GpuDisplay, source, GCForeground, &values);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char c = bits[y*w + x];
            switch (c) {
            case ' ':
                XDrawPoint (GpuDisplay, mask, fgc, x, y);
                XDrawPoint (GpuDisplay, source, fgc, x, y);
                break;
            case 'X':
                XDrawPoint (GpuDisplay, mask, bgc, x, y);
                XDrawPoint (GpuDisplay, source, bgc, x, y);
                break;
            case 'O':
                XDrawPoint (GpuDisplay, mask, bgc, x, y);
                XDrawPoint (GpuDisplay, source, fgc, x, y);
                break;
            default:
                abort();
            }
        }
    }
    
            
    XColor fg;
    fg.red = 0;
    fg.green = 0;
    fg.blue = 0;

    XColor bg;
    bg.red = 255*256;
    bg.green = 255*256;
    bg.blue = 255*256;

    return XCreatePixmapCursor(GpuDisplay, source,mask, &fg,&bg, xhot,yhot);
}


#endif // XWINDOWS

#ifdef WINNT

  

//
// This WINNT region contains functions and globals unique
// to the Win32/WGL implementation.
//



static HINSTANCE GpuhInstance = NULL;
static HWND GpuGlobalHWND = NULL;

// A special global message used to broadcast notify events between apps
static const UINT GpuBroadcastMsg = RegisterWindowMessage ("gpubroadcast");

// A special return value used to return from GpuWindowMainloop
static const LRESULT GpuBreakValue = 0xDEADBEEF;

static HCURSOR GpuCursor[GpuWindow::LastCursor];


void
win32_command (HWND hwnd, int id, HWND hwndctl, UINT codenotify)
{
    FORWARD_WM_COMMAND (hwnd, id, hwndctl, codenotify, DefWindowProc);
}



void
win32_destroy (HWND hwnd)
{
    PostQuitMessage (0);
}



LRESULT CALLBACK
win32_wndproc (HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
    PAINTSTRUCT ps;
    const int HIGH_BIT = 0x8000;  // high bit mask for a short
    GpuWindow *window = (GpuWindow *)GetWindowLong (hwnd, GWL_USERDATA);

    if (window != NULL) {

        if (window->break_out_of_mainloop) {
            window->break_out_of_mainloop = false;
            return GpuBreakValue;
        }

        GpuEvent event(*window, *(window->canvas), window->user_data);

        // Check for the special broadcast message and invoke callbacks
        if (message == GpuBroadcastMsg) {
            for (GpuUserSelect *cur=GpuSelectList; cur != NULL; /*EMPTY*/) {
                GpuUserSelect *next = cur->next;
                if (cur->readfd && cur->fd == -1)
                    cur->callback (&wparam);  // pass wparam instead of data!?
                cur = next;
            }
            return 0;
        }
        
        switch (message) {
        case WM_CREATE:
            return 0;

        case WM_CLOSE:
            event.type = GpuEvent::CloseWindow;
            window->event_handler (event);
            return 0;

        case WM_TIMER:
            if (wparam == IDT_TIMER1 && GpuTimeoutCallback != NULL)
                GpuTimeoutCallback (GpuTimeoutUserData);
            return 0;

        case WM_SOCKET_NOTIFY: {
            SOCKET hSocket = (SOCKET) wparam;
            bool reading = false;
            switch ((long) lparam) {
            case FD_READ: reading = true; break;
            case FD_WRITE:
            case FD_OOB:
            case FD_ACCEPT:
            case FD_CONNECT:
            case FD_CLOSE:
                break;
            }
            
            for (GpuUserSelect *cur=GpuSelectList; cur != NULL; /*EMPTY*/) {
                GpuUserSelect *next = cur->next;
                if (cur->readfd == reading && cur->fd == hSocket)
                    cur->callback (cur->user_data);
                cur = next;
            }
            break;
        }

        case WM_KEYDOWN:
            event.type = GpuEvent::KeyDown;
            event.key = (TCHAR)wparam;

            // High-order bit indicates key is pressed, low-order is toggle
            event.shift = GetKeyState (VK_SHIFT) & HIGH_BIT;
            event.control = GetKeyState (VK_CONTROL) & HIGH_BIT;
            event.alt = GetKeyState (VK_MENU) & HIGH_BIT;
            
            switch (wparam) {
            case 0x08: // backspace
            case 0x0A: // linefeed
            case 0x1B: // escape
            case 0x0D: // carriage retrun
            case VK_LEFT: // left arrow
            case VK_RIGHT: // right arrow
            case VK_DOWN: // down arrow
            case VK_UP: // up arrow
            case VK_END:
            case VK_INSERT:
            case VK_DELETE:
            case VK_F2:
                break;
            case VK_HOME:
                event.key = GpuKeyHome;
                break;
            case VK_PRIOR: 
                event.key = GpuKeyPageUp;
                break;
            case VK_NEXT: 
                event.key = GpuKeyPageDown;
                break;
            case 0xBB:
                event.key = '=';
                break;
            case 0xBD:
                event.key = '-';
                break;
            default:
                event.key = tolower (event.key);
            }
            if (window->event_handler)
                window->event_handler (event);
            return 0;

        case WM_LBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_RBUTTONDOWN:
            event.type = GpuEvent::MouseDown;
            switch (message) {
            case WM_LBUTTONDOWN : event.button = GpuEvent::LeftMouse; break;
            case WM_MBUTTONDOWN : event.button = GpuEvent::MiddleMouse; break;
            case WM_RBUTTONDOWN : event.button = GpuEvent::RightMouse; break;
            }
            if (wparam & MK_SHIFT) {
                event.shift = true;
            }
            if (wparam & MK_CONTROL) {
                event.control = true;
            }
            event.x = LOWORD (lparam);
            event.y = window->height - HIWORD (lparam);
            if (window->event_handler)
                window->event_handler (event);
            return 0;

        case WM_MOUSEMOVE:
            SetCursor (GpuCursor[window->curcursor]);
            event.type = GpuEvent::MouseDrag;
            if (wparam & MK_LBUTTON) {
                event.button = GpuEvent::LeftMouse;
            } else if (wparam & MK_MBUTTON) {
                event.button = GpuEvent::MiddleMouse;
            } else if (wparam & MK_RBUTTON) {
                event.button = GpuEvent::RightMouse;
            } else {
                event.button = GpuEvent::NoMouse;
            }
            if (wparam & MK_SHIFT) {
                event.shift = true;
            }
            if (wparam & MK_CONTROL) {
                event.control = true;
            }
            event.x = LOWORD (lparam);
            event.y = window->height - HIWORD (lparam);
            if (window->event_handler)
                window->event_handler (event);
            return 0;

        case WM_COMMAND:
            return HANDLE_WM_COMMAND (hwnd, wparam, lparam, win32_command);

        case WM_DESTROY:
            return HANDLE_WM_DESTROY (hwnd, wparam, lparam, win32_destroy);

        case WM_SETFOCUS:
            SetFocus (hwnd);
            return 0;

        case WM_SIZE:
            window->width = LOWORD (lparam);
            window->height = HIWORD (lparam);
            event.type = GpuEvent::Resize;
            event.w = window->width;
            event.h = window->height;
            if (window->event_handler)
                window->event_handler (event);
            return 0;

        case WM_MOVE:
            window->xorigin = LOWORD (lparam);
            window->yorigin = HIWORD (lparam);
            return 0;

        case WM_PAINT:
            window->mapped = true;
            BeginPaint (hwnd, &ps);
            window->repaint();
            EndPaint (hwnd, &ps);
            return 0;

        case WM_SETCURSOR:
            SetCursor (GpuCursor[window->curcursor]);
            return 0;
            
        default:
            break;
        }
    }

    return DefWindowProc (hwnd, message, wparam, lparam);
}



inline HWND
create_ogl_win32_window (HWND parent, int x, int y,int w, int h,
    HGLRC *hglrc, HDC *hdc, GpuWindow *window, bool doublebuffered)
{
    HWND hwnd;
    PIXELFORMATDESCRIPTOR pfd;
    int format;
    static int initialized = 0;


    if (!initialized) {
	WNDCLASS wc;
        initialized = 1;
        wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wc.lpfnWndProc = win32_wndproc;
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        wc.hInstance = GpuhInstance;
        wc.hIcon = LoadIcon (NULL, IDI_APPLICATION);
        wc.hCursor = GpuCursor[GpuWindow::ArrowCursor];
        wc.hbrBackground = (HBRUSH)GetStockObject (LTGRAY_BRUSH);
        wc.lpszMenuName = NULL;
        wc.lpszClassName = "gpuclass";
        ATOM a = RegisterClass (&wc);
        DASSERT (a);
    }

    // update the w&h to account for the title bar
    w += 2*GetSystemMetrics (SM_CXFRAME);
    h += 2*GetSystemMetrics (SM_CYFRAME) + GetSystemMetrics (SM_CYCAPTION);

    // Create the opengl window 
    hwnd = CreateWindow ("gpuclass", "OpenGL Win32",
                         WS_OVERLAPPEDWINDOW, x, y, w, h, parent, NULL,
                         GpuhInstance, NULL );

    if (hwnd == NULL) {
        DWORD err = GetLastError();
        fprintf (stderr, "Cannot create WIN32 Window %d\n", err);
        exit (EXIT_FAILURE);
    }

    SetWindowLong (hwnd, GWL_USERDATA, (long)window);
    
    *hdc = GetDC (hwnd);

    // set to static GpuCreationDrawable and used for select events
    if (GpuGlobalHWND == NULL)
        GpuGlobalHWND = hwnd;
    
    // set the pixel format for the DC
    ZeroMemory (&pfd, sizeof (pfd));
    pfd.nSize = sizeof (pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; 
    if (doublebuffered)
        pfd.dwFlags |= PFD_DOUBLEBUFFER;
//    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 16;
    pfd.iLayerType = PFD_MAIN_PLANE;
    format = ChoosePixelFormat (*hdc, &pfd);
    SetPixelFormat (*hdc, format, &pfd);
 
    // create and enable the render context (RC)
    *hglrc = wglCreateContext (*hdc);
    if (GpuCreationContext == NULL) {
        GpuCreationContext = *hglrc;
        if (GpuOGL::set_root_context != NULL)
            GpuOGL::set_root_context (GpuCreationContext);
    } else if (!wglShareLists (GpuCreationContext, *hglrc)) {
        fprintf (stderr, "Cannot share OpenGL contexts\n");
        fflush (stderr);
    }
        
    return hwnd;
}

#endif  // WINNT


//
// Shared functions
//

inline void
create_user_select (int fd, void (*callback)(void *user_data),
    void *user_data, bool readfd)
{
    GpuUserSelect *s = new GpuUserSelect;
    s->fd = fd;
    s->callback = callback;
    s->user_data = user_data;
    s->readfd = readfd;
    s->added = false;
    s->next = GpuSelectList;
    GpuSelectList = s;
}



void
GpuAddReadSelect (int fd, void (*callback)(void *user_data),
    void *user_data)
{
    create_user_select (fd, callback, user_data, true);
#ifdef WINNT
    // send a message to activate the new socket
    PostMessage (NULL, WM_USER, NULL, NULL);
#endif    
}



void
GpuAddWriteSelect (int fd, void (*callback)(void *user_data),
    void *user_data)
{
    create_user_select (fd, callback, user_data, false);
#ifdef WINNT
    // send a message to activate the new socket
    PostMessage (NULL, WM_USER, NULL, NULL);
#endif    
}



void
GpuRemoveSelect (int fd)
{
    GpuUserSelect *cur = GpuSelectList;
    GpuUserSelect *prev = NULL;
    while (cur != NULL) {
        GpuUserSelect *next = cur->next;
        if (cur->fd == fd) {
            if (cur == GpuSelectList)
                GpuSelectList = cur->next;
            if (prev != NULL)
                prev->next = cur->next;
            delete cur;
        }
        prev = cur;
        cur = next;
    }
#ifdef WINNT
    // send a message to de-activate the new socket
    PostMessage (NULL, WM_USER, NULL, NULL);
#endif    
}




GpuWindow::GpuWindow (int x, int y, unsigned int w, unsigned int h,
                      bool doublebuffered) 
    : GpuDrawable(w, h), xorigin(x), yorigin(y),
      event_handler(NULL), user_data(NULL),
      break_out_of_mainloop(false),
      window_title(0), blackout_on(false), 
      mapped (false), curcursor (ArrowCursor)
{
    assert (w > 0);
    assert (h > 0);

    // store on global list so we can find in mainloop()
    GpuWindowList.push_back (this);
    
#ifdef LINUX
#ifndef DISABLE_GL    
    open_display();

    Window parent = DefaultRootWindow (GpuDisplay);
    window = create_ogl_x_window ("OGL", parent, xorigin, yorigin, w,h,
        &glx_context, doublebuffered);

    // Intercept window manager quit events
    quit_atom = XInternAtom (GpuDisplay, "WM_DELETE_WINDOW", false);
    XSetWMProtocols (GpuDisplay, window, &quit_atom, 1);
#endif    
#endif

#ifdef WINNT
    GpuhInstance = GetModuleHandle (NULL);
    hwnd = create_ogl_win32_window (NULL, xorigin, yorigin, w, h,
        &hglrc, &hdc, this, doublebuffered);
#endif
    
    static bool initialized_cursors = false;
    if (!initialized_cursors) {
        initialized_cursors = true;

#ifdef LINUX
        GpuCursor[ArrowCursor] = XCreateFontCursor (GpuDisplay,
            XC_top_left_arrow);
        GpuCursor[WatchCursor] = XCreateFontCursor (GpuDisplay,XC_watch);
        GpuCursor[HandCursor] = create_cursor (hand_width, hand_height,
            hand_data, hand_hotx, hand_hoty);
        GpuCursor[WipeCursor] = create_cursor (wipe_width, wipe_height,
            wipe_data, wipe_hotx, wipe_hoty);
        GpuCursor[ZoomInCursor] = create_cursor (zoomin_width,zoomin_height,
            zoomin_data, zoomin_hotx, zoomin_hoty);
        GpuCursor[ZoomOutCursor] = create_cursor (zoomout_width,
            zoomout_height, zoomout_data, zoomout_hotx, zoomout_hoty);
        GpuCursor[CrossCursor] = create_cursor (cross_width, cross_height,
            cross_data, cross_hotx, cross_hoty);
#else
        GpuCursor[ArrowCursor] = LoadCursor (NULL, IDC_ARROW);
        GpuCursor[WatchCursor] = LoadCursor (NULL, IDC_WAIT);
        GpuCursor[HandCursor] = LoadCursor (NULL, IDC_HAND);
        GpuCursor[WipeCursor] = LoadCursor (NULL, IDC_IBEAM);
        // FIXME: Convert chars to bitmasks for magnifying glass cursors!
        GpuCursor[ZoomInCursor] = GpuCursor[ArrowCursor];
        GpuCursor[ZoomOutCursor] = GpuCursor[ArrowCursor];
        GpuCursor[CrossCursor] = LoadCursor (NULL, IDC_CROSS);
#endif        
    }
    
}



bool
remove_from_windowlist (GpuWindow *window)
{
    // Remove window on destruction so we drop out of
    // GpuWindowMainloop after all windows are destroyed.
    for (std::vector<GpuWindow *>::iterator i =
             GpuWindowList.begin(); i != GpuWindowList.end(); i++) {
        if (*i == window) {
            GpuWindowList.erase (i);
            return true;
        }
    }
    return false;
}

    
GpuWindow::~GpuWindow ()
{
    remove_from_windowlist (this);
}



void
GpuWindow::timeout (long sec, long usec,
    void (*callback)(void *user_data), void *user_data)
{
#ifdef WINNT
    if (GpuTimeoutCallback != NULL)
        KillTimer (hwnd, IDT_TIMER1);
    if (callback != NULL)
        SetTimer (hwnd, IDT_TIMER1, sec * 1000 + usec/1000, NULL);
#endif
    GpuTimeoutSec = sec;
    GpuTimeoutUSec = usec;
    GpuTimeoutCallback = callback;
    GpuTimeoutUserData = user_data;
}


#ifdef XWINDOWS
#ifndef DISABLE_GL    
int
gpu_window_handle_event (XEvent& event)
{
    // find the window that contains this X Window structure
    GpuWindow *window = gpu_find_by_xwindow(event.xany.window);
    assert (window != NULL);

    if (window->break_out_of_mainloop) {
        // Reset in case main loop is re-entered later
        window->break_out_of_mainloop = false;
        return -1;
    }
    if (window->event_handler == NULL) return 0;

    // Note: re-initialized each time to insure user_data is valid
    GpuEvent ve(*window, *window->canvas, window->user_data);

    char buffer[128];
    KeySym keysym;
    XComposeStatus status;
    
    switch (event.type) {
    case Expose:
        while (XCheckWindowEvent (GpuDisplay, event.xany.window,
                   ExposureMask, &event));
        window->mapped = true;
        window->repaint();
        break;
    case ConfigureNotify:
        while (XCheckWindowEvent (GpuDisplay, event.xany.window,
                   StructureNotifyMask, &event));
        window->width = event.xconfigure.width;
        window->height = event.xconfigure.height;
        window->xorigin = event.xconfigure.x;
        window->yorigin = event.xconfigure.y;
        window->mapped = true;
        ve.type = GpuEvent::Resize;
        ve.w = window->width;
        ve.h = window->height;
        window->event_handler (ve);
        window->repaint();
        break;
    case PropertyNotify:
        // Note: If we don't do this, iv misses the initial repaint
        while (XCheckWindowEvent (GpuDisplay, event.xany.window,
                   PropertyChangeMask, &event));
        break;
    case KeyPress:
        ve.type = GpuEvent::KeyDown;
        XLookupString (&event.xkey, buffer, 128, &keysym, &status);
        ve.key = ((int) keysym - XK_0 + '0');
        ve.x = event.xkey.x;
        ve.y = event.xkey.y;
        xstate_to_modifiers (event.xkey.state, ve);
        window->event_handler (ve);
        break;
    case KeyRelease:
        ve.type = GpuEvent::KeyUp;
        XLookupString (&event.xkey, buffer, 128, &keysym, &status);
        ve.key = ((int) keysym - XK_0 + '0');
        xstate_to_modifiers (event.xkey.state, ve);
        window->event_handler (ve);
        break;
    case ButtonPress:
    case ButtonRelease:
        ve.type = event.type == ButtonPress ? GpuEvent::MouseDown :
            GpuEvent::MouseUp;
        ve.x = event.xbutton.x;
        ve.y = window->h() - event.xbutton.y;
        xstate_to_modifiers (event.xbutton.state, ve);
        ve.button = xbutton_to_button (event.xbutton.button);
        window->event_handler (ve);
        break;
    case MotionNotify:
        while (XCheckWindowEvent (GpuDisplay, event.xany.window,
                   PointerMotionMask, &event));
        ve.type = GpuEvent::MouseDrag;
        ve.x = event.xmotion.x;
        ve.y = window->h() - event.xmotion.y;
        xstate_to_modifiers (event.xmotion.state, ve);
        ve.button = state_to_button (event.xmotion.state);
        window->event_handler (ve);
        break;
    case ClientMessage:
        DASSERT (event.xclient.data.l[0] == (int)window->quit_atom);
        ve.type = GpuEvent::CloseWindow;
        window->event_handler (ve);
        break;
    }

    return 0;
}

#endif
#endif



// Called after all windows have been created?? to begin the
// event-based display of the windows.
void
GpuWindowMainloop (bool block)
{
#ifndef DISABLE_GL    
#ifdef XWINDOWS    
    XEvent event;

    // Block until all windows and sockets are deleted
    while (!GpuWindowList.empty() || GpuSelectList) {
        // block in select
	struct timeval timeout;
        if (block) {
            timeout.tv_sec = GpuTimeoutSec;
            timeout.tv_usec = GpuTimeoutUSec;
        } else {
            timeout.tv_sec = 0;
            timeout.tv_usec = 0;
        }
	fd_set read_fds, write_fds;
        FD_ZERO(&read_fds);
        FD_ZERO(&write_fds);
        XFlush(GpuDisplay);
        FD_SET(ConnectionNumber(GpuDisplay), &read_fds);

        GpuUserSelect *cur = GpuSelectList;
        while (cur != NULL) {
            if (cur->readfd) 
                FD_SET(cur->fd, &read_fds);
            else 
                FD_SET(cur->fd, &write_fds);
            cur = cur->next;
        }

        int count = select (FD_SETSIZE, &read_fds, &write_fds, NULL, &timeout);
        if (count >= 0) {
            bool event_handled = false;
            if (FD_ISSET (ConnectionNumber (GpuDisplay), &read_fds) ||
                FD_ISSET (ConnectionNumber (GpuDisplay), &write_fds)) {
                for (int pending = XPending (GpuDisplay);
                     pending > 0 && XPending (GpuDisplay); pending--) {
                    event_handled = true;
                    XNextEvent (GpuDisplay, &event);
                    if (gpu_window_handle_event (event) < 0)
                        return;
                }
            }

            cur = GpuSelectList;
            while (cur != NULL) {
                GpuUserSelect *next = cur->next;
                if (FD_ISSET (cur->fd, &read_fds) ||
                    FD_ISSET (cur->fd, &write_fds)) {
                    event_handled = true;
                    cur->callback (cur->user_data);
                }
                cur = next;
            }

            if (!event_handled) {
                if (!block)
                    break;
                else if (GpuTimeoutCallback != NULL)
                    GpuTimeoutCallback (GpuTimeoutUserData);
            }
        }
    }
#endif
    
#ifdef WINNT    
    BOOL status;
    MSG msg, ui_msg;

    // Block until all windows and sockets are deleted
    // Note: Windows has a default window created as a static variable!
    while (!GpuWindowList.empty() || GpuSelectList) {
        // Add any new select file descriptors to the global creation window
        for (GpuUserSelect *cur=GpuSelectList; cur != NULL; cur=cur->next){
            if (cur->added)
                continue;
            cur->added = true;
            if (cur->readfd) {
                if (cur->fd >= 0 &&
                    SOCKET_ERROR == WSAAsyncSelect (cur->fd, GpuGlobalHWND, 
                                                    WM_SOCKET_NOTIFY,
                                                    FD_READ)) {
                    std::cerr << "Cannot add read select on fd "
                              << cur->fd << "\n";
                }
            } else {
                if (SOCKET_ERROR == WSAAsyncSelect (cur->fd, GpuGlobalHWND, 
                                                    WM_SOCKET_NOTIFY,
                                                    FD_WRITE)) {
                    std::cerr << "Cannot add write select on fd "
                              << cur->fd << "\n";
                }
            }
        }
        
        if (block) 
            status = GetMessage (&msg, NULL, 0, 0);
        else
            status = PeekMessage (&msg, NULL, 0, 0, PM_REMOVE);

        // If GetMessage got WM_QUIT, or no message for PeekMessage, return
        if (status == 0) break;
            
        // process mouse and keyboard events at a higher priority
        // than timer or socket notification to allow ui interaction
        // even if we get lots of timer and socket events
        if (PeekMessage(&ui_msg, 0, WM_MOUSEFIRST, WM_MOUSELAST, PM_REMOVE)) {
            TranslateMessage (&ui_msg);
            if (DispatchMessage (&ui_msg) == GpuBreakValue)
                break;
        }
        if (PeekMessage (&ui_msg, 0, WM_KEYFIRST, WM_KEYLAST, PM_REMOVE)) {
            TranslateMessage (&ui_msg);
            if (DispatchMessage (&ui_msg) == GpuBreakValue)
                break;
        }

        // process the low-priority event
        TranslateMessage (&msg);
        if (DispatchMessage (&msg) == GpuBreakValue)
            break;
    }
#endif
#endif
}



void
GpuWindow::set_event_handler (GpuEventHandler handler,
    void *user_data)
{
    event_handler = handler;
    this->user_data = user_data;
}



void
GpuWindow::configure (int x, int y, int w, int h)
{
    width = w;
    height = h;
    xorigin = x;
    yorigin = y;

#ifndef DISABLE_GL    
#ifdef XWINDOWS
    XMoveResizeWindow (GpuDisplay, window, x, y, w, h);
#endif

#ifdef WINNT
    // update the w&h to account for the title bar
    w += 2 * GetSystemMetrics (SM_CXFRAME);
    h += 2 * GetSystemMetrics (SM_CYFRAME) + GetSystemMetrics (SM_CYCAPTION);
    SetWindowPos (hwnd, HWND_TOP, x, y, w, h, 0);
#endif
#endif
}




void
GpuWindow::title (char *title, char *icontitle)
{
    if (window_title != NULL)
        free (window_title);
    window_title = strdup (title);
    
#ifndef DISABLE_GL    
#ifdef XWINDOWS
    XStoreName (GpuDisplay, window, title);
    char *istr = icontitle == NULL ? title : icontitle;
    XSetIconName (GpuDisplay, window, istr);
#endif

#ifdef WINNT
    SetWindowText (hwnd, title);
#endif
#endif    
}



void
GpuWindow::repaint()
{
    if (!mapped)
        return;
    if (event_handler == NULL) 
        return;
    
    GpuEvent event(*this, *canvas, user_data);
    event.type = GpuEvent::Redraw;
    event_handler (event);
}



void
GpuWindow::post_repaint()
{
    // FIXME/WINDOWS: Must add ability to post a repaint event for windows
#ifdef XWINDOWS
    XEvent event;
    XExposeEvent &expose = event.xexpose;
    expose.type = Expose;
    expose.window = window;
    expose.display = GpuDisplay;
    expose.x = xorigin;
    expose.y = yorigin;
    expose.width = width;
    expose.height = height;

    XSendEvent (GpuDisplay, window, False, ExposureMask, &event);
    // If another thread is in select() already (eg AlgoVis),
    // it won't get the event for a second or two unless we flush.
    XFlush (GpuDisplay);
#endif
}



void
GpuWindow::show (bool status)
{
#ifndef DISABLE_GL    
#ifdef XWINDOWS
    if (window != 0)
        if (status)
            XMapWindow(GpuDisplay, window);
        else
            XUnmapWindow(GpuDisplay, window);
#endif

#ifdef WINNT
    if (hwnd != NULL) {
        ShowWindow(hwnd, status ? SW_SHOW : SW_HIDE);
        UpdateWindow(hwnd);
    }
#endif
#endif    
}


void
GpuWindow::resizeable (bool status)
{
#ifndef WINNT
    // FIXME/WINNT: Must add ability to disable and enable window resizing
    XSizeHints sizehints;

    if (status) {
        sizehints.min_width = 0;
        sizehints.min_height = 0;
        GpuOGL::screen_size (sizehints.max_width, sizehints.max_height);
    } else {
        sizehints.min_width = w();
        sizehints.min_height = h();
        sizehints.max_width = w();
        sizehints.max_height = h();
    }

    sizehints.flags = PMinSize | PMaxSize;
    char *title;
    if (window_title == NULL)
        title = "OGL";
    else
        title = window_title;
    
    XSetStandardProperties (GpuDisplay, window, title, title, None,
        NULL, 0, &sizehints);
#endif
}



void
GpuWindow::raise ()
{
#ifdef XWINDOWS
    XRaiseWindow (GpuDisplay, window);
#endif
#ifdef WINNT
    // FIXME: Windows XP does not allow you to actually set the foreground
    // image, but this will cause the icon to flash in the taskbar.
    //
    // From the Windows manual pages:
    //
    //   Windows 2000/XP: A process that can set the foreground window
    //   can enable another process to set the foreground window by
    //   calling the AllowSetForegroundWindow function, or by calling
    //   the BroadcastSystemMessage function with the BSF_ALLOWSFW
    //   flag. The foreground process can disable calls to
    //   SetForegroundWindow by calling the LockSetForegroundWindow
    //   function.

    SetForegroundWindow (hwnd);
    BringWindowToTop (hwnd);
    UpdateWindow (hwnd);
#endif    
}



void
GpuOGL::screen_size (int &width, int &height)
{
#ifdef XWINDOWS
    Window rootwindow = RootWindow(GpuDisplay,DefaultScreen(GpuDisplay));
    XWindowAttributes attr;
    XGetWindowAttributes (GpuDisplay, rootwindow, &attr);
    width = attr.width;
    height = attr.height;
#endif
#ifdef WINNT
    DEVMODE dmSettings;
    dmSettings.dmSize = sizeof(DEVMODE);
    // get the settings from the current display mode, incl. taskbar
    if (!EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dmSettings)) {
        // call failed, use another method to get active user area
        width = GetSystemMetrics (SM_CXFULLSCREEN);
        height = GetSystemMetrics (SM_CYFULLSCREEN);
    } else {
        width = dmSettings.dmPelsWidth;
        height = dmSettings.dmPelsHeight;
    }
#endif
}



GpuEvent::GpuEvent (GpuWindow& window, GpuCanvas &canvas,
    void *user_data) : canvas(canvas), window(window), type(NoEvent),
     key(0), x(0), y(0), w(0), h(0),
     button(NoMouse), alt(false), shift(false), control(false),
     user_data(user_data)
{}


#ifdef XWINDOWS

// WARNING: Black Magic Ahead
//
// Getting blackout to work under KDE is a bit tricky.  You need to
// unmap the window before setting the decoration hints to get rid of
// the window frame.  Then you need to remap the window afterwards.
//
// I'm betting that this still has problem with older Gnome
// distributions, say around RH 7.2, which use a layered window
// manager.  For that to work, I think we need to forcibly remove the
// Gnome panel by searching for it in the entire window list and then
// hiding it.
//
// Another brittle piece of code has to do with the order of configure
// and show operations.  If they get switched, it doesn't work.
//
// FIXME: There is a bug under Gnome which looses the mouse position
//        if the window is moved from its starting point and then
//        switched into blackout mode and back twice.

static int
window_mapped_state (Window window)
{
    Atom actual_type_return;
    int actual_format_return;
    unsigned long nitems_return, bytes_after_return;
    unsigned char* prop_return;
  
    Atom atom = XInternAtom(GpuDisplay, "WM_STATE", False);

    XGetWindowProperty(GpuDisplay, window, atom, 0, 1, False, atom,
        &actual_type_return, &actual_format_return, &nitems_return,
        &bytes_after_return, &prop_return);
    
    if (actual_type_return == atom) {
        int WM_STATE = (int)(((unsigned long *)prop_return)[0]);
        XFree (prop_return);
        return WM_STATE;
    }

    return -1;
}  



static Bool
unmap_window_predicate (Display *dpy, XEvent *event, XPointer arg)
{
    return event->type == UnmapNotify || event->type == ReparentNotify ||
        event->type == PropertyNotify;
}



static void
unmap_window (Window window)
{
    bool wait_for_wm_state = true;
    bool wait_for_reparent = true;
    bool wait_for_unmap = true;

    int wm_state = window_mapped_state (window);
    if (wm_state <= 0)
        wait_for_wm_state = 0;

    Window root_return, parent_return;
    Window *children_return = NULL;
    unsigned int nchildren_return;

    if (XQueryTree (GpuDisplay, window, &root_return, &parent_return,
           &children_return, &nchildren_return)) {
        if (root_return == parent_return)
            wait_for_reparent = 0;
        if(children_return)
            XFree (children_return);
    } else {
        root_return = DefaultRootWindow (GpuDisplay);
    }
    
    XWithdrawWindow(GpuDisplay, window, DefaultScreen(GpuDisplay));

    // Loop until the window is actually unmapped
    Atom atom = XInternAtom(GpuDisplay, "WM_STATE", False);
    for (int n = 0; wait_for_unmap ||
            (n < 50 && (wait_for_reparent || wait_for_wm_state)); n++) {

        XEvent event;
        while (XCheckIfEvent(GpuDisplay, &event, unmap_window_predicate,NULL)){
            switch (event.type) {
            case UnmapNotify:
                wait_for_unmap = 0;
                break;
            case ReparentNotify:
                if (event.xreparent.parent == root_return)
                    wait_for_reparent = 0;
                break;
            case PropertyNotify:
                if (event.xproperty.atom == atom) {
                    wm_state = window_mapped_state (window);
                    if (wm_state <= 0) {
                        wait_for_wm_state = 0;
                        wait_for_reparent = 0;
                    }
                }
            default:
                break;
            }
        }
        usleep(10000);
    }
}

#endif    


void
GpuWindow::blackout (bool status)
{
    // FIXME/WINNT: Blackout not implemented for Windows
    if (status) {
        if (blackout_on)
            return;
        blackout_on = true;

        // save current position
        blackout_x = x();
        blackout_y = y();
        blackout_w = w();
        blackout_h = h();

#ifdef XWINDOWS
        // Note: Motif decorations can only be changed on unmapped windows    
        unmap_window (window);

        Atom WM_HINTS = XInternAtom(GpuDisplay, "_MOTIF_WM_HINTS", True);
        if(WM_HINTS != None) {
            PropMwmHints MWMHints = { MWM_HINTS_DECORATIONS, 0, 0, 0, 0 };
            XChangeProperty(GpuDisplay, window, WM_HINTS, WM_HINTS, 32,
                PropModeReplace, (unsigned char *)&MWMHints,
                sizeof (MWMHints) / sizeof(long));
        }
#endif
        
        int width, height;
        GpuOGL::screen_size (width, height);

#ifdef WINNT
        // Change the display over to fullscreen with current settings
        DEVMODE dmSettings;
        dmSettings.dmSize = sizeof(DEVMODE);
        if (EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dmSettings)) {
            // must set these fields even though they are not changing!
            dmSettings.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL;
            ChangeDisplaySettings(&dmSettings, CDS_FULLSCREEN);
            // Fullscreen only works with popup windows.  Parts of the
            // docs clearly say that you can't change the window style
            // of a window after it was created, but the SetWindowPos
            // function doc describes the required magic incantation
            SetWindowLong (hwnd, GWL_STYLE, WS_POPUP);
            SetWindowPos (hwnd, HWND_TOP, 0, 0, width, height, 
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
#endif
        configure (0, 0, width, height);
        show (true);
    } else {
        if (!blackout_on)
            return;
        blackout_on = false;
        
#ifdef XWINDOWS
        unmap_window (window);
        Atom WM_HINTS = XInternAtom (GpuDisplay, "_MOTIF_WM_HINTS", True);
        if (WM_HINTS != None) 
            XDeleteProperty(GpuDisplay, window, WM_HINTS);

        PropMwmHints mwm_hints;
        mwm_hints.flags = MWM_HINTS_DECORATIONS;
        mwm_hints.decorations = 1;
        Atom atom;
        atom = XInternAtom(GpuDisplay, _XA_MOTIF_WM_HINTS, False);
        
        XChangeProperty(GpuDisplay, window, atom, atom, 32, PropModeReplace,
            (unsigned char *) &mwm_hints, PROP_MOTIF_WM_HINTS_ELEMENTS);
        
#endif
#ifdef WINNT
        // Switch back to a normal window with magic incantation
        SetWindowLong (hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);
        SetWindowPos (hwnd, HWND_TOP, 0, 0, width, height, 
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        // Switch back to normal, non-fullscreen, mode
        ChangeDisplaySettings (NULL, 0);
#endif
        configure (blackout_x, blackout_y, blackout_w, blackout_h);
        show (true);
    }
}



GpuWindow::CursorType
GpuWindow::cursor (CursorType cursor)
{
    CursorType old = curcursor;
    curcursor = cursor;
#ifdef XWINDOWS
    if (cursor == ArrowCursor)
        XUndefineCursor (GpuDisplay, window);
    else 
        XDefineCursor (GpuDisplay, window, GpuCursor[cursor]);
#else
    SetCursor (GpuCursor[cursor]);
#endif    
    return old;
}



#ifdef WINNT
bool
GpuWindow::broadcast_message (WPARAM wparam, LPARAM lparam)
{
    return SendNotifyMessage((HWND) HWND_BROADCAST, // broadcast to all windows
                             GpuBroadcastMsg,       // initiates conversation 
                             wparam, lparam);       // message parameters
}



bool
GpuWindow::is_broadcast_message (MSG msg)
{
    return msg.message == GpuBroadcastMsg;
}

#endif



void
GpuOGL::bell (int volume)
{
    // FIXME/WINNT: Either remove this for XWindows or add bell for Windows
#ifdef XWINDOWS
    if (GpuDisplay == None)
        return;
    XBell (GpuDisplay, volume);
#endif    
}
