// dx9window.cpp
#include "dx9window.hpp"

using namespace brook;

const char* DX9Window::kWindowClassName = "DX9Window_Class";

DX9Window::DX9Window()
{
	registerWindowClass();
	windowHandle = ::CreateWindow( kWindowClassName, "DX9 Test", WS_OVERLAPPEDWINDOW,
		100, 100, 512, 512, ::GetDesktopWindow(), NULL, GetModuleHandle(NULL), this );
}

DX9Window::~DX9Window()
{
	::DestroyWindow( windowHandle );
}

DX9Window* DX9Window::create()
{
  DX9PROFILE("DX9Window::create")
	DX9Window* result = new DX9Window();
	return result;
}

void DX9Window::show()
{
	::ShowWindow( windowHandle, SW_SHOW );
}

void DX9Window::hide()
{
	::ShowWindow( windowHandle, SW_HIDE );
}

HWND DX9Window::getWindowHandle()
{
	return windowHandle;
}

LRESULT DX9Window::handleMessage( UINT inMessage, WPARAM wParam, LPARAM lParam )
{
	switch( inMessage )
	{
	case WM_DESTROY:
		::PostQuitMessage( 0 );
		return 0;
	}

	return ::DefWindowProc( windowHandle, inMessage, wParam, lParam );
}

void DX9Window::finalize()
{
	if( windowHandle == NULL ) return;

	// TIM: TODO: handle destruction...
}

void DX9Window::registerWindowClass()
{
	static bool sInitialized = false;
	if( sInitialized ) return;
	sInitialized = true;

	WNDCLASSEX classDesc;
	ZeroMemory( &classDesc, sizeof(classDesc) );
	classDesc.cbSize = sizeof(classDesc);

	classDesc.style = CS_CLASSDC;
	classDesc.lpfnWndProc = (WNDPROC)(DX9Window::windowCallback);
	classDesc.hInstance = GetModuleHandle(NULL);
	classDesc.lpszClassName = kWindowClassName;

	RegisterClassEx( &classDesc );
}

LRESULT WINAPI DX9Window::windowCallback( HWND inWindowHandle, UINT inMessage, WPARAM wParam, LPARAM lParam )
{
	DX9Window* window = (DX9Window*)GetWindowLong( inWindowHandle, GWL_USERDATA );

	if( inMessage == WM_CREATE )
	{
		CREATESTRUCT* creationInfo = (CREATESTRUCT*)(lParam);
		window = (DX9Window*)creationInfo->lpCreateParams;
		SetWindowLong( inWindowHandle, GWL_USERDATA, (long)window );
	}

	if( window == NULL )
		return DefWindowProc( inWindowHandle, inMessage, wParam, lParam );

	LRESULT result = window->handleMessage( inMessage, wParam, lParam );

	if( inMessage == WM_DESTROY )
		window->finalize();

	return result;
}
