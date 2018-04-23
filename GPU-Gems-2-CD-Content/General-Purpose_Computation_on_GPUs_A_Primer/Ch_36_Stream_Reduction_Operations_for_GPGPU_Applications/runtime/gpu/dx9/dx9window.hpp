// dx9window.hpp
#pragma once

#include "dx9base.hpp"

namespace brook {

  class DX9Window
  {
  public:
	  static DX9Window* create();
	  ~DX9Window();

	  void show();
	  void hide();

	  HWND getWindowHandle();

  private:
	  DX9Window();
	  LRESULT handleMessage( UINT inMessage, WPARAM wParam, LPARAM lParam );
	  void finalize();

	  HWND windowHandle;

	  static const char* kWindowClassName;
	  static void registerWindowClass();
	  static LRESULT WINAPI windowCallback( HWND inWindowHandle, UINT inMessage, WPARAM wParam, LPARAM lParam );
  };
}