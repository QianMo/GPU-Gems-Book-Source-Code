//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef KeyStateH
#define KeyStateH

#include <map>
//---------------------------------------------------------------------------
namespace Window {

class KeyState {
protected:
	static const int keyDelta;
	std::map<int,bool> keys;
	void checkGlutModifiers();

public:
	static const int convertGlutSpecialKey(const int key);
	static const int convertGlutKey(const char key);

	void glutKeyPress(const unsigned char key);
	void glutKeyUp(const unsigned char key);
	void glutSpecialKeyPress(const int key);
	void glutSpecialKeyUp(const int key);
	void glutMouseButton(const int button, const int state);

	void keyPress(const int key);
	void keyUp(const int key);

	bool isKeyDown(const int key);

	static const int KEY_TAB;
	static const int KEY_ESCAPE;
    static const int KEY_ENTER;
    static const int KEY_LEFT_ARROW;
    static const int KEY_RIGHT_ARROW;
    static const int KEY_UP_ARROW;
    static const int KEY_DOWN_ARROW;
    static const int KEY_HOME;
    static const int KEY_END;
    static const int KEY_PAGE_UP;
    static const int KEY_PAGE_DOWN;
    static const int KEY_INSERT;
    static const int KEY_DELETE;
    static const int KEY_SPACE;
    static const int KEY_F1;
    static const int KEY_F2;
    static const int KEY_F3;
    static const int KEY_F4;
    static const int KEY_F5;
    static const int KEY_F6;
    static const int KEY_F7;
    static const int KEY_F8;
    static const int KEY_F9;
    static const int KEY_F10;
    static const int KEY_F11;
    static const int KEY_F12;
	static const int KEY_MOUSE_LEFT;
	static const int KEY_MOUSE_MIDDLE;
	static const int KEY_MOUSE_RIGHT;
//	static const int KEY_MOUSE_X;
//	static const int KEY_MOUSE_Y;
	static const int KEY_SHIFT;
	static const int KEY_CTRL;
	static const int KEY_ALT;
};

//namespace
}
#endif
