//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#include "KeyState.h"
#include <GL/glHeader.h>
//---------------------------------------------------------------------------
//#pragma package(smart_init)
namespace Window {

const int KeyState::keyDelta = 256;
const int KeyState::KEY_TAB = '\t';
const int KeyState::KEY_ESCAPE = 0x1B;
const int KeyState::KEY_ENTER = 0x0D;
const int KeyState::KEY_LEFT_ARROW = GLUT_KEY_LEFT+keyDelta;
const int KeyState::KEY_RIGHT_ARROW = GLUT_KEY_RIGHT+keyDelta;
const int KeyState::KEY_UP_ARROW = GLUT_KEY_UP+keyDelta;
const int KeyState::KEY_DOWN_ARROW = GLUT_KEY_DOWN+keyDelta;
const int KeyState::KEY_HOME = GLUT_KEY_HOME+keyDelta;
const int KeyState::KEY_END = GLUT_KEY_END+keyDelta;
const int KeyState::KEY_PAGE_UP = GLUT_KEY_PAGE_UP+keyDelta;
const int KeyState::KEY_PAGE_DOWN = GLUT_KEY_PAGE_DOWN+keyDelta;
const int KeyState::KEY_INSERT = GLUT_KEY_INSERT+keyDelta;
const int KeyState::KEY_DELETE = 0x2E;
const int KeyState::KEY_SPACE = ' ';
const int KeyState::KEY_F1 = GLUT_KEY_F1+keyDelta;
const int KeyState::KEY_F2 = GLUT_KEY_F2+keyDelta;
const int KeyState::KEY_F3 = GLUT_KEY_F3+keyDelta;
const int KeyState::KEY_F4 = GLUT_KEY_F4+keyDelta;
const int KeyState::KEY_F5 = GLUT_KEY_F5+keyDelta;
const int KeyState::KEY_F6 = GLUT_KEY_F6+keyDelta;
const int KeyState::KEY_F7 = GLUT_KEY_F7+keyDelta;
const int KeyState::KEY_F8 = GLUT_KEY_F8+keyDelta;
const int KeyState::KEY_F9 = GLUT_KEY_F9+keyDelta;
const int KeyState::KEY_F10 = GLUT_KEY_F10+keyDelta;
const int KeyState::KEY_F11 = GLUT_KEY_F11+keyDelta;
const int KeyState::KEY_F12 = GLUT_KEY_F12+keyDelta;
const int KeyState::KEY_MOUSE_LEFT = GLUT_LEFT_BUTTON+2*keyDelta;
const int KeyState::KEY_MOUSE_MIDDLE = GLUT_MIDDLE_BUTTON+2*keyDelta;
const int KeyState::KEY_MOUSE_RIGHT = GLUT_RIGHT_BUTTON+2*keyDelta;
//const int KeyState::KEY_MOUSE_X = 3+2*keyDelta;
//const int KeyState::KEY_MOUSE_Y = 4+2*keyDelta;
const int KeyState::KEY_SHIFT = GLUT_ACTIVE_SHIFT+3*keyDelta;
const int KeyState::KEY_CTRL = GLUT_ACTIVE_CTRL+3*keyDelta;
const int KeyState::KEY_ALT = GLUT_ACTIVE_ALT+3*keyDelta;

bool KeyState::isKeyDown(const int key) {
	return keys[key];
}

void KeyState::checkGlutModifiers() {
	int state = glutGetModifiers();
	keys[KEY_SHIFT] = (state & GLUT_ACTIVE_SHIFT) != 0;
	keys[KEY_CTRL] = (state & GLUT_ACTIVE_CTRL) != 0;
	keys[KEY_ALT] = (state & GLUT_ACTIVE_ALT) != 0;
}

void KeyState::keyPress(const int key) {
	keys[key] = true;
}

void KeyState::keyUp(const int key) {
	keys[key] = false;
}

const int KeyState::convertGlutSpecialKey(const int key) {
	return key+keyDelta;
}

const int KeyState::convertGlutKey(const char key) {
	return key;
}

void KeyState::glutKeyPress(const unsigned char key) {
	checkGlutModifiers();
	keyPress(key);
}

void KeyState::glutKeyUp(const unsigned char key) {
	checkGlutModifiers();
	keyUp(key);
}

void KeyState::glutSpecialKeyPress(const int key) {
	checkGlutModifiers();
	keyPress(convertGlutSpecialKey(key));
}

void KeyState::glutSpecialKeyUp(const int key) {
	checkGlutModifiers();
	keyUp(convertGlutSpecialKey(key));
}

void KeyState::glutMouseButton(const int button, const int state) {
	checkGlutModifiers();
	if(state == GLUT_UP) {
		keyUp(button+2*keyDelta);
	}
	else {
		keyPress(button+2*keyDelta);
	}
}

//namespace
}