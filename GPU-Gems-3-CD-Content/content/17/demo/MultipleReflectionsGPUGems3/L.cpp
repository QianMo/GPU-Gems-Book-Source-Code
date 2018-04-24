#include "dxstdafx.h"
#include ".\l.h"

L::L(void)
{
	wide = NULL;
}

L::~L(void)
{
}

wchar_t* L::operator+(const char* multi)
{
	if(wide)
		delete wide;
	wide = new wchar_t[nWide = MultiByteToWideChar(CP_ACP, 0, multi, -1, NULL, 0)];
	MultiByteToWideChar(CP_ACP, 0, multi, -1, wide, nWide);
	return wide;
}

L L::l;

wchar_t* L::clone(const char* multi)
{
	l+multi;
	wchar_t* ret = new wchar_t[wcslen(l.wide)+1];
	wcscpy(ret, l.wide);
	return ret;
}

wchar_t* L::cloneW(const wchar_t* wide)
{
	wchar_t* ret = new wchar_t[wcslen(wide)+1];
	wcscpy(ret, wide);
	return ret;
}

LC::LC(void)
{
	narrow = NULL;
}

LC::~LC(void)
{
}

char* LC::operator-(const wchar_t* wide)
{
	if(narrow)
		delete narrow;
	narrow = new char[nNarrow = WideCharToMultiByte(CP_ACP, 0, wide, -1, NULL, 0, NULL, NULL)];
	WideCharToMultiByte(CP_ACP, 0, wide, -1, narrow, nNarrow, NULL, NULL);
	return narrow;
}

LC LC::c;

char* LC::clone(const wchar_t* wide)
{
	c-wide;
	char* ret = new char[strlen(c.narrow)+1];
	strcpy(ret, c.narrow);
	return ret;
}