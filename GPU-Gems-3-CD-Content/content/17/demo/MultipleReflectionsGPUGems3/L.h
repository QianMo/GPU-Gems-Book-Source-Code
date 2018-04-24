#pragma once

class L
{
	wchar_t* wide;
	unsigned int nWide;

	L(void);
	~L(void);
public:
	
	wchar_t* operator+(const char* multi);
	static L l;

	static wchar_t* clone(const char* multi);
	static wchar_t* cloneW(const wchar_t* wide);
};

class LC
{
	char* narrow;
	unsigned int nNarrow;

	LC(void);
	~LC(void);
public:
	
	char* operator-(const wchar_t* wide);
	static LC c;

	static char* clone(const wchar_t* wide);
};
