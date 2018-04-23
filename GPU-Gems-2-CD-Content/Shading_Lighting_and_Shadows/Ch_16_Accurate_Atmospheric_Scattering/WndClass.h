/*
s_p_oneil@hotmail.com
Copyright (c) 2000, Sean O'Neil
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of this project nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __WndClass_h__
#define __WndClass_h__

#include "Resource.h"

// Assertion and trace declarations
#include <assert.h>
#ifdef _DEBUG
#define ASSERT			assert
#define TRACE			Trace
#else
#define ASSERT			((void)0)
#define TRACE			((void)0)
#endif

#define RELEASE(a)		if(a) { a->Release(); a = NULL; }


inline void Trace(const char *pszFormat, ...)
{
	char szBuffer[8192];
	va_list va;
	va_start(va, pszFormat);
	vsprintf(szBuffer, pszFormat, va);
	va_end(va);
	strcat(szBuffer, "\n");
	OutputDebugString(szBuffer);
}


// Window and control wrapper classes
class CRect : public RECT
{
public:
	CRect()											{}
	CRect(int x1, int y1, int x2, int y2)			{ left = x1; top = y1; right = x2; bottom = y2; }
	CRect(POINT tl, POINT br)						{ left = tl.x; top = tl.y; right = br.x; bottom = br.y; }
	int Width()										{ return right - left; }
	int Height()									{ return bottom - top; }
	void SetRect(int x1, int y1, int x2, int y2)	{ left = x1; top = y1; right = x2; bottom = y2; }
	void SetRect(POINT tl, POINT br)				{ left = tl.x; top = tl.y; right = br.x; bottom = br.y; }
	void SetRectEmpty()								{ left = top = right = bottom = 0; }
};

class CWnd
{
public:
	HWND m_hWnd;

	CWnd(HWND hWnd=NULL)							{ m_hWnd = hWnd; }
	virtual ~CWnd()									{}

	operator HWND()									{ return m_hWnd; }
	bool operator==(HWND hWnd)						{ return hWnd == m_hWnd; }
	bool operator!=(HWND hWnd)						{ return hWnd != m_hWnd; }
	void operator=(HWND hWnd)						{ m_hWnd = hWnd; }

	BOOL CreateEx(HINSTANCE hInstance, LPCTSTR lpClassName, LPCTSTR lpWindowName, DWORD dwExStyle, DWORD dwStyle, LPCRECT pRect, HWND hParent=NULL, HMENU hMenu=NULL, LPVOID lpParam=NULL)
	{
		m_hWnd = ::CreateWindowEx(dwExStyle, lpClassName, lpWindowName, dwStyle, pRect->left, pRect->top, pRect->right-pRect->left, pRect->bottom-pRect->top, hParent, hMenu, hInstance, lpParam);
		return (m_hWnd != NULL);
	}
	BOOL DestroyWindow()							{ return ::DestroyWindow(m_hWnd); }

	long SetWindowLong(int nIndex, long nValue)		{ return ::SetWindowLong(m_hWnd, nIndex, nValue); }
	long GetWindowLong(int nIndex)					{ return ::GetWindowLong(m_hWnd, nIndex); }
	DWORD GetStyle()								{ return (DWORD)::GetWindowLong(m_hWnd, GWL_STYLE); }
	void SetStyle(DWORD dw)							{ ::SetWindowLong(m_hWnd, GWL_STYLE, dw); }
	DWORD GetExStyle()								{ return (DWORD)::GetWindowLong(m_hWnd, GWL_EXSTYLE); }
	void SetExStyle(DWORD dw)						{ ::SetWindowLong(m_hWnd, GWL_EXSTYLE, dw); }

	LRESULT SendMessage(UINT n, WPARAM w=0, LPARAM l=0)	{ return ::SendMessage(m_hWnd, n, w, l); }
	BOOL PostMessage(UINT n, WPARAM w=0, LPARAM l=0){ return ::PostMessage(m_hWnd, n, w, l); }
	void SetWindowText(LPCTSTR psz)					{ ::SetWindowText(m_hWnd, psz); }
	int GetWindowText(LPTSTR psz, int n)			{ return ::GetWindowText(m_hWnd, psz, n); }
	int GetWindowTextLength()						{ return ::GetWindowTextLength(m_hWnd); }
	
	void UpdateWindow()								{ ::UpdateWindow(m_hWnd); }
	BOOL ShowWindow(int nCmdShow)					{ return ::ShowWindow(m_hWnd, nCmdShow); }
	BOOL EnableWindow(BOOL bEnable)					{ return ::EnableWindow(m_hWnd, bEnable); }
	BOOL IsWindowEnabled()							{ return ::IsWindowEnabled(m_hWnd); }
	BOOL IsWindowVisible()							{ return ::IsWindowVisible(m_hWnd); }
	BOOL IsIconic()									{ return ::IsIconic(m_hWnd); }
	BOOL IsZoomed()									{ return ::IsZoomed(m_hWnd); }
	void MoveWindow(LPCRECT pRect, BOOL b=TRUE)		{ MoveWindow(pRect->left, pRect->top, pRect->right - pRect->left, pRect->bottom - pRect->top, b); }
	void MoveWindow(int x, int y, int w, int h, BOOL b=TRUE)	{ ::MoveWindow(m_hWnd, x, y, w, h, b); }
	BOOL SetWindowPos(HWND hWnd, LPCRECT pRect, UINT n=0)	{ return ::SetWindowPos(m_hWnd, hWnd, pRect->left, pRect->top, pRect->right - pRect->left, pRect->bottom - pRect->top, n); }
	BOOL SetWindowPos(HWND hWnd, int x, int y, int w, int h, UINT n=0)	{ return ::SetWindowPos(m_hWnd, hWnd, x, y, w, h, n); }
	void BringWindowToTop()							{ ::BringWindowToTop(m_hWnd); }
	void GetWindowRect(LPRECT pRect)				{ ::GetWindowRect(m_hWnd, pRect); }
	void GetClientRect(LPRECT pRect)				{ ::GetClientRect(m_hWnd, pRect); }
	void ClientToScreen(LPPOINT pPoint)				{ ::ClientToScreen(m_hWnd, pPoint); }
	void ClientToScreen(LPRECT pRect)				{ ::ClientToScreen(m_hWnd, (LPPOINT)pRect); ::ClientToScreen(m_hWnd, ((LPPOINT)pRect)+1); }
	void ScreenToClient(LPPOINT pPoint)				{ ::ScreenToClient(m_hWnd, pPoint); }
	void ScreenToClient(LPRECT pRect)				{ ::ScreenToClient(m_hWnd, (LPPOINT)pRect); ::ScreenToClient(m_hWnd, ((LPPOINT)pRect)+1); }
	void CalcWindowRect(LPRECT pRect, UINT n=0)		{ ::AdjustWindowRect(pRect, GetStyle(), FALSE); }

	HWND GetActiveWindow()							{ return ::GetActiveWindow(); }
	HWND SetActiveWindow()							{ return ::SetActiveWindow(m_hWnd); }
	HWND GetCapture()								{ return ::GetCapture(); }
	HWND SetCapture()								{ return ::SetCapture(m_hWnd); }
	HWND GetFocus()									{ return ::GetFocus(); }
	HWND SetFocus()									{ return ::SetFocus(m_hWnd); }
	HWND GetParent()								{ return ::GetParent(m_hWnd); }
	HWND SetParent(HWND hWnd)						{ return ::SetParent(m_hWnd, hWnd); }
	HICON SetIcon(HICON h, BOOL b)					{ return (HICON)::SendMessage(m_hWnd, WM_SETICON, b, (LPARAM)h); }
	HICON GetIcon(BOOL b)							{ return (HICON)::SendMessage(m_hWnd, WM_GETICON, b, 0); }
	HMENU GetMenu()									{ return ::GetMenu(m_hWnd); }
	BOOL SetMenu(HMENU hMenu)						{ return ::SetMenu(m_hWnd, hMenu); }
	void DrawMenuBar()								{ ::DrawMenuBar(m_hWnd); }
	void RedrawWindow(LPCRECT pRect, UINT n)		{ ::RedrawWindow(m_hWnd, pRect, NULL, n); }
	int GetDlgCtrlID()								{ return ::GetDlgCtrlID(m_hWnd); }

	void SetRedraw(BOOL b)							{ ::SendMessage(m_hWnd, WM_SETREDRAW, b, 0); }
	void Invalidate(BOOL b=TRUE, LPCRECT pRect=NULL){ ::InvalidateRect(m_hWnd, pRect, b); }
	void Validate(LPCRECT pRect=NULL)				{ ::ValidateRect(m_hWnd, pRect); }
	UINT SetTimer(UINT nID, UINT nElapse)			{ return ::SetTimer(m_hWnd, nID, nElapse, NULL); }
	BOOL KillTimer(int nID)							{ return ::KillTimer(m_hWnd, nID); }
};

class CDialog : public CWnd
{
protected:
	int m_nID;
	HINSTANCE m_hInstance;
	HWND m_hWndParent;

public:
	CDialog(int nID=0, HINSTANCE hInstance=NULL, HWND hWndParent=NULL)
	{
		m_nID = nID;
		m_hInstance = hInstance;
		m_hWndParent = hWndParent;
	}
	static BOOL CALLBACK DlgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		CDialog *pDlg;
		switch(uMsg)
		{
			case WM_INITDIALOG:
			{
				pDlg = (CDialog *)lParam;
				::SetWindowLong(hWnd, GWL_USERDATA, (long)pDlg);
				pDlg->m_hWnd = hWnd;
				pDlg->CenterDialog();
				return pDlg->OnInitDialog(hWnd);
			}
			case WM_COMMAND:
			{
				pDlg = (CDialog *)::GetWindowLong(hWnd, GWL_USERDATA);
				if(wParam == IDOK)
					return pDlg->OnOK();
				if(wParam == IDCANCEL)
					return pDlg->OnCancel();
				return pDlg->OnCommand(wParam, lParam);
			}
		}
		return FALSE;
	}
	int DoModal()								{ return ::DialogBoxParam(m_hInstance, MAKEINTRESOURCE(m_nID), m_hWndParent, DlgProc, (LPARAM)this); }
	virtual bool OnInitDialog(HWND hWnd)		{ return true; }
	virtual bool OnOK()							{ ::EndDialog(m_hWnd, IDOK); return false; }
	virtual bool OnCancel()						{ ::EndDialog(m_hWnd, IDCANCEL); return false; }
	virtual bool OnCommand(WPARAM w, LPARAM l)	{ return false; }
	void CenterDialog()
	{
		RECT rect1, rect2;
		GetWindowRect(&rect1);
		CWnd(m_hWndParent ? m_hWndParent : GetDesktopWindow()).GetWindowRect(&rect2);
		rect1.left = rect2.left + ((rect2.right-rect2.left)-(rect1.right-rect1.left))/2;
		rect1.top = rect2.top + ((rect2.bottom-rect2.top)-(rect1.bottom-rect1.top))/2;
		SetWindowPos(NULL, &rect1, SWP_NOSIZE | SWP_NOZORDER);
	}
};

class CStatic : public CWnd
{
	CStatic(HWND hWndParent, int nComboID)
	{
		m_hWnd = ::GetDlgItem(hWndParent, nComboID);
		ASSERT(::IsWindow(m_hWnd));
	}

	HICON SetIcon(HICON hIcon)					{ return (HICON)::SendMessage(m_hWnd, STM_SETICON, (WPARAM)hIcon, 0L); }
	HICON GetIcon() const						{ return (HICON)::SendMessage(m_hWnd, STM_GETICON, 0, 0L); }
	HENHMETAFILE SetEnhMetaFile(HENHMETAFILE h)	{ return (HENHMETAFILE)::SendMessage(m_hWnd, STM_SETIMAGE, IMAGE_ENHMETAFILE, (LPARAM)h); }
	HENHMETAFILE GetEnhMetaFile() const			{ return (HENHMETAFILE)::SendMessage(m_hWnd, STM_GETIMAGE, IMAGE_ENHMETAFILE, 0L); }
	HBITMAP SetBitmap(HBITMAP hBitmap)			{ return (HBITMAP)::SendMessage(m_hWnd, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hBitmap); }
	HBITMAP GetBitmap() const					{ return (HBITMAP)::SendMessage(m_hWnd, STM_GETIMAGE, IMAGE_BITMAP, 0L); }
	HCURSOR SetCursor(HCURSOR hCursor)			{ return (HCURSOR)::SendMessage(m_hWnd, STM_SETIMAGE, IMAGE_CURSOR, (LPARAM)hCursor); }
	HCURSOR GetCursor()							{ return (HCURSOR)::SendMessage(m_hWnd, STM_GETIMAGE, IMAGE_CURSOR, 0L); }
};

class CEdit : public CWnd
{
public:
	CEdit(HWND hWndParent, int nComboID)
	{
		m_hWnd = ::GetDlgItem(hWndParent, nComboID);
		ASSERT(::IsWindow(m_hWnd));
	}
	void GetSel(int& nStartChar, int& nEndChar)	{ ::SendMessage(m_hWnd, EM_GETSEL, (WPARAM)&nStartChar,(LPARAM)&nEndChar); }
	DWORD GetSel()								{ return ::SendMessage(m_hWnd, EM_GETSEL, 0, 0); }
	void SetSel(DWORD dwSelection)				{ ::SendMessage(m_hWnd, EM_SETSEL, LOWORD(dwSelection), HIWORD(dwSelection)); }
	void SetSel(int nStartChar, int nEndChar)	{ ::SendMessage(m_hWnd, EM_SETSEL, nStartChar, nEndChar); }
	void SetLimitText(UINT nMax)				{ ::SendMessage(m_hWnd, EM_SETLIMITTEXT, nMax, 0); }
	void Clear()								{ ::SendMessage(m_hWnd, WM_CLEAR, 0, 0); }
	void Copy()									{ ::SendMessage(m_hWnd, WM_COPY, 0, 0); }
	void Cut()									{ ::SendMessage(m_hWnd, WM_CUT, 0, 0); }
	void Paste()								{ ::SendMessage(m_hWnd, WM_PASTE, 0, 0); }
	BOOL SetReadOnly(BOOL bReadOnly)			{ return (BOOL)::SendMessage(m_hWnd, EM_SETREADONLY, bReadOnly, 0L); }
};

class CButton : public CWnd
{
public:
	CButton(HWND hWndParent, int nComboID)
	{
		m_hWnd = ::GetDlgItem(hWndParent, nComboID);
		ASSERT(::IsWindow(m_hWnd));
	}

	UINT GetState()							{ return (UINT)::SendMessage(m_hWnd, BM_GETSTATE, 0, 0); }
	void SetState(BOOL b)					{ ::SendMessage(m_hWnd, BM_SETSTATE, b, 0); }
	int GetCheck()							{ return (int)::SendMessage(m_hWnd, BM_GETCHECK, 0, 0); }
	void SetCheck(int n)					{ ::SendMessage(m_hWnd, BM_SETCHECK, n, 0); }
	UINT GetButtonStyle()					{ return (UINT)::GetWindowLong(m_hWnd, GWL_STYLE) & 0xff; }
	void SetButtonStyle(UINT n, BOOL b)		{ ::SendMessage(m_hWnd, BM_SETSTYLE, n, (LPARAM)b); }
};

class CComboBox : public CWnd
{
public:
	CComboBox(HWND hWndParent, int nComboID)
	{
		m_hWnd = ::GetDlgItem(hWndParent, nComboID);
		ASSERT(::IsWindow(m_hWnd));
	}
	int GetCount()							{ return (int)::SendMessage(m_hWnd, CB_GETCOUNT, 0, 0); }
	int GetCurSel()							{ return (int)::SendMessage(m_hWnd, CB_GETCURSEL, 0, 0); }
	int SetCurSel(int n)					{ return (int)::SendMessage(m_hWnd, CB_SETCURSEL, n, 0); }
	DWORD GetEditSel()						{ return (DWORD)::SendMessage(m_hWnd, CB_GETEDITSEL, 0, 0); }
	BOOL LimitText(int n)					{ return (BOOL)::SendMessage(m_hWnd, CB_LIMITTEXT, n, 0); }
	BOOL SetEditSel(int nStart, int nEnd)	{ return (BOOL)::SendMessage(m_hWnd, CB_SETEDITSEL, 0, MAKELONG(nStart, nEnd)); }
	DWORD GetItemData(int n)				{ return ::SendMessage(m_hWnd, CB_GETITEMDATA, n, 0); }
	int SetItemData(int n, DWORD dw)		{ return (int)::SendMessage(m_hWnd, CB_SETITEMDATA, n, (LPARAM)dw); }
	void *GetItemDataPtr(int n)				{ return (LPVOID)GetItemData(n); }
	int SetItemDataPtr(int n, void *pData)	{ return SetItemData(n, (DWORD)(LPVOID)pData); }
	int GetLBText(int n, LPTSTR psz)		{ return (int)::SendMessage(m_hWnd, CB_GETLBTEXT, n, (LPARAM)psz); }
	int GetLBTextLen(int n)					{ return (int)::SendMessage(m_hWnd, CB_GETLBTEXTLEN, n, 0); }
	int AddString(LPCTSTR psz)				{ return (int)::SendMessage(m_hWnd, CB_ADDSTRING, 0, (LPARAM)psz); }
	int DeleteString(UINT n)				{ return (int)::SendMessage(m_hWnd, CB_DELETESTRING, n, 0);}
	int InsertString(int n, LPCTSTR psz)	{ return (int)::SendMessage(m_hWnd, CB_INSERTSTRING, n, (LPARAM)psz); }
	void ResetContent()						{ ::SendMessage(m_hWnd, CB_RESETCONTENT, 0, 0); }
	int FindString(int n, LPCTSTR psz)		{ return (int)::SendMessage(m_hWnd, CB_FINDSTRING, n, (LPARAM)psz); }
	int SelectString(int n, LPCTSTR psz)	{ return (int)::SendMessage(m_hWnd, CB_SELECTSTRING, n, (LPARAM)psz); }
};

class CListBox : public CWnd
{
	CListBox(HWND hWndParent, int nComboID)
	{
		m_hWnd = ::GetDlgItem(hWndParent, nComboID);
		ASSERT(::IsWindow(m_hWnd));
	}

	int GetCount() const					{ return (int)::SendMessage(m_hWnd, LB_GETCOUNT, 0, 0); }
	int GetCurSel() const					{ return (int)::SendMessage(m_hWnd, LB_GETCURSEL, 0, 0); }
	int SetCurSel(int n)					{ return (int)::SendMessage(m_hWnd, LB_SETCURSEL, n, 0); }
	int GetHorizontalExtent() const			{ return (int)::SendMessage(m_hWnd, LB_GETHORIZONTALEXTENT, 0, 0); }
	void SetHorizontalExtent(int n)			{ ::SendMessage(m_hWnd, LB_SETHORIZONTALEXTENT, n, 0); }
	int GetSelCount() const					{ return (int)::SendMessage(m_hWnd, LB_GETSELCOUNT, 0, 0); }
	int GetSelItems(int n, LPINT rg) const	{ return (int)::SendMessage(m_hWnd, LB_GETSELITEMS, n, (LPARAM)rg); }
	int GetTopIndex() const					{ return (int)::SendMessage(m_hWnd, LB_GETTOPINDEX, 0, 0); }
	int SetTopIndex(int n)					{ return (int)::SendMessage(m_hWnd, LB_SETTOPINDEX, n, 0);}
	DWORD GetItemData(int n) const			{ return ::SendMessage(m_hWnd, LB_GETITEMDATA, n, 0); }
	int SetItemData(int n, DWORD dw)		{ return (int)::SendMessage(m_hWnd, LB_SETITEMDATA, n, (LPARAM)dw); }
	void* GetItemDataPtr(int n) const		{ return (LPVOID)::SendMessage(m_hWnd, LB_GETITEMDATA, n, 0); }
	int SetItemDataPtr(int n, void *pData)	{ return SetItemData(n, (DWORD)(LPVOID)pData); }
	int GetItemRect(int n, LPRECT p) const	{ return (int)::SendMessage(m_hWnd, LB_GETITEMRECT, n, (LPARAM)p); }
	int GetSel(int n) const					{ return (int)::SendMessage(m_hWnd, LB_GETSEL, n, 0); }
	int SetSel(int n, BOOL b)				{ return (int)::SendMessage(m_hWnd, LB_SETSEL, b, n); }
	int GetText(int n, LPTSTR psz) const	{ return (int)::SendMessage(m_hWnd, LB_GETTEXT, n, (LPARAM)psz); }
	int GetTextLen(int n) const				{ return (int)::SendMessage(m_hWnd, LB_GETTEXTLEN, n, 0); }
	void SetColumnWidth(int n)				{ ::SendMessage(m_hWnd, LB_SETCOLUMNWIDTH, n, 0); }
	BOOL SetTabStops(int n, LPINT rg)		{ return (BOOL)::SendMessage(m_hWnd, LB_SETTABSTOPS, n, (LPARAM)rg); }
	void SetTabStops()						{ ::SendMessage(m_hWnd, LB_SETTABSTOPS, 0, 0); }
	BOOL SetTabStops(const int& cx)			{ return (BOOL)::SendMessage(m_hWnd, LB_SETTABSTOPS, 1, (LPARAM)(LPINT)&cx); }
	int SetItemHeight(int n, UINT cy)		{ return (int)::SendMessage(m_hWnd, LB_SETITEMHEIGHT, n, MAKELONG(cy, 0)); }
	int GetItemHeight(int n) const			{ return (int)::SendMessage(m_hWnd, LB_GETITEMHEIGHT, n, 0L); }
	int FindStringExact(int n, LPCTSTR psz) const	{ return (int)::SendMessage(m_hWnd, LB_FINDSTRINGEXACT, n, (LPARAM)psz); }
	int GetCaretIndex() const				{ return (int)::SendMessage(m_hWnd, LB_GETCARETINDEX, 0, 0L); }
	int SetCaretIndex(int n, BOOL bScroll)	{ return (int)::SendMessage(m_hWnd, LB_SETCARETINDEX, n, MAKELONG(bScroll, 0)); }
	int AddString(LPCTSTR lpszItem)			{ return (int)::SendMessage(m_hWnd, LB_ADDSTRING, 0, (LPARAM)lpszItem); }
	int DeleteString(UINT n)				{ return (int)::SendMessage(m_hWnd, LB_DELETESTRING, n, 0); }
	int InsertString(int n, LPCTSTR psz)	{ return (int)::SendMessage(m_hWnd, LB_INSERTSTRING, n, (LPARAM)psz); }
	void ResetContent()						{ ::SendMessage(m_hWnd, LB_RESETCONTENT, 0, 0); }
	int Dir(UINT attr, LPCTSTR psz)			{ return (int)::SendMessage(m_hWnd, LB_DIR, attr, (LPARAM)psz); }
	int FindString(int n, LPCTSTR psz) const{ return (int)::SendMessage(m_hWnd, LB_FINDSTRING, n, (LPARAM)psz); }
	int SelectString(int n, LPCTSTR psz)	{ return (int)::SendMessage(m_hWnd, LB_SELECTSTRING, n, (LPARAM)psz); }
	int SelItemRange(BOOL b, int f, int l)	{ return b ? (int)::SendMessage(m_hWnd, LB_SELITEMRANGEEX, f, l) : (int)::SendMessage(m_hWnd, LB_SELITEMRANGEEX, l, f); }
	void SetAnchorIndex(int n)				{ ::SendMessage(m_hWnd, LB_SETANCHORINDEX, n, 0); }
	int GetAnchorIndex() const				{ return (int)::SendMessage(m_hWnd, LB_GETANCHORINDEX, 0, 0); }
	LCID GetLocale() const					{ return (LCID)::SendMessage(m_hWnd, LB_GETLOCALE, 0, 0); }
	LCID SetLocale(LCID n)					{ return (LCID)::SendMessage(m_hWnd, LB_SETLOCALE, (WPARAM)n, 0); }
};


// Device context and drawing classes
class CDC
{
public:
	HDC m_hDC;
	HWND m_hWnd;

public:
	CDC(HDC hDC=NULL, HWND hWnd=NULL)				{ m_hDC = hDC; m_hWnd = hWnd; }
	operator HDC()									{ return m_hDC; }
	bool operator==(HDC hDC)						{ return hDC == m_hDC; }
	bool operator!=(HDC hDC)						{ return hDC != m_hDC; }
	void operator=(HDC hDC)							{ m_hDC = hDC; }

	COLORREF SetBkColor(COLORREF c)					{ return ::SetBkColor(m_hDC, c); }
	int SetBkMode(int nMode)						{ return ::SetBkMode(m_hDC, nMode); }
	COLORREF SetTextColor(COLORREF c)				{ return ::SetTextColor(m_hDC, c); }
	BOOL TextOut(int x, int y, const char *psz)		{ return ::TextOut(m_hDC, x, y, psz, strlen(psz)); }
	HGDIOBJ SelectObject(HGDIOBJ hGDI)				{ return ::SelectObject(m_hDC, hGDI); }
	BOOL BitBlt(HDC hDC, int x, int y, int w, int h, int xSrc=0, int ySrc=0, DWORD dwROP=SRCCOPY)
	{
		return ::BitBlt(hDC, x, y, w, h, m_hDC, xSrc, ySrc, dwROP);
	}
	BOOL CenteredTextOut(const char *psz)
	{
		CRect rect;
		SIZE size;
		int nLength = strlen(psz);
		::GetClientRect(m_hWnd, &rect);
		::GetTextExtentPoint32(m_hDC, psz, nLength, &size);
		return ::TextOut(m_hDC, (rect.Width()-size.cx)/2, (rect.Height()-size.cy)/2, psz, nLength);
	}
};

class CClientDC : public CDC
{
public:
	CClientDC(HWND hWnd=NULL)	{ if(hWnd) Init(hWnd); }
	void Init(HWND hWnd)		{ m_hWnd = hWnd; m_hDC = ::GetDC(m_hWnd); }
	void Release()				{ if(m_hDC) { ::ReleaseDC(m_hWnd, m_hDC); m_hDC = NULL; } }
	~CClientDC()				{ Release(); }
};

class CWindowDC : public CDC
{
public:
	CWindowDC(HWND hWnd=NULL)	{ if(hWnd) Init(hWnd); }
	void Init(HWND hWnd)		{ m_hWnd = hWnd; m_hDC = ::GetWindowDC(m_hWnd); }
	void Release()				{ if(m_hDC) { ::ReleaseDC(m_hWnd, m_hDC); m_hDC = NULL; } }
	~CWindowDC()				{ Release(); }
};

class CPaintDC : public CDC
{
public:
	PAINTSTRUCT m_ps;

public:
	CPaintDC(HWND hWnd)			{ m_hWnd = hWnd; ::BeginPaint(m_hWnd, &m_ps); m_hDC = m_ps.hdc; }
	~CPaintDC()					{ ::EndPaint(m_hWnd, &m_ps); }
};

class CCompatibleDC : public CDC
{
public:
	CCompatibleDC(HDC hDC=NULL)	{ m_hDC = ::CreateCompatibleDC(hDC); }
	~CCompatibleDC()			{ if(m_hDC) ::DeleteDC(m_hDC); }
};

class CWinApp : public CWnd
{
// Attributes
protected:
	HINSTANCE m_hInstance;
	HINSTANCE m_hPrevInstance;
	const char *m_pszCmdLine;
	int m_nShowCmd;
	char m_szAppName[_MAX_PATH];
	char m_szAppPath[_MAX_PATH];
	char m_szStartupPath[_MAX_PATH];
	char m_szRegistryKey[_MAX_PATH];

public:
	static CWinApp *m_pMainApp;

// Operations
protected:

public:	
	CWinApp(HINSTANCE hInstance, HINSTANCE hPrevInstance=NULL, char *pszCmdLine="", int nShowCmd=SW_SHOWNORMAL)
	{
		m_pMainApp = this;
		m_hInstance = hInstance;
		m_hPrevInstance = hPrevInstance;
		m_pszCmdLine = pszCmdLine;
		m_nShowCmd = nShowCmd;
		LoadString(IDS_APPLICATION, m_szAppName);
		::GetModuleFileName(NULL, m_szAppPath, _MAX_PATH);
		::GetCurrentDirectory(_MAX_PATH, m_szStartupPath);
		sprintf(m_szRegistryKey, "Software\\%s", m_szAppName);
	}
	~CWinApp()
	{
		m_pMainApp = NULL;
	}

	virtual bool InitInstance()					{ return false; }
	virtual int ExitInstance()					{ return 0; }
	virtual bool OnIdle()						{ return false; }
	virtual bool PreTranslateMessage(MSG *pMsg)	{ return false; }
	virtual int Run()
	{
		MSG msg;
		msg.message = 0;
		while(msg.message != WM_QUIT)
		{
			OnIdle();
			while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				if(msg.message == WM_QUIT)
					break;
				if(!PreTranslateMessage(&msg))
				{
					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
			}
		}
		return 0;
	}

	HINSTANCE GetInstanceHandle()				{ return m_hInstance; }
	const char *GetAppName()					{ return m_szAppName; }
	const char *GetAppPath()					{ return m_szAppPath; }
	const char *GetStartupPath()				{ return m_szStartupPath; }
	const char *GetRegistryKey()				{ return m_szRegistryKey; }
	void SetAppName(const char *psz)			{ strcpy(m_szAppName, psz); }
	void SetAppPath(const char *psz)			{ strcpy(m_szAppPath, psz); }
	void SetStartupPath(const char *psz)		{ strcpy(m_szStartupPath, psz); }
	void SetRegistryKey(const char *psz)		{ strcpy(m_szRegistryKey, psz); }

	int MessageBox(const char *psz, UINT uType=MB_OK)		{ return ::MessageBox(NULL, psz, m_szAppName, uType); }
	int LoadString(int nID, char *psz, int nMax=_MAX_PATH)	{ return ::LoadString(m_hInstance, nID, psz, nMax); }
	int GetProfileInt(const char *pszSection, const char *pszEntry, int nDefault=0)
	{
		HKEY hKey;
		DWORD dw, dwType, dwValue;
		char szBuffer[_MAX_PATH];
		sprintf(szBuffer, "%s\\%s", m_szRegistryKey, pszSection);
		if(RegCreateKeyEx(HKEY_CURRENT_USER, szBuffer, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, &dw) == ERROR_SUCCESS)
		{
			dw = sizeof(DWORD);
			if(RegQueryValueEx(hKey, pszEntry, NULL, &dwType, (unsigned char *)&dwValue, &dw) == ERROR_SUCCESS && dwType == REG_DWORD)
				nDefault = dwValue;
			RegCloseKey(hKey);
		}
		return nDefault;
	}
	bool WriteProfileInt(const char *pszSection, const char *pszEntry, int nValue)
	{
		HKEY hKey;
		DWORD dw;
		char szBuffer[_MAX_PATH];
		sprintf(szBuffer, "%s\\%s", m_szRegistryKey, pszSection);
		bool bSuccess = false;
		if(RegCreateKeyEx(HKEY_CURRENT_USER, szBuffer, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, &dw) == ERROR_SUCCESS)
		{
			bSuccess = (RegSetValueEx(hKey, pszEntry, 0, REG_DWORD, (unsigned char *)&nValue, sizeof(DWORD)) == ERROR_SUCCESS);
			RegCloseKey(hKey);
		}
		return bSuccess;
	}
	const char *GetProfileString(const char *pszSection, const char *pszEntry, const char *pszDefault="")
	{
		HKEY hKey;
		DWORD dw, dwType;
		static char szBuffer[_MAX_PATH];
		sprintf(szBuffer, "%s\\%s", m_szRegistryKey, pszSection);
		if(RegCreateKeyEx(HKEY_CURRENT_USER, szBuffer, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, &dw) == ERROR_SUCCESS)
		{
			dw = _MAX_PATH;
			if(RegQueryValueEx(hKey, pszEntry, NULL, &dwType, (unsigned char *)szBuffer, &dw) == ERROR_SUCCESS && dwType == REG_SZ)
				pszDefault = szBuffer;
			RegCloseKey(hKey);
		}
		return pszDefault;
	}
	bool WriteProfileString(const char *pszSection, const char *pszEntry, const char *pszValue)
	{
		HKEY hKey;
		DWORD dw;
		char szBuffer[_MAX_PATH];
		sprintf(szBuffer, "%s\\%s", m_szRegistryKey, pszSection);
		bool bSuccess = false;
		if(RegCreateKeyEx(HKEY_CURRENT_USER, szBuffer, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, &dw) == ERROR_SUCCESS)
		{
			bSuccess = pszValue ? (RegSetValueEx(hKey, pszEntry, 0, REG_SZ, (unsigned char *)pszValue, strlen(pszValue)+1) == ERROR_SUCCESS) :
								  (RegDeleteValue(hKey, pszEntry) == ERROR_SUCCESS);
			RegCloseKey(hKey);
		}
		return bSuccess;
	}
};

inline CWinApp *GetApp()		{ return CWinApp::m_pMainApp; }

#endif // __WndClass_h__
