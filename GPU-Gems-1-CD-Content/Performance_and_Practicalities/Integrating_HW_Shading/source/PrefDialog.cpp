// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a class and functions used to edit the preferences

#include "c4d.h"
#include "PrefDialog.h"
#include "IDnumbers.h"
#include "c4d_symbols.h"
#include "lib_prefs.h"
#include <assert.h>

// only pointers, otherwise Cinema will not load dll
static String* editorPath = 0;
static String* editorCL = 0;
// They can't be class members because the objects are destroyed all the time.

class FXPrefsDialog : public PrefsDlg_Base
{
public:
	FXPrefsDialog(PrefsDialogHookClass* hook);
	virtual void SetValues(BaseContainer *data);
	virtual LONG CommandValues(LONG id, const BaseContainer &msg, BaseContainer *data);
};

FXPrefsDialog::FXPrefsDialog(PrefsDialogHookClass* hook)
: PrefsDlg_Base(DLG_PREF, hook)
{}

void FXPrefsDialog::SetValues(BaseContainer *data)
{
	SetString(IDC_EDITOR_PATH, *editorPath);
	SetString(IDC_EDITOR_CL, *editorCL);
}

LONG FXPrefsDialog::CommandValues(LONG id, const BaseContainer &msg, BaseContainer *data)
{
	switch (id)
	{
	case IDC_EDITOR_PATH:
		if(!GetString(IDC_EDITOR_PATH, *editorPath))
			return FALSE;
		break;
	case IDC_EDITOR_CL:
		if(!GetString(IDC_EDITOR_CL, *editorCL))
			return FALSE;
		break;
	case IDC_SET_EDITOR_PATH:
		{
			Filename f;
			Bool ok = f.FileSelect(FSTYPE_ANYTHING, 0, NULL);
			if(ok)
			{
				*editorPath = f.GetString();
				SetString(IDC_EDITOR_PATH, *editorPath);
			}
		}
		break;
	}
	return TRUE;
}

class FXPrefs : public PrefsDialogHookClass
{
public:
	virtual SubDialog *Alloc();
	virtual void Free(SubDialog *dlg);
};

SubDialog* FXPrefs::Alloc()
{
	return gNew FXPrefsDialog(this);
}

void FXPrefs::Free(SubDialog *dlg)
{
	gDelete(dlg);
}

FXPrefs* prefs = NULL;
FXPrefsDialog* dlg = NULL;

Bool RegisterPrefs(void)
{
	editorPath = gNew String;
	if(editorPath == NULL)
		return FALSE;

	editorCL = gNew String;
	if(editorCL == NULL)
	{
		gDelete(editorPath)
		return FALSE;
	}

	prefs = gNew FXPrefs;
	dlg = (FXPrefsDialog*) prefs->Alloc();
	if(dlg == NULL)
		return FALSE;
	String name = GeLoadString(IDS_PREFS);
	if (!name.Content()) return FALSE;
	if(! prefs->Register(ID_PREFERENCES, name))
		return FALSE;

	Filename f = GeGetPluginPath() + Filename("C4Dfx.prefs");
	BaseFile* b = BaseFile::Alloc();
	if(b == NULL || ! b->Open(f) || ! b->ReadString(editorPath) || ! b->ReadString(editorCL))
	{
		*editorPath = "";
		*editorCL = "/L:<LINE> \"<FILE>\"";
	}
	BaseFile::Free(b);

	return TRUE;
}

void FreePrefs(void)
{
	Filename f = GeGetPluginPath() + Filename("C4Dfx.prefs");

	BaseFile* b = BaseFile::Alloc();
	if(b != NULL)
	{
		Bool result = b->Open(f, GE_WRITE) && b->WriteString(*editorPath) && b->WriteString(*editorCL);
		assert(result);
		BaseFile::Free(b);
	}

	if(prefs != NULL)
		prefs->Free(dlg);
	gDelete(prefs);

	gDelete(editorCL);
	gDelete(editorPath);
}

bool GetEditorPath(char* s)
{
	editorPath->GetCString(s, 255, St8bit); // should better be of dynamic length
	return true;
}

bool GetCommandL(char* s, int line, const char* file)
{
	String str(*editorCL);

	LONG linePos;
	Bool lineUsed;
	lineUsed = str.FindFirst("<LINE>", &linePos);

	String str1(str);

	if(lineUsed)
	{
		LONG sl = String("<LINE>").GetLength();
		str1 = str.SubStr(0, linePos)
			+ LongToString(line)
			+ str.SubStr(linePos + sl, str.GetLength() - linePos - sl);
	}

	LONG filePos;
	Bool fileUsed;
	fileUsed = str1.FindFirst("<FILE>", &filePos);

	String str2(str1);

	if(fileUsed)
	{
		LONG fl = String("<FILE>").GetLength();
		str2 = str1.SubStr(0, filePos)
			+ file
			+ str1.SubStr(filePos + fl, str1.GetLength() - filePos - fl);
	}
	
	str = " " + str2;
	str.GetCString(s, 255, St8bit); // should better be of dynamic length

	return true;
}