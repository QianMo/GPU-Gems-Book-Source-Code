// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a class for the C4Dfx material

#include "FXMaterial.h"
#include "c4d_symbols.h"
#include "Mfxmaterial.h"
#include "IDnumbers.h"
#include "C4DWrapper.h"
#include "WinInit.h"
#include <string.h>

FXMaterial::FXMaterial(void)
: doInit(false), forRendering(false), error(false)
{}

NodeData* FXMaterial::Alloc(void)
{
	return gNew FXMaterial;
}

void FXMaterial::CalcSurface(PluginMaterial *mat, VolumeData *vd)
{
	vd->col = Vector(0.0, 1.0, 0.0);
}

// We are loaded from a file.
Bool FXMaterial::Read(GeListNode* node, HyperFile* hf, LONG level)
{
	BaseContainer *data=((PluginMaterial*)node)->GetDataInstance();

	path = data->GetFilename(FX_PATH);
	
	doInit = true;
	setToDefault = false;

	AutoAlloc<Description> desc;
	if(!desc)
		return FALSE;
	long flags = 0L;
	if(!node->GetDescription(desc, flags)) // calls our GetDDescription
		return FALSE;

	return TRUE;
}

// The renderer wants to load us.
Bool FXMaterial::ReadForRendering(GeListNode* node)
{
	doInit = true;
	setToDefault = false;
	forRendering = true;
	Bool ok = Read(node, NULL, 0L); // abuse
	forRendering = false;
	return ok && !error;
}

Bool FXMaterial::GetDDescription(GeListNode* node, Description* description, LONG& flags)
{
	error = true;

	if (!description || !node) return FALSE;
	if (!description->LoadDescription(node->GetType())) return FALSE;

	BaseContainer *data=((PluginMaterial*)node)->GetDataInstance();

	Filename pathNew = data->GetFilename(FX_PATH);

	if(path != pathNew)
	{
		path = pathNew;
		doInit = true; // maybe doInit _is_ already true
		setToDefault = true;
	}

	if(doInit)
	{
		char pathC[256]; // should better be dynamic
		path.GetString().GetCString(pathC, 255, St8bit);
		
		const char* errors = 0;

		if(!wrapper.Load(pathC, &errors))
		{
			doInit = false; // even if we cannot load the file

			if(!forRendering && errors != 0 && errors[0] != 0) // no error message dialogs if we are rendering
			{
				GePrint(String("C4Dfx: ") + path.GetString());
				GePrint(errors);

				bool found = false; // found opening and closing parenthesis; line number is in between
				unsigned int cl = 0;
				if(errors[0] == '('  &&  strlen(errors) > 1)
				{
					for(cl = 1; cl<strlen(errors); ++cl)
					{
						if(errors[cl]==')')
						{
							found = true;
							break;
						}
					}
				}

				String s(errors);
				if(found)
				{
					s += "Edit .fx file?";
					if(QuestionDialog(s))
					{
						//Filename file = data->GetFilename(FX_PATH);
						char f[256];
						path.GetString().GetCString(f, 255, St8bit);
						WinInit::StartEditor(f, s.SubStr(1, cl-1).StringToLong());
					}
				}
				else // no syntax error
				{
					MessageDialog(s);
				}
			}
			return FALSE;
		}

		if(!forRendering) // we may savely throw away the effect
			wrapper.ReleaseEffect();
	}

	if(! wrapper.BuildUI_Technique(doInit, setToDefault, description, data))
		return FALSE;

	if(! wrapper.BuildUI_Parameters(doInit, setToDefault, description, data))
		return FALSE;

	doInit = false;
	error = false;

	flags |= DESCFLAGS_DESC_LOADED;
	return MaterialData::GetDDescription(node, description, flags);
}

Bool FXMaterial::GetDEnabling(GeListNode* node, const DescID& id, GeData& t_data, LONG flags, const BaseContainer* itemdesc)
{
	return wrapper.IsParamEnabled(id[0].id);
}

Bool FXMaterial::Message(GeListNode* node, LONG type, void* data)
{
	BaseContainer *dat = ((PluginMaterial*)node)->GetDataInstance();

	switch (type)
	{
		case MSG_DESCRIPTION_COMMAND:
			{
				switch(((DescriptionCommand*)data)->id[0].id)
				{
					case FX_RELOAD_W_DEFAULT:
						doInit = true;
						setToDefault = true;
						break;
					case FX_RELOAD:
						doInit = true;
						setToDefault = false;
						break;
					case FX_EDIT:
						{
							Filename file = dat->GetFilename(FX_PATH);
							char s[256];
							file.GetString().GetCString(s, 255, St8bit);
							WinInit::StartEditor(s, 1);
						}
						break;
					default: ;
				}
			}
			break;
		default: ;
	}

	return MaterialData::Message(node, type, data);
}

Bool FXMaterial::Init(GeListNode *node)
{
	BaseContainer *data=((PluginMaterial*)node)->GetDataInstance();
	return TRUE;
}

Bool FXMaterial::CopyTo(NodeData* dest, GeListNode* snode, GeListNode* dnode, LONG flags, AliasTrans* trn)
{
	FXMaterial* destm = (FXMaterial*)dest;
	destm->path = path;
	destm->doInit = true; // has to be loaded from .fx file
	destm->setToDefault = false; // has to be set to default values
	return TRUE;
}

FXWrapper* FXMaterial::GetFXWrapper(void)
{
	return &wrapper;
}

Bool RegisterFXMaterial(void)
{
	String name = GeLoadString(IDS_FXMATERIAL);
	return RegisterMaterialPlugin(ID_MATERIAL_PLUGIN, name, 0, FXMaterial::Alloc, "Mfxmaterial", 0);
}