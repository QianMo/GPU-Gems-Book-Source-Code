#include <windows.h>
#include "..\lib\paralelo3d.h"
#include "resource.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>

extern HWND hWndMain;
extern HINSTANCE hInstance;
extern pRender *render;

char *appname="pStereogram";
char *appfile="pStereogram.exe";

void draw_scene();
void key_pressed(int key,int flags);
void animate();
void init();
void destroy();
void command(UINT msg, WPARAM wParam, LPARAM lParam);
void update_menu();
void set_demo(int demo);
void load_scene();

int depthmap=-1;
int tilemap=-1;
int resmap=-1;

int viewmode=2;
int ressize=512;
int substrips=4;
int depthinvert=0;

float factors[]={ 0.1f,0.25f,0.5f,0.75f,1.0f,2.0f };
int strips[]={ 8,12,16 };
int selfactor=1,selstrip=0;

pString meshfile(" data\\mesh\\room1.p3d");
pString depthfile(" data\\depth\\depth1.tga");
pString tilefile(" data\\tile\\tile1.tga");

CGprogram prog_frag = NULL;

CGparameter param_frag_depthmap = 0;
CGparameter param_frag_resmap = 0;
CGparameter param_frag_info = 0;
CGparameter param_frag_factor = 0;

#define frag_ext	"GL_ARB_fragment_program"
#define frag_profile CG_PROFILE_ARBFP1

void init()
{	
	render->texmipmap=0;

	char *glstr=(char *)glGetString(GL_EXTENSIONS);
	if (!strstr(glstr,frag_ext))
		MessageBox(hWndMain,"Fragment program extension is not available !\n\nThis demo will execute in this enviroment.",appfile,MB_OK|MB_ICONSTOP);

	pString str;
	char *programtext=0;
	
	str=meshfile[0]!=' '?meshfile:render->app_path+((const char *)meshfile+1);
	render->load_mesh(str);

	str=depthfile[0]!=' '?depthfile:render->app_path+((const char *)depthfile+1);
	depthmap=render->load_tex(str);

	str=tilefile[0]!=' '?tilefile:render->app_path+((const char *)tilefile+1);
	tilemap=render->load_tex(str);

	resmap=render->create_texture(ressize,ressize,4,0);

	str=render->app_path;
	str+="cg\\stereogram_frag.cg";
	programtext=CgLoadProgramText(str);
	if (programtext==0)
		MessageBox(hWndMain,"Error loading fragment program!",appfile,MB_OK);
	else
	{
		prog_frag = cgCreateProgram(render->cgcontext, CG_SOURCE, programtext, frag_profile, "main_frag",0);
		CgCheckError();
		if (prog_frag==0)
			MessageBox(hWndMain,cgGetErrorString(cgGetError()),appfile,MB_OK);
		else
		{
			cgGLLoadProgram(prog_frag);
			CgCheckError();

			param_frag_depthmap = cgGetNamedParameter(prog_frag, "depthmap");
			param_frag_resmap = cgGetNamedParameter(prog_frag, "resmap");
			param_frag_info = cgGetNamedParameter(prog_frag, "strips_info");
			param_frag_factor = cgGetNamedParameter(prog_frag, "depth_factor");
		}
		
		delete programtext;
	}

	update_menu();
}

void destroy()
{
	render->free_mesh();
	cgDestroyProgram(prog_frag);
}

void draw_scene()
{
	if ((viewmode&1)==0)
		return;

	glViewport(0,0,ressize,ressize);
	render->draw_scene();
	glViewport(0,0,render->sizex,render->sizey);
	
	render->sel_tex(depthmap);
	glCopyTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT,0,0,ressize,ressize,0);
}

void draw_rect(pVector& rect,pVector& uv,int copytex=1)
{
	glBegin(GL_QUADS); 
	glTexCoord2f(uv.x,uv.y);
	glVertex2f(rect.x,rect.y);
	glTexCoord2f(uv.x,uv.w);
	glVertex2f(rect.x,rect.w);
	glTexCoord2f(uv.z,uv.w);
	glVertex2f(rect.z,rect.w);
	glTexCoord2f(uv.z,uv.y);
	glVertex2f(rect.z,rect.y);
	glEnd();

	if (copytex)
	{
		int irect[4];
		irect[0]=(int)(rect.x+0.5f)-2;
		irect[1]=(int)(rect.y+0.5f);
		irect[2]=(int)(rect.z+0.5f)+2;
		irect[3]=(int)(rect.w+0.5f);
		if (irect[0]<0) irect[0]=0;
		if (irect[2]>ressize) irect[2]=ressize;
		glCopyTexSubImage2D(GL_TEXTURE_2D,0,
			irect[0],irect[1],irect[0],irect[1],
			irect[2]-irect[0],irect[3]-irect[1]);
	}
}

void draw_info()
{
	int i,j=strips[selstrip]+1,k;
	float f=1.0f/j;
	float size=ressize*f;
	float ss=1.0f/substrips;
	pVector uv(0,0,0,1),rect(0,0,0,(float)ressize);

	glColor4f(1,1,1,1);
	glDisable(GL_BLEND);
	glDisable(GL_CULL_FACE);
	glReadBuffer(GL_BACK);
	glDisable(GL_DEPTH_TEST);

	if (viewmode&2)
	{
		render->sel_tex(tilemap);
		rect.x=0;
		rect.z=size+1;
		uv.x=0;
		uv.z=1;
		uv.w=(float)strips[selstrip];
		draw_rect(rect,uv,0);
		uv.w=1;

		if (param_frag_depthmap)
		if (depthmap!=-1)
		{
			cgGLSetTextureParameter(param_frag_depthmap,render->picid[depthmap]);
			cgGLEnableTextureParameter(param_frag_depthmap);
		}
		else
			cgGLDisableTextureParameter(param_frag_depthmap);

		if (param_frag_resmap)
		if (resmap!=-1)
		{
			cgGLSetTextureParameter(param_frag_resmap,render->picid[resmap]);
			cgGLEnableTextureParameter(param_frag_resmap);
		}
		else
			cgGLDisableTextureParameter(param_frag_resmap);

		render->sel_tex(resmap);
		glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGB,0,0,ressize,ressize,0);

		if (param_frag_info)
			cgGLSetParameter2f(param_frag_info, 1.0f/strips[selstrip], f);
		if (param_frag_factor)
			cgGLSetParameter1f(param_frag_factor, depthinvert?factors[selfactor]:-factors[selfactor]);

		cgGLBindProgram(prog_frag);
		cgGLEnableProfile(frag_profile);

		for( i=1;i<j;i++ )
		for( k=0;k<substrips;k++ )
		{
			rect.x=i*size+size*k*ss;
			rect.z=rect.x+size*ss;
			uv.x=i*f+f*k*ss;
			uv.z=uv.x+f*ss;
			
			draw_rect(rect,uv);
		}

		if (param_frag_depthmap)
			cgGLDisableTextureParameter(param_frag_depthmap);
		if (param_frag_resmap)
			cgGLDisableTextureParameter(param_frag_resmap);

		cgGLDisableProfile(frag_profile);

		render->sel_tex(resmap);
	}
	else
		render->sel_tex(depthmap);

	rect.x=0;
	rect.z=(float)render->sizex;
	rect.w=(float)render->sizey;
	uv.x=0;
	uv.z=1;

	glBegin(GL_QUADS); 
	glTexCoord2f(uv.x,uv.y);
	glVertex2f(rect.x,rect.y);
	glTexCoord2f(uv.x,uv.w);
	glVertex2f(rect.x,rect.w);
	glTexCoord2f(uv.z,uv.w);
	glVertex2f(rect.z,rect.w);
	glTexCoord2f(uv.z,uv.y);
	glVertex2f(rect.z,rect.y);
	glEnd();

	render->sel_tex(-1);
}

void animate()
{
}

void key_pressed(int key,int flags)
{
	if (key=='.' || key=='>')
		if (selfactor<5)
			selfactor++;
	if (key==',' || key=='<')
		if (selfactor>0)
			selfactor--;
	if (key=='-' || key=='_')
		if (selstrip>0)
			selstrip--;
	if (key=='=' || key=='+')
		if (selstrip<2)
			selstrip++;
	update_menu();
}

int OpenFileDialog(const char *title,const char *filter,const char *ext,const char *initdir,char *filename,int len)
{
	OPENFILENAME ofn;
	memset(&ofn,0,sizeof(OPENFILENAME));
	
	ofn.lStructSize=sizeof(OPENFILENAME);
	ofn.hwndOwner=hWndMain;
	ofn.hInstance=hInstance;
	ofn.lpstrFilter=filter;
	ofn.lpstrDefExt=ext;
	ofn.lpstrInitialDir=initdir;
	ofn.lpstrTitle=title;
	ofn.Flags=OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST;
	ofn.lpstrFile=filename;
	ofn.nMaxFile=len;

	if (GetOpenFileName(&ofn))
		return 1;
	return 0;
}

int SaveFileDialog(const char *title,const char *filter,const char *ext,const char *initdir,char *filename,int len)
{
	OPENFILENAME ofn;
	memset(&ofn,0,sizeof(OPENFILENAME));
	
	ofn.lStructSize=sizeof(OPENFILENAME);
	ofn.hwndOwner=hWndMain;
	ofn.hInstance=hInstance;
	ofn.lpstrFilter=filter;
	ofn.lpstrDefExt=ext;
	ofn.lpstrInitialDir=initdir;
	ofn.lpstrTitle=title;
	ofn.Flags=OFN_OVERWRITEPROMPT|OFN_PATHMUSTEXIST;
	ofn.lpstrFile=filename;
	ofn.nMaxFile=len;

	if (GetSaveFileName(&ofn))
		return 1;
	return 0;
}

void update_menu()
{
	if (render==0) return;

	HMENU menu=GetMenu(hWndMain);

	CheckMenuItem(menu,ID_VIEW_FULLSCREEN,render->fullscreen==1?MF_CHECKED:MF_UNCHECKED);

	CheckMenuItem(menu,ID_VIEW_GENERATESTEREOGRAM,(viewmode&2)?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_DEPTHFROM3DMESH,(viewmode&1)?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_INVERTDEPTH,depthinvert?MF_CHECKED:MF_UNCHECKED);

	CheckMenuItem(menu,ID_VIEW_NUMBERORSTRIPS_8,selstrip==0?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_NUMBERORSTRIPS_12,selstrip==1?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_NUMBERORSTRIPS_16,selstrip==2?MF_CHECKED:MF_UNCHECKED);

	CheckMenuItem(menu,ID_VIEW_DEPTHFACTOR_010,selfactor==0?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_DEPTHFACTOR_025,selfactor==1?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_DEPTHFACTOR_050,selfactor==2?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_DEPTHFACTOR_075,selfactor==3?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_DEPTHFACTOR_100,selfactor==4?MF_CHECKED:MF_UNCHECKED);
	CheckMenuItem(menu,ID_VIEW_DEPTHFACTOR_200,selfactor==5?MF_CHECKED:MF_UNCHECKED);
	
	CheckMenuItem(menu,ID_VIEW_TEXTUREFILTERING,render->texfilter?MF_CHECKED:MF_UNCHECKED);

	for( int i=0;i<render->profile.num;i++ )
		CheckMenuItem(menu,ID_RENDER_PROFILE+i,render->profile[i]==render->cur_profile?MF_CHECKED:MF_UNCHECKED);
}

void command(UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (render && LOWORD(wParam)>=ID_RENDER_PROFILE && LOWORD(wParam)<ID_RENDER_PROFILE+render->profile.num)
	{
		render->profile_select(LOWORD(wParam)-ID_RENDER_PROFILE);
		update_menu();
	}
	else
	switch( LOWORD(wParam) )
	{
		case ID_VIEW_FULLSCREEN:
			{
			SetCursor(LoadCursor(0,IDC_WAIT));
			destroy();
			render->destroy();
			render->fullscreen=!render->fullscreen;
			render->create(0);
			init();
			update_menu();
			SetCursor(LoadCursor(0,IDC_ARROW));
			}
			break;

		case ID_FILEOPEN_3DMESH:
			{
			char filename[256]="";
			if (OpenFileDialog("Open 3D Mesh"," All scene files (*.p3d;*.3ds)\0*.p3d;*.3ds\0 Paralelo3D scene (*.p3d)\0*.p3d\0 3DStudio scene files (*.3ds)\0*.3ds\0","*.p3d;*.3ds",render->app_path+"data\\mesh\\",filename,255))
				{
				meshfile=filename;
				destroy();
				init();
				}
			}
			break;

		case ID_FILEOPEN_DEPTHMAP:
			{
			char filename[256]="";
			if (OpenFileDialog("Open Depth Image","All image files (24/32 bits/pixel)\0*.jpg;*.tga\0JPEG image files\0*.jpg\0Targa image files\0*.tga\0\0","*.tga;*.jpg",render->app_path+"data\\depth\\",filename,255))
				{
				depthfile=filename;
				destroy();
				init();
				}
			}
			break;

		case ID_FILEOPEN_TILEMAP:
			{
			char filename[256]="";
			if (OpenFileDialog("Open Tile Image","All image files (24/32 bits/pixel)\0*.jpg;*.tga\0JPEG image files\0*.jpg\0Targa image files\0*.tga\0\0","*.tga;*.jpg",render->app_path+"data\\tile\\",filename,255))
				{
				tilefile=filename;
				tilemap=render->load_tex(filename);
				}
			}
			break;
		
		case ID_VIEW_GENERATESTEREOGRAM:
			viewmode^=2;
			update_menu();
			break;

		case ID_VIEW_DEPTHFROM3DMESH:
			viewmode^=1;
			destroy();
			init();
			break;

		case ID_VIEW_INVERTDEPTH: 
			depthinvert=!depthinvert;
			update_menu();
			break;

		case ID_VIEW_TEXTUREFILTERING:
			render->texfilter=!render->texfilter;
			render->update_texflags();
			update_menu();
			break;

		case ID_VIEW_DEPTHFACTOR_010:
		case ID_VIEW_DEPTHFACTOR_025:
		case ID_VIEW_DEPTHFACTOR_050:
		case ID_VIEW_DEPTHFACTOR_075:
		case ID_VIEW_DEPTHFACTOR_100:
		case ID_VIEW_DEPTHFACTOR_200:
			selfactor=LOWORD(wParam)-ID_VIEW_DEPTHFACTOR_010;
			update_menu();
			break;

		case ID_VIEW_NUMBERORSTRIPS_8:
		case ID_VIEW_NUMBERORSTRIPS_12:
		case ID_VIEW_NUMBERORSTRIPS_16:
			selstrip=LOWORD(wParam)-ID_VIEW_NUMBERORSTRIPS_8;
			update_menu();
			break;

		case ID_EXIT:
			render->free_mesh();
			DestroyWindow(hWndMain);
			break;

		case ID_ABOUT:
			MessageBox(
				hWndMain,
				"pStereogram\n\nFabio Policarpo\nfabio@paralelo.com.br",
				appfile,
				MB_ICONINFORMATION|MB_OK);
			break;

		case ID_FILE_RENDERINFO:
			{
				pString str,buf;

				buf="Render Information\n\n";

				str.format("Color bits: %i\n",render->colorbits);
				buf+=str;
				str.format("Depth bits: %i\n",render->depthbits);
				buf+=str;
				str.format("Stencil bits: %i\n",render->stencilbits);
				buf+=str;
				str.format("Num texture units: %i\n",render->maxtextureunits);
				buf+=str;
				str.format("Max texture2D size: %i\n",render->maxtex2dsize);
				buf+=str;
				str.format("Max texture3D size: %i\n",render->maxtex3dsize);
				buf+=str;
				str.format("Two side stencil: %s\n",EXT_stencil_two_side?"YES":"NO");
				buf+=str;
				buf+="\n";
				buf+="Profiles:\n";
				for( int i=0;i<render->profile.num;i++ )
				{
					render->profile[i]->print(str);
					buf+=str;
					buf+="\n";
				}
				
				MessageBox(hWndMain,(const char *)buf,
						appfile,MB_ICONINFORMATION|MB_OK);
			}
			break;
	}
}
