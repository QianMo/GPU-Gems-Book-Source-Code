#include <windows.h>
#include <GL/gl.h>
#include "text.h"
#include "texmanager.h"



texmanager::texmanager()
{
allocatedmemory=0;
}

int texmanager::load(char *name)
/*-------------------------------------------------------------------------------*/
/* loads a texture manager from a description file                               */
/* file format is:                                                               */
/* numtextures [NUMTEXTURES]                                                     */
/* TEXTURE [POSITION] [FILENAME] OPAQUE for regular opaque textures              */
/* TEXTURE [POSITION] [FILENAME] BILLBOARD [KEYCOLOR] for billboards             */
/* TEXTURE [POSITION] [FILENAME] TRANSLUCENT LUMINOSITY for rgb-based            */
/* TEXTURE [POSITION] [FILENAME] TRANSLUCENT FIXED [ALPHA]                       */
/* TEXTURE [POSITION] [FILENAME] TRANSLUCENT LINEAR [COLOR]                      */
/*-------------------------------------------------------------------------------*/
{
text t(name);
int i;

// numtextures
numtextures=t.countword("#TEXTURE");
translationtable=new unsigned int[numtextures];
type=new int[numtextures];
glGenTextures(numtextures,translationtable);
texdata=new texture[numtextures];
// load each texture object
for (i=0;i<numtextures;i++)
	{
	t.seek("#TEXTURE");
	char *name=t.getword();
	int newchunksize=texdata[i].loadtga(name,translationtable[i]);
	char *cl=t.getword();
	type[i]=TEXTURE_OPAQUE;
	if (!strcmp(cl,"OPACITY")) type[i]=TEXTURE_OPACITY;
	if (!strcmp(cl,"BLEND")) type[i]=TEXTURE_BLEND;
	if (!strcmp(cl,"ADDITIVEBLEND")) type[i]=TEXTURE_BLEND_ADD;
	delete cl;
	if (newchunksize==-1) 
		{
		return -1;
		}
	else allocatedmemory+=newchunksize;
	delete name;
	}
return allocatedmemory;
}


void texmanager::usetexture(int texid)
{
glBindTexture(GL_TEXTURE_2D,translationtable[texid]);
}


void texmanager::removetexture(int texid)
{
glDeleteTextures(1,&(translationtable[texid]));
}



texmanager::~texmanager()
{
glDeleteTextures(numtextures,translationtable);
}