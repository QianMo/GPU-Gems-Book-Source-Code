// a texture cache
// DSC 

#ifndef _TEXMANAGER_INC
#define _TEXMANAGER_INC

//#include "defines.h"
#include "point.h"
#include "texture.h"
//#include "settings.h"


#define TEXTURE_OPAQUE 0
#define TEXTURE_OPACITY 1
#define TEXTURE_BLEND 2
#define TEXTURE_BLEND_ADD 3

class texmanager
        {
		public:
			int numtextures;
			int allocatedmemory;
			texture *texdata;
			unsigned int *translationtable;
			int *available;
			int *type;
			
			texmanager();
			int load(char *);
			void usetexture(int);
			void removetexture(int);
			~texmanager();
        };


#endif

