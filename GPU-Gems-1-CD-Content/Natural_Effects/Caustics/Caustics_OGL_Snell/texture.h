#ifndef _TEXTURE_INC
#define _TEXTURE_INC


class texture
        {
		public:

			char *name;
			unsigned int ident;
			int sizex,sizey;
			int depth;				// in bytes per pixel
			bool alpha;
			
			texture();
			int loadtex(char *,int);
			int loadtga(char *,int);
			void use();
			~texture();
        };


#endif
