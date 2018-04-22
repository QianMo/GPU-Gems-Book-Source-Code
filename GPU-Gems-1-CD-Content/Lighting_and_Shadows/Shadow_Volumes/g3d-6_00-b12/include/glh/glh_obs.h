/*
    glh - is a platform-indepenedent C++ OpenGL helper library 


    Copyright (c) 2000 Cass Everitt
	Copyright (c) 2000 NVIDIA Corporation
    All rights reserved.

    Redistribution and use in source and binary forms, with or
	without modification, are permitted provided that the following
	conditions are met:

     * Redistributions of source code must retain the above
	   copyright notice, this list of conditions and the following
	   disclaimer.

     * Redistributions in binary form must reproduce the above
	   copyright notice, this list of conditions and the following
	   disclaimer in the documentation and/or other materials
	   provided with the distribution.

     * The names of contributors to this software may not be used
	   to endorse or promote products derived from this software
	   without specific prior written permission. 

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
	   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
	   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
	   REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
	   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
	   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
	   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
	   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
	   POSSIBILITY OF SUCH DAMAGE. 


    Cass Everitt - cass@r3.nu
*/

// This is a file for simple GL helper classes...

#ifndef GLH_OBS_H
#define GLH_OBS_H

#ifdef _WIN32
# include <windows.h>
#endif

#include "GL/gl.h"
#include "GL/glext.h"

#include "glh_extensions.h"

#include "glh_linear.h"

namespace glh
{
	class display_list
	{
	public:
		display_list() 
			: valid(false) {}
		
		virtual ~display_list()
		{ del(); }
		
		void call_list()
		{ if(valid) glCallList(dlist); }
		
		void new_list(GLenum mode)
		{ if(!valid) gen(); glNewList(dlist, mode); }
		
		void end_list()
		{ glEndList(); }
		
		void del()
		{ if(valid) glDeleteLists(dlist, 1); valid = false; }
		
		bool is_valid() const { return valid; }
		
	private:
		
		void gen() { dlist = glGenLists(1); valid=true; }
		
		bool valid;
		GLuint dlist;
	};
	
	class lazy_build_display_list
	{
	public:
		lazy_build_display_list(void (* builder)() = 0) 
			: valid(false), needs_rebuild(true), build_func(builder) {}
		
		virtual ~lazy_build_display_list()
		{ del(); }

		void set_build_func( void (* builder)())
		{ build_func = builder; }

		void call_list()
		{ 
			if(! valid) gen();
			if(needs_rebuild) rebuild_list();
			glCallList(dlist);
		}
		
		void del()
		{ if(valid) glDeleteLists(dlist, 1); valid = false; needs_rebuild = true;}
		
		bool is_valid() const { return valid; }

		// something changed, so rebuild list before next call_list()
		void rebuild() { needs_rebuild = true; }
		
	private:
		
		void gen() { dlist = glGenLists(1); valid=true; }
		void rebuild_list()
		{
			glNewList(dlist, GL_COMPILE);
			if(build_func) (* build_func)(); // call list building routine
			glEndList();
		}
		
		bool valid;
		bool needs_rebuild;
		GLuint dlist;
		void (* build_func)();
	};
	
	
	class tex_object
	{
	public:
		tex_object(GLenum tgt) 
			: target(tgt), valid(false) {}
		
		virtual ~tex_object()
		{ del(); }
		
		void bind()
		{ if(!valid) gen(); glBindTexture(target, texture); }
		
		// convenience methods

#ifdef MACOS
		void parameter(GLenum pname, int i)
		{ glTexParameteri(target, pname, i); }
#endif

		void parameter(GLenum pname, GLint i)
		{ glTexParameteri(target, pname, i); }

		void parameter(GLenum pname, GLfloat f)
		{ glTexParameterf(target, pname, f); }

		void parameter(GLenum pname, GLint * ip)
		{ glTexParameteriv(target, pname, ip); }

		void parameter(GLenum pname, GLfloat * fp)
		{ glTexParameterfv(target, pname, fp); }

		void enable() { glEnable(target); }
		void disable() { glDisable(target); }


		void del()
		{ if(valid) glDeleteTextures(1, &texture); valid = false; }
		bool is_valid() const { return valid; }
		
		void gen() { glGenTextures(1, &texture); valid=true; }
		
		GLenum target;
		bool valid;
		GLuint texture;
	};
	
	class tex_object_1D : public tex_object
	{ public: tex_object_1D() : tex_object(GL_TEXTURE_1D) {} };
	
	class tex_object_2D : public tex_object
	{ public: tex_object_2D() : tex_object(GL_TEXTURE_2D) {} };

    class tex_object_3D : public tex_object
	{ public: tex_object_3D() : tex_object(GL_TEXTURE_3D) {} };


# ifdef GL_ARB_texture_cube_map
	class tex_object_cube_map : public tex_object
	{ public: tex_object_cube_map() : tex_object(GL_TEXTURE_CUBE_MAP_ARB) {} };
# elif GL_EXT_texture_cube_map
	class tex_object_cube_map : public tex_object
	{ public: tex_object_cube_map() : tex_object(GL_TEXTURE_CUBE_MAP_EXT) {} };
# endif

#if defined(GL_EXT_texture_rectangle)
	class tex_object_rectangle : public tex_object
	{ public: tex_object_rectangle() : tex_object(GL_TEXTURE_RECTANGLE_EXT) {} };
#elif defined(GL_NV_texture_rectangle)
	class tex_object_rectangle : public tex_object
	{ public: tex_object_rectangle() : tex_object(GL_TEXTURE_RECTANGLE_NV) {} };
#endif


# ifdef GL_NV_vertex_program
	class vertex_program_base
	{
	public:
		vertex_program_base(GLenum tgt) 
			: valid(false), target(tgt) {}
		
		virtual ~vertex_program_base()
		{ del(); }
		
		void bind()
		{ if(!valid) gen(); glBindProgramNV(target, program); }
		
		void unbind()
		{ glBindProgramNV(target, 0); }
		
		void load(GLuint size, const GLubyte * prog_text)
		{
			if(!valid) gen();
			glLoadProgramNV(target, program, size, prog_text);
			GLint errpos;
			glGetIntegerv(GL_PROGRAM_ERROR_POSITION_NV, &errpos);
			if(errpos != -1)
			{
				fprintf(stderr, "program error:\n");
				int bgn = errpos - 10;
				bgn < 0 ? 0 : bgn;
				const char * c = (const char *)(prog_text + bgn);
				for(int i = 0; i < 30; i++)
				{
					if(bgn+i >= int(size-1))
						break;
					fprintf(stderr, "%c", *c++);
				}
				fprintf(stderr, "\n");
			}
		}

		void load(GLuint size, const char * prog_text)
		{
			if(!valid) gen();
			glLoadProgramNV(target, program, size, (const GLubyte *) prog_text);
			GLint errpos;
			glGetIntegerv(GL_PROGRAM_ERROR_POSITION_NV, &errpos);
			if(errpos != -1)
			{
				fprintf(stderr, "program error:\n");
				int bgn = errpos - 10;
				//bgn < 0 ? 0 : bgn;
				const char * c = (const char *)(prog_text + bgn);
				for(int i = 0; i < 30; i++)
				{
					if(bgn+i >= int(size-1))
						break;
					fprintf(stderr, "%c", *c++);
				}
				fprintf(stderr, "\n");
			}
		}

		void load(const char * prog_text)
		{ load(strlen(prog_text), prog_text); }


		void del()
		{ if(valid) glDeleteProgramsNV(1, &program); valid = false; }
		
		bool is_valid() const { return valid; }
		
	private:
		
		void gen() { glGenProgramsNV(1, &program); valid=true; }
		
		bool valid;
		GLenum target;
		GLuint program;
	};

	class vertex_program : public vertex_program_base
	{
	public:
		vertex_program() 
			: vertex_program_base(GL_VERTEX_PROGRAM_NV) {}
	};		

	class vertex_state_program : public vertex_program_base
	{
	public:
		vertex_state_program() 
			: vertex_program_base(GL_VERTEX_STATE_PROGRAM_NV) {}
	};

	class lazy_load_vertex_program : public vertex_program_base
	{
		public:
		lazy_load_vertex_program(void (*vp_loader)() = 0) 
			: vertex_program_base(GL_VERTEX_PROGRAM_NV), needs_load(true), loader(vp_loader) {}

		void bind()
		{
			vertex_program_base::bind();
			if(needs_load && loader)
			{
				(* loader)();
				needs_load = false;
			}
		}

		void reload() { needs_load = true; }

		private:
			bool needs_load;
			void (* loader)();
	};		


#endif

# ifdef GL_ARB_vertex_program
	class arb_vertex_program_base
	{
	public:
		arb_vertex_program_base(GLenum tgt) 
			: valid(false), target(tgt) {}
		
		virtual ~arb_vertex_program_base()
		{ del(); }
		
		void bind()
		{ 
            if(!valid) 
                gen(); 
            glBindProgramARB(target, program); 
        }
		
		void unbind()
		{ glBindProgramARB(target, 0); }
		
		void load(GLuint size, const GLubyte * prog_text)
		{
            load(size, (const char *) prog_text);
		}

		void load(GLuint size, const char * prog_text)
		{
			if(!valid) gen();
            glBindProgramARB(target, program);

            glProgramStringARB(target, 
                               GL_PROGRAM_FORMAT_ASCII_ARB, 
                               size,
                               (const GLubyte *) prog_text);

			GLint errpos;
			glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &errpos);
			if(errpos != -1)
			{
				fprintf(stderr, "program error:\n");
				int bgn = errpos - 10;
				//bgn < 0 ? 0 : bgn;
				const char * c = (const char *)(prog_text + bgn);
				for(int i = 0; i < 30; i++)
				{
					if(bgn+i >= int(size-1))
						break;
					fprintf(stderr, "%c", *c++);
				}
				fprintf(stderr, "\n");
			}
		}

		void load(const char * prog_text)
		{ load(strlen(prog_text), prog_text); }


		void del()
		{ if(valid) glDeleteProgramsARB(1, &program); valid = false; }
		
		bool is_valid() const { return valid; }
		
	private:
		
		void gen() { glGenProgramsARB(1, &program); valid=true; }
		
		bool valid;
		GLenum target;
		GLuint program;
	};

	class arb_vertex_program : public arb_vertex_program_base
	{
	public:
		arb_vertex_program() 
			: arb_vertex_program_base(GL_VERTEX_PROGRAM_NV) {}
	};		

	class lazy_load_arb_vertex_program : public arb_vertex_program_base
	{
		public:
		lazy_load_arb_vertex_program(void (*vp_loader)() = 0) 
			: arb_vertex_program_base(GL_VERTEX_PROGRAM_NV), needs_load(true), loader(vp_loader) {}

		void bind()
		{
			arb_vertex_program_base::bind();
			if(needs_load && loader)
			{
				(* loader)();
				needs_load = false;
			}
		}

		void reload() { needs_load = true; }

		private:
			bool needs_load;
			void (* loader)();
	};		


#endif

# ifdef GL_NV_fragment_program

	class fragment_program
	{
	public:
		fragment_program() 
			: valid(false) {}
		
		virtual ~fragment_program()
		{ del(); }
		
		void bind()
		{ if(!valid) gen(); glBindProgramNV(GL_FRAGMENT_PROGRAM_NV, program); }
		
		void load(GLuint size, const GLubyte * prog_text)
		{
			if(!valid) gen();
			glLoadProgramNV(GL_FRAGMENT_PROGRAM_NV, program, size, prog_text);
			GLint errpos;
			glGetIntegerv(GL_PROGRAM_ERROR_POSITION_NV, &errpos);
			if(errpos != -1)
			{
				fprintf(stderr, "program error:\n");
				int bgn = errpos - 10;
				bgn < 0 ? 0 : bgn;
				const char * c = (const char *)(prog_text + bgn);
				for(int i = 0; i < 30; i++)
				{
					if(bgn+i >= int(size-1))
						break;
					fprintf(stderr, "%c", *c++);
				}
				fprintf(stderr, "\n");
			}
		}

		void load(GLuint size, const char * prog_text)
		{
			if(!valid) gen();
			glLoadProgramNV(GL_FRAGMENT_PROGRAM_NV, program, size, (const GLubyte *) prog_text);
			GLint errpos;
			glGetIntegerv(GL_PROGRAM_ERROR_POSITION_NV, &errpos);
			if(errpos != -1)
			{
				fprintf(stderr, "program error:\n");
				int bgn = errpos - 10;
				bgn < 0 ? 0 : bgn;
				const char * c = (const char *)(prog_text + bgn);
				for(int i = 0; i < 30; i++)
				{
					if(bgn+i >= int(size-1))
						break;
					fprintf(stderr, "%c", *c++);
				}
				fprintf(stderr, "\n");
			}
		}

		void load(const char * prog_text)
		{ load(strlen(prog_text), prog_text); }

		void del()
		{ if(valid) glDeleteProgramsNV(1, &program); valid = false; }
		
		bool is_valid() const { return valid; }
		
	private:
		
		void gen() { glGenProgramsNV(1, &program); valid=true; }
		
		bool valid;
		GLuint program;
	};


#endif


}
#endif
