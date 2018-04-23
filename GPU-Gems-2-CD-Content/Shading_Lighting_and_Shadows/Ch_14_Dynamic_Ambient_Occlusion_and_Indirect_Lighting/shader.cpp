//
//  shader.cpp - some utility routines to load shaders that have been precompiled by Cg
//
//      shaders are loaded from the current directory
//

#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <tchar.h>
#include <malloc.h>
#include <GL\gl.h>
#include <GL\glext.h>
#include <stdio.h>

PFNGLLOADPROGRAMNVPROC               glLoadProgramNV;
PFNGLBINDPROGRAMNVPROC               glBindProgramNV;
PFNGLPROGRAMLOCALPARAMETER4FVNVPROC	glProgramLocalParameter4fvNV;

#define GET_PROC_ADDRESS wglGetProcAddress

#define QUERY_EXTENSION_ENTRY_POINT(name, type)               \
    name = (type)GET_PROC_ADDRESS(#name);

static char shader1[65536];


int
init_shader(char *name, int id, int shaderType)
{
    FILE *f;
    int len;
    char filename[4096];

    if (!glBindProgramNV) {
        QUERY_EXTENSION_ENTRY_POINT(glBindProgramNV, PFNGLBINDPROGRAMNVPROC);
        QUERY_EXTENSION_ENTRY_POINT(glLoadProgramNV, PFNGLLOADPROGRAMNVPROC);
		QUERY_EXTENSION_ENTRY_POINT(glProgramLocalParameter4fvNV, PFNGLPROGRAMLOCALPARAMETER4FVNVPROC);

    }
    sprintf(filename, ".\\%s", name);
    f = fopen(filename, "r");
    if (!f) {
        printf("Error loading shader %s\n", filename);
        return 0;
    }
    len = (int) fread(shader1, 1, sizeof(shader1)-1, f);
    shader1[len] = 0;
    glLoadProgramNV(shaderType, id, (int) strlen(shader1),
        (GLubyte *) shader1);
    glBindProgramNV(shaderType, id);
    if (f)
        fclose(f);

    return 1;
}

void
set_shader(int id, int shaderType)
{
    glBindProgramNV(shaderType, id);
}

void
set_shader_parameter(int id, char *name, float *value)
{
	glProgramLocalParameter4fvNV(id, (GLsizei) strlen(name), (unsigned char *) name, value);
}
