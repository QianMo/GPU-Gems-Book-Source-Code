/*----------------------------------------------------------------------
|
| $Id: GLSLShader.hh,v 1.1 2004/11/08 11:05:46 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef GLSL_SHADER_INCLUDED
#define GLSL_SHADER_INCLUDED

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "defines.h"

#include <GL/glew.h>
#define GL_GLEXT_PROTOTYPES 1
#define GLAPI
#include <GL/glext.h>
#include <GL/glu.h>

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

class GLSLShader 
{
public:
    GLSLShader();
    ~GLSLShader();

    bool loadFromFile(const char* vsfile, const char* fsfile);
    bool loadFromString(const char* vshader, const char* fshader);

    void bind() const;
    void release() const;
    GLuint destroy();
    GLint getUniformLocation(GLcharARB *name);

private:
    // no copying
    GLSLShader& operator= (const GLSLShader& rhs);
    GLSLShader(const GLSLShader& shader);

    void printInfoLog(GLhandleARB obj);

    GLhandleARB program_;
    GLhandleARB vs_,fs_;
    GLint vcompiled_,fcompiled_,linked_;
};

		
#endif // GLSL_SHADER_INCLUDED
/*----------------------------------------------------------------------
|
| $Log: GLSLShader.hh,v $
| Revision 1.1  2004/11/08 11:05:46  DOMAIN-I15+prkipfer
| introduced GLSL shader and special texture handling classes
|
|
+---------------------------------------------------------------------*/
