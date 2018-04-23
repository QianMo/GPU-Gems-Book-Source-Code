#ifdef USE_RCSID
static const char RCSid_GLSLShader[] = "$Id: GLSLShader.C,v 1.1 2004/11/08 11:05:46 DOMAIN-I15+prkipfer Exp $";
#endif

/*----------------------------------------------------------------------
|
|
| $Log: GLSLShader.C,v $
| Revision 1.1  2004/11/08 11:05:46  DOMAIN-I15+prkipfer
| introduced GLSL shader and special texture handling classes
|
|
|
+---------------------------------------------------------------------*/

#include "GLSLShader.h"

#include <fstream>
#include <vector>
#include <string>

GLSLShader::GLSLShader() 
    : program_(0)
    , vs_(0)
    , fs_(0)
    , vcompiled_(0)
    , fcompiled_(0)
    , linked_(0) 
{
}

GLSLShader::~GLSLShader()
{
    destroy();
}

bool 
GLSLShader::loadFromFile(const char* vsfile, const char* fsfile)
{
    std::ifstream vfin(vsfile);
    std::ifstream ffin(fsfile);

    if(!vfin) {
        errormsg("Unable to open vertex shader "<<vsfile);
        return false;
    }

    if(!ffin) {
        errormsg("Unable to open fragment shader "<<fsfile);
        return false;
    }

    std::string v_shader_str;
    std::string f_shader_str;
    std::string line;

    // Probably not efficient, but who cares...
    while(std::getline(vfin, line)) {
        v_shader_str += line + "\n";
    }

    // Probably not efficient, but who cares...
    while(std::getline(ffin, line)) {
        f_shader_str += line + "\n";
    }

    bool retVal = loadFromString(v_shader_str.c_str(), f_shader_str.c_str());

    if (retVal) {
        debugmsg("Loaded program ["<<vsfile<<","<<fsfile<<"]");
// 	debugmsg("vertex shader code:"<<std::endl<<v_shader_str.c_str());
// 	debugmsg("fragment shader code:"<<std::endl<<f_shader_str.c_str());
    }

    return retVal;
}

bool 
GLSLShader::loadFromString(const char* vshader, const char* fshader) 
{
    if (program_) destroy();

    assert(!program_);

    vs_ = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    checkGLError("cannot create vertex shader object");
    fs_ = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    checkGLError("cannot create fragment shader object");

    glShaderSourceARB(vs_,1,&vshader,NULL);
    checkGLError("sourcing vertex shader");
    glShaderSourceARB(fs_,1,&fshader,NULL);
    checkGLError("sourcing fragment shader");

    glCompileShaderARB(vs_);
    checkGLError("compiling vertex shader");
    glGetObjectParameterivARB(vs_,GL_OBJECT_COMPILE_STATUS_ARB,&vcompiled_);
    printInfoLog(vs_);

    if (!vcompiled_) {
	errormsg("error compiling vertex shader");
	return false;
    }

    glCompileShaderARB(fs_);
    checkGLError("compiling fragment shader");
    glGetObjectParameterivARB(fs_,GL_OBJECT_COMPILE_STATUS_ARB,&fcompiled_);
    printInfoLog(fs_);

    if (!fcompiled_) {
	errormsg("error compiling fragment shader");
	return false;
    }

    program_ = glCreateProgramObjectARB();
    checkGLError("creating program object");

    glAttachObjectARB(program_,vs_);
    checkGLError("attaching vertex shader to program");
    glAttachObjectARB(program_,fs_);
    checkGLError("attaching fragment shader to program");

    glLinkProgramARB(program_);
    checkGLError("linking program");
    glGetObjectParameterivARB(program_,GL_OBJECT_LINK_STATUS_ARB,&linked_);
    printInfoLog(program_);

    if (!linked_) {
	errormsg("cannot link shaders to program object");
	return false;
    }

    return true;
}

void
GLSLShader::printInfoLog(GLhandleARB obj)
{
    int infologLength = 0;
    int charsWritten = 0;

    checkGLError("before reading info log");

    glGetObjectParameterivARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB, &infologLength);
    checkGLError("reading info log length");

    if (infologLength > 1) {
	std::vector<GLcharARB> infoLog;
	infoLog.resize(infologLength);
	glGetInfoLogARB(obj, infologLength, &charsWritten, &(infoLog[0]));
	infomsg("info log("<<infologLength<<"): "<<&(infoLog[0]));
    }

    checkGLError("after reading info log");
}

GLint
GLSLShader::getUniformLocation(GLcharARB *name) 
{
    GLint loc = glGetUniformLocationARB(program_, name);
    if (loc == -1) errormsg("No such uniform name \""<<name<<"\"");
    checkGLError("get uniform location");
    return loc;
}

void
GLSLShader::bind() const 
{ 
    if (linked_) 
	glUseProgramObjectARB(program_); 
}

void
GLSLShader::release() const 
{ 
    glUseProgramObjectARB(0); 
}

GLuint 
GLSLShader::destroy() 
{
    if (program_) {
	// shaders will be detached and marked for deletion
	// automatically
	glDeleteObjectARB(program_);
	checkGLError("error deleting program object");
	program_ = vs_ = fs_ = 0;
	vcompiled_ = fcompiled_ = linked_ = 0;
    }
    return 1;
}
