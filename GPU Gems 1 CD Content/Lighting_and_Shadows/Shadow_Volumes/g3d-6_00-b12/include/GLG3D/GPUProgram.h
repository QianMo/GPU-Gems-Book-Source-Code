/**
  @file GPUProgram.h

  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2003-04-13
  @edited  2003-11-16
*/

#ifndef GLG3D_GPUPROGRAM_H
#define GLG3D_GPUPROGRAM_H

#include "graphics3D.h"
#include "GLG3D/glheaders.h"

#include "GLG3D/Texture.h"

namespace G3D {

typedef ReferenceCountedPointer<class GPUProgram> GPUProgramRef;

/**
  Base class for VertexProgram and PixelProgram

  If you use GPUProgramRef instead of GPUProgram*, the texture memory will be
  garbage collected.

  If an error is encountered inside a shader in debug mode, that error is printed to the
  debug window (under MSVC) and the programmer can fix the error and reload the
  shader without reloading the program.

  Subclasses of GPUProgram (VertexProgram and PixelProgram) automatically recognize
  the output of NVIDIA's Cg compiler and will load constants from comments of the
  form:

  <CODE>    # const c[N] = V0 V1 V2 V3</CODE>

  Likewise, variable bindings are detected in Cg output. Create an ArgList mapping
  variable names to the actual parameters and provide it when you call
  RenderDevice::setVertexProgram(). Alternately, you can track matrices and load
  variables yourself using the NVIDIA or ARB APIs. 
  
  For example:

  <PRE>
    glTrackMatrixNV(GL_VERTEX_PROGRAM_NV, 0, GL_MODELVIEW_PROJECTION_NV, GL_IDENTITY_NV);
    glTrackMatrixNV(GL_VERTEX_PROGRAM_NV, 4, GL_MODELVIEW, GL_INVERSE_NV);
    glProgramParameter4fvNV(GL_VERTEX_PROGRAM_NV, 8, Vector4(myVariable,0,0,0));
  </PRE>

 */
class GPUProgram : public ReferenceCountedObject {
public:
    /** Internal use only */
    enum Type {
		FLOAT4X4, FLOAT3X3, FLOAT2X2, 
		FLOAT4, FLOAT3, FLOAT2, FLOAT1,
		SAMPLER1D, SAMPLER2D, SAMPLER3D, SAMPLERCUBE, SAMPLERRECT
	};

    /** Internal use only */
    enum Source {VARIABLE, CONSTANT};

    /** Internal use only
	 Converts a string name (e.g. "float4") to a Cg type.
	 Returns false if it can't match the type.
	*/
	static bool CgType(const std::string& s, Type& t);

    /** Internal use only.  For printing error messages. */
	static std::string toString(const Type& t);

private:

    friend class RenderDevice;

    /**
     Manages the constants in a vertex or fragment program produced by Cg, where
     the header has comments of the form:

      # const c[N] = V0 V1 V2 V3

      where N, V0, V1, V2, and V3 are numbers.
     */
    class BindingTable {
    public:
        /**
         A constant, variable, or matrix binding for a vertex program.
         */
        class Binding {
        public:

            /**
             When the slot is UNASSIGNED, the variable
             was declared but never used; it has no slot.
             */
            enum {UNASSIGNED = -1};

            /**
             Variable/constant name
             */
            std::string         name;

            /**
             Which const register receives this value.
             */
            int                 slot;

            Type                type;

            Source              source;

			/** Used by constants only */
            Vector4             vector;
        };

        Array<Binding>          bindingArray;

        static bool symbolMatch(const Token& t, const std::string& s);

        /**
         Returns true if it is able to consume the next symbol, which must match s.
         */
        static bool consumeSymbol(TextInput& ti, const std::string& s);

        /** Called from parse() */
        void parseVariable(TextInput& ti);

        /** Called from parse() */
        void parseConstant(TextInput& ti);

    public:

        /**
         Initializes from a Cg generated assembly program.
         */
        void parse(const std::string& code);

        /**
         Binds constants using the NVIDIA API.
         */
        void nvBind(GLenum target) const;

        /**
         Binds constants using the ARB API.
         */
        void arbBind(GLenum target) const;

    };

public:
     /**
     Argument list for a vertex program.
     See RenderDevice::setVertexProgram
     and RenderDevice::setPixelProgram.
     */
    class ArgList {
    private:
        friend class GPUProgram;

        class Arg {
        public:

            /** Row-major */ 
            Vector4                    vector[4];

			TextureRef				   texture;

            Type                       type;
        };

        Table<std::string, Arg>        argTable;

    public:


		void set(const std::string& var, const TextureRef& val);
        void set(const std::string& var, const CoordinateFrame& val);
        void set(const std::string& var, const Matrix4& val);
        void set(const std::string& var, const Vector4& val);
        void set(const std::string& var, const Vector3& val);
        void set(const std::string& var, const Vector2& val);
        void set(const std::string& var, float          val);
        void clear();

    };

protected:
    /** Formal parameters */
    BindingTable                bindingTable;

    /** e.g. GL_VERTEX_PROGRAM_ARB, GL_FRAGMENT_PROGRAM_ARB */
    GLenum                      unit;

    std::string                 name;
    GLuint                      glProgram;

    std::string                 filename;

    /**
     Which extension set to use.
     */
    enum Extension {ARB, NVIDIA} extension;

    // Abstraction over NVIDIA/ARB extensions
    void genPrograms(int num, unsigned int* id) const;
    void bindProgram(int unit, unsigned int glProgram) const;
    void loadProgram(const std::string& code) const;
    void loadConstant(int slot, const Vector4& value) const;
    void getProgramError(int& pos, const unsigned char*& msg) const;
    void deletePrograms(int num, unsigned int* id) const;
    
protected:

    /**
     Determines which unit and extension API a shader uses from the first line.
     */
    static GLenum getUnitFromCode(const std::string& code, Extension& extension);

    /**
        Called by RenderDevice::setVertexProgram() and RenderDevice::setPixelProgram()
    */
    void setArgs(class RenderDevice*, const ArgList& args);

    GPUProgram(const std::string& name, const std::string& filename);

public:


    ~GPUProgram();


    /** Reload from supplied code or from the original file that
        was specified (handy when debugging shaders) */
    void reload(const std::string& code = "");

    GLuint getOpenGLID() const;

    /**
     Binds this program.
     */
    void bind();

    /**
     Unbinds and disables.
     */
    void disable();
};

}

#endif
