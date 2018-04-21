#ifndef GLFORMAT_H
#define GLFORMAT_H

#include "glheaders.h"
#include "G3D/g3dmath.h"

/** A macro that maps G3D types to OpenGL formats
    (e.g. glFormat(Vector3) == GL_FLOAT).

    Use DECLARE_GLFORMATOF(MyType, GLType) at top-level to define
    glFormatOf values for your own classes.

    Used by the vertex array infrastructure. */
// This implementation is designed to meet the following constraints:
//   1. Work around the many MSVC++ partial template bugs
//   2. Work for primitive types (e.g. int)
#define glFormatOf(T) (G3D::_internal::_GLFormat<T>::x())

namespace G3D {
namespace _internal {


template<class T> class _GLFormat {
public:
    static GLenum x() {
        return GL_NONE;
    }
};

}
}

/**
 Macro to declare the underlying format (as will be returned by glFormatOf)
 of a type.  For example,

  <PRE>
    DECLARE_GLFORMATOF(Vector4, GL_FLOAT)
  </PRE>

  Use this so you can make vertex arrays of your own classes and not just 
  the standard ones.
 */
#define DECLARE_GLFORMATOF(G3DType, GLType)          \
namespace G3D {                                      \
    namespace _internal {                            \
        template<> class _GLFormat<G3DType> {        \
        public:                                      \
            static GLenum x()  {                     \
                return GLType;                       \
            }                                        \
        };                                           \
    }                                                \
}

DECLARE_GLFORMATOF( Vector2,       GL_FLOAT)
DECLARE_GLFORMATOF( Vector3,       GL_FLOAT)
DECLARE_GLFORMATOF( Vector4,       GL_FLOAT)
DECLARE_GLFORMATOF( Vector3int16,  GL_SHORT)
DECLARE_GLFORMATOF( Vector2int16,  GL_SHORT)
DECLARE_GLFORMATOF( Color3uint8,   GL_UNSIGNED_BYTE)
DECLARE_GLFORMATOF( Color3,        GL_FLOAT)
DECLARE_GLFORMATOF( Color4,        GL_FLOAT)
DECLARE_GLFORMATOF( Color4uint8,   GL_UNSIGNED_BYTE)
DECLARE_GLFORMATOF( uint8,         GL_UNSIGNED_BYTE)
DECLARE_GLFORMATOF( uint16,        GL_UNSIGNED_SHORT)
DECLARE_GLFORMATOF( uint32,        GL_UNSIGNED_INT)
DECLARE_GLFORMATOF( int8,          GL_BYTE)
DECLARE_GLFORMATOF( int16,         GL_SHORT)
DECLARE_GLFORMATOF( int32,         GL_INT)
DECLARE_GLFORMATOF( float,         GL_FLOAT)
DECLARE_GLFORMATOF( double,        GL_DOUBLE)

#endif
