#ifndef NV_STREAMS_H
#define NV_STREAMS_H
/**
 * \file nv_streams.h
 *
 * Basic (binary) input and output streams and their inserters/extractors.
 *
 * The streams are defined using abstract base classes. The inserters and
 * extrators can then be implemented against the stream interfaces.
 *
 * Copyright (C) 2003 NVIDIA Corporation
 * 
 * This file is provided without support, instruction, or implied 
 * warranty of any kind.  NVIDIA makes no guarantee of its fitness
 * for a particular purpose and is not liable under any circumstances
 * for any damages or loss whatsoever arising from the use or 
 * inability to use this file or items derived from it.
 *
 */


//
// Includes
//

#include <nv_nvb/nv_nvbdecl.h>



//
// Forward declarations
//

class nv_attribute;

struct vec2;
struct vec3;
struct vec4;

struct mat3;
struct mat4;

struct quat;

/**
 * Abstract input stream.
 *
 * All this interface describes is, that it is possible to read bytes
 * from an input stream.
 *
 */
class DECLSPEC_NV_NVB nv_input_stream
{
public:
    //
    // Construction and destruction
    //

            /// Virtual destructor.
            virtual
   ~nv_input_stream()
            {
                ; // empty
            }

    //
    // Public methods
    //

            /**
             * Read data.
             *
             * Reads a number of bytes from a given buffer. The buffer
             * must already be allocated and of sufficiently sized to
             * accomodate the data.
             *
             * \param pData Pointer to the buffer where the data read are
             *      stored.
             * \param nBytes The number of bytes to be read.
             *
             * \return The number of bytes actually read.
             *
             */
            virtual
            int
    read(void * pData, int nBytes) = 0;
};


/**
 * Abstract output stream.
 *
 * All this interface describes it, that it is possible to write bytes
 * to an output stream.
 *
 */
class DECLSPEC_NV_NVB nv_output_stream
{
public:
    //
    // Construction and destruction
    //

            /// Virtual destructor.
            virtual
   ~nv_output_stream()
            {
                ; // empty
            }


    //
    // Public methods
    //

            /**
             * Write data.
             *
             * Writes a number of bytes. The number of bytes to be written must
             * not be greater than the buffer's size.
             *
             * \param pData Pointer to the data to be written.
             * \param nBytes Number of bytes to be written.
             *
             * \returns The number of bytes actually written.
             *
             */
            virtual
            int
    write(const void * pData, int nBytes) = 0;
};

/** 
 * Abstract input-output stream.
 *
 * This interface is a combination of the input- and the output stream's interfaces.
 *
 */
class DECLSPEC_NV_NVB nv_input_output_stream: virtual public nv_input_stream, virtual public nv_output_stream
{
public:
    //
    // Construction and desctruction
    //

            /// Virtual destructor.
            virtual
   ~nv_input_output_stream()
            {
                ; // empty
            }

    //
    // Public methods
    //

            /**
             * Read data.
             *
             * Reads a number of bytes from a given buffer. The buffer
             * must already be allocated and of sufficiently sized to
             * accomodate the data.
             *
             * \param pData Pointer to the buffer where the data read are
             *      stored.
             * \param nBytes The number of bytes to be read.
             *
             * \return The number of bytes actually read.
             *
             */
            virtual
            int
    read(void * pData, int nBytes) = 0;

            /**
             * Write data.
             *
             * Writes a number of bytes. The number of bytes to be written must
             * not be greater than the buffer's size.
             *
             * \param pData Pointer to the data to be written.
             * \param nBytes Number of bytes to be written.
             *
             * \returns The number of bytes actually written.
             *
             */
            virtual
            int
    write(const void * pData, int nBytes) = 0;
};


//
// Inserting functions
//

        /// Write a boolean value.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, bool bBool);

        /// Write an unsigned char.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, unsigned char nChar);

        /// Write a char.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, char nChar);

        /// Write an unsigned short.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, unsigned short nShort);

        /// Write a short.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOurputStream, short nShort);

        /// Write an unsigned int.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, unsigned int nInt);

        /// Write an int.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, int nInt);

        /// Write a float.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, float nFloat);

        /// Write a double.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, double nDouble);

        /// Write a vec2
        DECLSPEC_NV_NVB
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const vec2 & rVector);
        
        /// Write a vec3
        DECLSPEC_NV_NVB
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const vec3 & rVector);
        
        /// Write a vec4
        DECLSPEC_NV_NVB
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const vec4 & rVector);

        /// Write a mat3
        DECLSPEC_NV_NVB
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const mat3 & rMatrix);

        /// Write a mat4
        DECLSPEC_NV_NVB
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const mat4 & rMatrix);

        /// Write a quaternion.
        DECLSPEC_NV_NVB
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const quat & rQuaternion);


        /// Write an nv_attribute
        DECLSPEC_NV_NVB
        nv_output_stream & 
operator << (nv_output_stream & rOutputStream, const nv_attribute & oAttribute);



//
// Extracting functions
//

        /// Read a boolean.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, bool & bBool);

        /// Read an unsigned char.
        DECLSPEC_NV_NVB 
        nv_input_stream & 
operator >> (nv_input_stream & rInputStream, unsigned char & nChar);

        /// Read a char.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, char & nChar);

        /// Read an unsigned short.
        DECLSPEC_NV_NVB 
        nv_input_stream & 
operator >> (nv_input_stream & rInputStream, unsigned short & nShort);

        /// Read a short.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, short & nShort);

        /// Read an unsigned int.
        DECLSPEC_NV_NVB 
        nv_input_stream & 
operator >> (nv_input_stream & rInputStream, unsigned int & nInt);

        /// Read an int.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, int & nInt);

        /// Read a float.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, float & nFloat);

        /// Read a double.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, double & nDouble);

        /// Read a vec2
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, vec2 & rVector);

        /// Read a vec3
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, vec3 & rVector);

        /// Read a vec4
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, vec4 & rVector);

        /// Read a mat3
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, mat3 & rMatrix);

        /// Read a mat4
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, mat4 & rMatrix);

        /// Read a quaternion
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, quat & rMatrix);

        /// Read an nv_attribute
        DECLSPEC_NV_NVB
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_attribute & rAttribute);


#endif // NV_STREAMS_H
