/**
 @file BinaryOutput.h
 
 @maintainer Morgan McGuire, graphics3d.com
 
 @created 2001-08-09
 @edited  2003-05-25

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
 */

#ifndef G3D_BINARYOUTPUT_H
#define G3D_BINARYOUTPUT_H

#include <assert.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include "G3D/Color4.h"
#include "G3D/Color3.h"
#include "G3D/Vector4.h"
#include "G3D/Vector3.h"
#include "G3D/Vector2.h"
#include "G3D/g3dmath.h"
#include "G3D/Array.h"
#include "G3D/debug.h"
#include "G3D/BinaryInput.h"
#include "G3D/System.h"

namespace G3D {

/**
 Sequential or random access byte-order independent binary file access.

 The compress() call can be used to compressed with zlib and preceed it 
 by a little endian unsigned 32-bit int file size prior to
 commiting.
    */
class BinaryOutput {
private:
    // is the file big or little endian
    G3DEndian       fileEndian;
    std::string     filename;

    // True if the file endianess does not match the machine endian
    bool            swapBytes;

    Array<uint8>    buffer;

    // Next byte in file
    int             pos;

    // is this initialized?
    bool            init;

    /**
     Make sure at least bytes can be written, resizing if
     necessary.
     */
    void reserveBytes(int bytes) {
        if (pos + bytes >= buffer.length()) {
            buffer.resize(pos + bytes);
        }
    }

public:

    /**
     You must call setEndian() if you use this (memory) constructor.
     */
    BinaryOutput();

    /**
     Doesn't actually open the file; commit() does that.
     Use "<memory>" as the filename if you're going to commit
     to memory.
     */
    BinaryOutput(
        const std::string&  filename,
        G3DEndian           fileEndian);

    virtual ~BinaryOutput();
    
    /** Compresses the data in the buffer in place, preceeding it with a little-endian uint32.
        Call immediately before compress().*/
    void compress();

    /**
     Returns a pointer to the internal memory buffer.
     */
    const uint8* getCArray() const {
        return buffer.getCArray();
    }

    void setEndian(G3DEndian fileEndian);

    std::string getFilename() const {
        return filename;
    }

    /**
     Write the bytes to disk.  It is ok to call this 
     multiple times; it will just overwrite the previous file.

     Parent directories are created as needed if they do
     not exist.

     <B>Not</B> called from the destructor; you must call
     it yourself.
     */
    void commit();

    /**
     Write the bytes to memory (which must be of
     at least size() bytes).
     */
    void commit(uint8*);

    /**
     Returns the length of the file in bytes.
     */
    inline int getLength() const {
        return buffer.size();
    }

    inline int size() const {
        return buffer.size();
    }

    /**
     Sets the length of the file to n, padding
     with 0's past the current end.  Does not
     change the position of the next byte to be
     written unless n < getLength().
     */
    inline void setLength(int n) {
        if (n < buffer.size()) {
            pos = n;
        }
        buffer.resize(n);
    }

    /**
     Returns the current byte position in the file,
     where 0 is the beginning and getLength() - 1 is the end.
     */
    inline int getPosition() const {
        return pos;
    }

    /**
     Sets the position.  Can set past length, in which case
     the file is padded with zeros up to one byte before the
     next to be written.
     */
    inline void setPosition(int p) {
        if (p > buffer.size()) {
            setLength(p);
        }
        pos = p;
    }

    void writeBytes(
        const void*        b,
        int                 count) {

        reserveBytes(count);
        memcpy(buffer.getCArray() + pos, b, count);
        pos += count;
    }

    /**
     Writes a signed 8-bit integer to the current position.
     */
    inline void writeInt8(int8 i) {
        reserveBytes(1);
        buffer[pos] = *(uint8*)&i;
        pos++;
    }

    inline void writeBool8(bool b) {
        writeInt8(b ? 1 : 0);
    }

    inline void writeUInt8(int8 i) {
        reserveBytes(1);
        buffer[pos] = i;
        pos++;
    }

    void writeUInt16(uint16 u);

    inline void writeInt16(int16 i) {
        writeUInt16(*(uint16*)&i);
    }

    void writeUInt32(uint32 u);

    inline void writeInt32(int32 i) {
        writeUInt32(*(uint32*)&i);
    }

    void writeUInt64(uint64 u);

    inline void writeInt64(int64 i) {
        writeUInt64(*(uint64*)&i);
    }

    inline void writeFloat32(float32 f) {
        writeUInt32(*(uint32*)&f);
    }

    inline void writeFloat64(float64 f) {
        writeUInt64(*(uint64*)&f);
    }

    /**
     Write a string with NULL termination.
     */
    inline void writeString(const std::string& s) {
        writeString(s.c_str());
    }

    void writeString(const char* s);

    /**
     Write a string, ensuring that the total length
     including NULL is even.
     */
    void writeStringEven(const std::string& s) {
        writeStringEven(s.c_str());
    }

    void writeStringEven(const char* s);


    void writeString32(const char* s);

    /**
     Write a string with a 32-bit length field in front
     of it.
     */
    void writeString32(const std::string& s) {
        writeString32(s.c_str());
    }

    void writeVector4(const Vector4& v);

    void writeVector3(const Vector3& v);

    void writeVector2(const Vector2& v);

    void writeColor4(const Color4& v);

    void writeColor3(const Color3& v);

    /**
     Skips ahead n bytes.
     */
    inline void skip(int n) {
        if (pos + n > buffer.size()) {
            setLength(pos + n);
        }
        pos += n;
    }
};

}
#endif

