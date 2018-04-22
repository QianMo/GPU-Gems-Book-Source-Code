/**
 @file BinaryInput.h
 
 @maintainer Morgan McGuire, graphics3d.com
 
 @created 2001-08-09
 @edited  2003-05-25

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
 */

#ifndef G3D_BINARYINPUT_H
#define G3D_BINARYINPUT_H

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
#include "G3D/System.h"

namespace G3D {

/**
 Sequential or random access byte-order independent binary file access.
 Files compressed with zlib and beginning with an unsigned 32-bit int
 size are transparently decompressed when the compressed = true flag is
 specified to the constructor.

 Several classes contain serialize/deserialize and deserializing
 constructor methods.  See Vector3 for the design pattern.
 */
class BinaryInput {
public:

private:
    /**
     is the file big or little endian
     */
    G3DEndian       fileEndian;
    std::string     filename;

    bool            swapBytes;

    /**
     Length of file, in bytes
     */
    int             length;
    uint8*          buffer;

    /**
     Next byte in file
     */
    int             pos;

    /**
     When true, the buffer is freed in the deconstructor.
     */
    bool            freeBuffer;

public:

    /** false, constant to use with the copyMemory option */
    static const bool      NO_COPY;

	/**
	 If the file cannot be opened, a zero length buffer is presented.
	 */
    BinaryInput(
        const std::string&  filename,
        G3DEndian           fileEndian,
        bool                compressed = false);

    /**
     Creates input stream from an in memory source.
     Unless you specify copyMemory = false, the data is copied
     from the pointer, so you may deallocate it as soon as the
     object is constructed.  It is an error to specify copyMemory = false
     and compressed = true.
     */
    BinaryInput(
        const uint8*        data,
        int                 dataLen,
        G3DEndian           dataEndian,
        bool                compressed = false,
        bool                copyMemory = true);


    virtual ~BinaryInput();


    std::string getFilename() const {
        return filename;
    }

    /**
     returns a pointer to the internal memory buffer.
     */
    const uint8* getCArray() const {
        return buffer;
    }

    /**
     Performs bounds checks in debug mode.  [] are relative to
     the start of the file, not the current position.
     Seeks to the new position before reading.
     */
    inline const uint8 operator[](int n) {
        setPosition(n);
        return readUInt8();
    }

    /**
     Returns the length of the file in bytes.
     */
    inline int getLength() const {
        return length;
    }

    inline int size() const {
        return getLength();
    }

    /**
     Returns the current byte position in the file,
     where 0 is the beginning and getLength() - 1 is the end.
     */
    inline int getPosition() const {
        return pos;
    }

    /**
     Sets the position.  Cannot set past length.
     */
    inline void setPosition(int p) {
        debugAssertM(p <= length, "Read past end of file");
        pos = p;
    }

    /**
     Goes back to the beginning of the file.
     */
    inline void reset() {
        setPosition(0);
    }

    inline int8 readInt8() {
        debugAssertM(pos + 1 <= length, "Read past end of file");
        return buffer[pos++];
    }

    inline bool readBool8() {
        return (readInt8() != 0);
    }

    inline uint8 readUInt8() {
        debugAssertM(pos + 1 <= length, "Read past end of file");
        return ((uint8*)buffer)[pos++];
    }

    uint16 readUInt16();

    inline int16 readInt16() {
        uint16 a = readUInt16();
        return *(int16*)&a;
    }

    uint32 readUInt32();

    inline int32 readInt32() {
        uint32 a = readUInt32();
        return *(int32*)&a;
    }

    uint64 readUInt64();

    inline int64 readInt64() {
        uint64 a = readUInt64();
        return *(int64*)&a;
    }

    inline float32 readFloat32() {
        uint32 a = readUInt32();
        return *(float32*)&a;
    }

    inline float64 readFloat64() {
        uint64 a = readUInt64();
        return *(float64*)&a;
    }

    /**
     Returns the data in bytes.
     */
    void readBytes(int n, void* bytes);

    /**
     Reads an n character string.  The string is not
     required to end in NULL in the file but will
     always be a proper std::string when returned.
     */
    std::string readString(int n);

    /**
     Reads until NULL or the end of the file is encountered.
     */
    std::string readString();

    /**
     Reads until NULL or the end of the file is encountered.
     If the string has odd length (including NULL), reads 
     another byte.
     */
    std::string readStringEven();


    std::string readString32();

    Vector4 readVector4();
    Vector3 readVector3();
    Vector2 readVector2();

    Color4 readColor4();
    Color3 readColor3();


    /**
     Skips ahead n bytes.
     */
    inline void skip(int n) {
        debugAssertM(pos + n <= length, "Read past end of file");
        pos += n;
    }

	/**
	 Returns true if the position is not at the end of the file
	 */
	inline bool hasMore() const {
		return pos < length;
	}
};

}

#endif

