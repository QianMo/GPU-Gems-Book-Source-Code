/**
 @file G3D/Color4.inl

 Color functions

 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite Portions based on Dave Eberly's Magic Software Library at http://www.magic-software.com

 @created 2001-06-25
 @edited  2003-01-25

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
*/

namespace G3D {

inline Color4 operator*(const Color3& c3, const Color4& c4) {
    return Color4(c3.r * c4.r, c3.g * c4.g, c3.b * c4.b, c4.a);
}

//----------------------------------------------------------------------------

inline Color4::Color4 () {
    // For efficiency in construction of large arrays of vectors, the
    // default constructor does not initialize the vector.
}

//----------------------------------------------------------------------------

inline Color4::Color4(const Color3& c3, float a) {
    r = c3.r;
    g = c3.g;
    b = c3.b;
    this->a = a;
}

//----------------------------------------------------------------------------

inline Color4::Color4(
    float       r,
    float       g,
    float       b,
    float       a) :
    r(r), g(g), b(b), a(a) {
}

//----------------------------------------------------------------------------
inline Color4::Color4 (float afCoordinate[4]) {
    r = afCoordinate[0];
    g = afCoordinate[1];
    b = afCoordinate[2];
    a = afCoordinate[3];
}

//----------------------------------------------------------------------------

inline Color4::Color4(
    const Color4&           other) {

    r = other.r;
    g = other.g;
    b = other.b;
    a = other.a;
}

//----------------------------------------------------------------------------

inline float& Color4::operator[] (int i) const {
    return ((float*)this)[i];
}

//----------------------------------------------------------------------------
inline Color4::operator float* () {
    return (float*)this;
}

inline Color4::operator const float* () const {
    return (float*)this;
}

//----------------------------------------------------------------------------

inline bool Color4::fuzzyEq(const Color4& other) const {
    Color4 dif = (*this - other);
    return G3D::fuzzyEq(dif.r * dif.r + dif.g * dif.g + dif.b * dif.b + dif.a * dif.a, 0);
}

//----------------------------------------------------------------------------

inline bool Color4::fuzzyNe(const Color4& other) const {
    Color4 dif = (*this - other);
    return G3D::fuzzyNe(dif.r * dif.r + dif.g * dif.g + dif.b * dif.b + dif.a * dif.a, 0);
}


//----------------------------------------------------------------------------
inline Color4& Color4::operator= (const Color4& rkVector) {
    r = rkVector.r;
    g = rkVector.g;
    b = rkVector.b;
    a = rkVector.a;
    return *this;
}

//----------------------------------------------------------------------------

inline bool Color4::operator== (const Color4& other) const {
    return ( r == other.r && g == other.g && b == other.b && a == other.a);
}

//----------------------------------------------------------------------------

inline bool Color4::operator!= (const Color4& other) const {
    return ( r != other.r || g != other.g || b != other.b || a != other.a);
}

//----------------------------------------------------------------------------
inline Color4 Color4::operator+ (const Color4& other) const {
    return Color4(r + other.r, g + other.g, b + other.b, a + other.a);
}

//----------------------------------------------------------------------------
inline Color4 Color4::operator- (const Color4& rkVector) const {
    return Color4(r -rkVector.r, g - rkVector.g, b - rkVector.b);
}

//----------------------------------------------------------------------------

inline Color4 Color4::operator* (float fScalar) const {
    return Color4(fScalar*r, fScalar*g, fScalar*b, fScalar*a);
}

//----------------------------------------------------------------------------

inline Color4 Color4::operator- () const {
    return Color4(-r, -g, -b, -a);
}

//----------------------------------------------------------------------------

inline Color4 operator* (float fScalar, const Color4& rkVector) {
    return Color4(fScalar*rkVector.r, fScalar*rkVector.g,
                  fScalar*rkVector.b, fScalar*rkVector.a);
}

//----------------------------------------------------------------------------

inline Color4& Color4::operator+= (const Color4& rkVector) {
    r += rkVector.r;
    g += rkVector.g;
    b += rkVector.b;
    a += rkVector.a;
    return *this;
}

//----------------------------------------------------------------------------

inline Color4& Color4::operator-= (const Color4& rkVector) {
    r -= rkVector.r;
    g -= rkVector.g;
    b -= rkVector.b;
    a -= rkVector.a;
    return *this;
}

//----------------------------------------------------------------------------

inline Color4& Color4::operator*= (float fScalar) {
    r *= fScalar;
    g *= fScalar;
    b *= fScalar;
    a *= fScalar;
    return *this;
}


}

