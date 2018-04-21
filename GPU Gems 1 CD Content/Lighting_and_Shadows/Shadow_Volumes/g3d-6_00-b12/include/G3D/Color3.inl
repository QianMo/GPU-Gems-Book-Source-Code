/**
 @file Color3.inl

 Color functions

 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite Portions written by Laura Wollstadt, graphics3d.com
 @cite Portions based on Dave Eberly's Magic Software Library at http://www.magic-software.com

 @created 2001-06-02
 @edited  2004-01-09

 Copyright 2000-2004, Morgan McGuire.
 All rights reserved.
*/

namespace G3D {

//----------------------------------------------------------------------------
inline Color3::Color3 () {
}

//----------------------------------------------------------------------------

inline Color3::Color3(const float fX, const float fY, const float fZ) {
    r = fX;
    g = fY;
    b = fZ;
}

//----------------------------------------------------------------------------
inline Color3::Color3(const float afCoordinate[3]) {
    r = afCoordinate[0];
    g = afCoordinate[1];
    b = afCoordinate[2];
}

//----------------------------------------------------------------------------
inline Color3::Color3 (const Color3& rkVector) {
    r = rkVector.r;
    g = rkVector.g;
    b = rkVector.b;
}

//----------------------------------------------------------------------------
inline float& Color3::operator[] (int i) const {
    return ((float*)this)[i];
}

//----------------------------------------------------------------------------
inline Color3::operator float* () {
    return (float*)this;
}

inline Color3::operator const float* () const {
    return (float*)this;
}

//----------------------------------------------------------------------------

inline bool Color3::fuzzyEq(const Color3& other) const {
    return G3D::fuzzyEq((*this - other).squaredLength(), 0);
}

//----------------------------------------------------------------------------

inline bool Color3::fuzzyNe(const Color3& other) const {
    return G3D::fuzzyNe((*this - other).squaredLength(), 0);
}


//----------------------------------------------------------------------------
inline Color3& Color3::operator= (const Color3& rkVector) {
    r = rkVector.r;
    g = rkVector.g;
    b = rkVector.b;
    return *this;
}

//----------------------------------------------------------------------------
inline bool Color3::operator== (const Color3& rkVector) const {
    return ( r == rkVector.r && g == rkVector.g && b == rkVector.b );
}

//----------------------------------------------------------------------------
inline bool Color3::operator!= (const Color3& rkVector) const {
    return ( r != rkVector.r || g != rkVector.g || b != rkVector.b );
}

//----------------------------------------------------------------------------
inline Color3 Color3::operator+ (const Color3& rkVector) const {
    return Color3(r + rkVector.r, g + rkVector.g, b + rkVector.b);
}

//----------------------------------------------------------------------------
inline Color3 Color3::operator- (const Color3& rkVector) const {
    return Color3(r -rkVector.r, g - rkVector.g, b - rkVector.b);
}

//----------------------------------------------------------------------------
inline Color3 Color3::operator* (float fScalar) const {
    return Color3(fScalar*r, fScalar*g, fScalar*b);
}

//----------------------------------------------------------------------------
inline Color3 Color3::operator* (const Color3& rkVector) const {
    return Color3(r * rkVector.r, g  * rkVector.g, b * rkVector.b);
}

//----------------------------------------------------------------------------
inline Color3 Color3::operator- () const {
    return Color3( -r, -g, -b);
}

//----------------------------------------------------------------------------
inline Color3 operator* (float fScalar, const Color3& rkVector) {
    return Color3(fScalar*rkVector.r, fScalar*rkVector.g,
                  fScalar*rkVector.b);
}

//----------------------------------------------------------------------------
inline Color3& Color3::operator+= (const Color3& rkVector) {
    r += rkVector.r;
    g += rkVector.g;
    b += rkVector.b;
    return *this;
}

//----------------------------------------------------------------------------
inline Color3& Color3::operator-= (const Color3& rkVector) {
    r -= rkVector.r;
    g -= rkVector.g;
    b -= rkVector.b;
    return *this;
}

//----------------------------------------------------------------------------
inline Color3& Color3::operator*= (float fScalar) {
    r *= fScalar;
    g *= fScalar;
    b *= fScalar;
    return *this;
}

//----------------------------------------------------------------------------
inline Color3& Color3::operator*= (const Color3& rkVector) {
    r *= rkVector.r;
    g *= rkVector.g;
    b *= rkVector.b;
    return *this;
}
//----------------------------------------------------------------------------
inline float Color3::squaredLength () const {
    return r*r + g*g + b*b;
}

//----------------------------------------------------------------------------
inline float Color3::length () const {
    return sqrt(r*r + g*g + b*b);
}

//----------------------------------------------------------------------------
inline Color3 Color3::direction () const {
    float lenSquared = r * r + g * g + b * b;

    if (lenSquared != 1.0) {
        return *this / sqrt(lenSquared);
    } else {
        return *this;
    }
}

//----------------------------------------------------------------------------
inline float Color3::dot (const Color3& rkVector) const {
    return r*rkVector.r + g*rkVector.g + b*rkVector.b;
}

//----------------------------------------------------------------------------
inline Color3 Color3::cross (const Color3& rkVector) const {
    return Color3(g*rkVector.b - b*rkVector.g, b*rkVector.r - r*rkVector.b,
                  r*rkVector.g - g*rkVector.r);
}

//----------------------------------------------------------------------------
inline Color3 Color3::unitCross (const Color3& rkVector) const {
    Color3 kCross(g*rkVector.b - b*rkVector.g, b*rkVector.r - r*rkVector.b,
                  r*rkVector.g - g*rkVector.r);
    kCross.unitize();
    return kCross;
}

}


