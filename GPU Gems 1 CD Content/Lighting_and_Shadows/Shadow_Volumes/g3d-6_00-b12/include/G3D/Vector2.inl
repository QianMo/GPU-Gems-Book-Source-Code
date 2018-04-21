/**
 @file Vector2.inl
 
 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite Portions by Laura Wollstadt, graphics3d.com
 
 @cite Portions based on Dave Eberly'x Magic Software Library
 at http://www.magic-software.com
 
 
 @created 2001-06-02
 @edited  2003-05-13
  Copyright 2000-2004, Morgan McGuire.
  All rights reserved.
 */

namespace G3D {

//----------------------------------------------------------------------------

inline unsigned int hashCode(const G3D::Vector2& v) {
     return v.hashCode();
}

//----------------------------------------------------------------------------
inline Vector2::Vector2 () {
    x = y = 0;
}

//----------------------------------------------------------------------------

inline Vector2::Vector2 (float fS, float fT) {
    x = fS;
    y = fT;
}

//----------------------------------------------------------------------------
inline Vector2::Vector2 (float afCoordinate[2]) {
    x = afCoordinate[0];
    y = afCoordinate[1];
}

//----------------------------------------------------------------------------
inline Vector2::Vector2 (const Vector2& rkVector) {
    x = rkVector.x;
    y = rkVector.y;
}

//----------------------------------------------------------------------------
inline float& Vector2::operator[] (int i) const {
    return ((float*)this)[i];
}

//----------------------------------------------------------------------------
inline Vector2::operator float* () {
    return (float*)this;
}

inline Vector2::operator const float* () const {
    return (float*)this;
}

//----------------------------------------------------------------------------
inline Vector2& Vector2::operator= (const Vector2& rkVector) {
    x = rkVector.x;
    y = rkVector.y;
    return *this;
}

//----------------------------------------------------------------------------
inline bool Vector2::operator== (const Vector2& rkVector) const {
    return ( x == rkVector.x && y == rkVector.y);
}

//----------------------------------------------------------------------------
inline bool Vector2::operator!= (const Vector2& rkVector) const {
    return ( x != rkVector.x || y != rkVector.y);
}

//----------------------------------------------------------------------------
inline Vector2 Vector2::operator+ (const Vector2& rkVector) const {
    return Vector2(x + rkVector.x, y + rkVector.y);
}

//----------------------------------------------------------------------------
inline Vector2 Vector2::operator- (const Vector2& rkVector) const {
    return Vector2(x - rkVector.x, y - rkVector.y);
}

//----------------------------------------------------------------------------
inline Vector2 Vector2::operator* (float fScalar) const {
    return Vector2(fScalar*x, fScalar*y);
}

//----------------------------------------------------------------------------

inline Vector2 Vector2::operator- () const {
    return Vector2( -x, -y);
}

//----------------------------------------------------------------------------

inline Vector2 operator* (float fScalar, const Vector2& rkVector) {
    return Vector2(fScalar*rkVector.x, fScalar*rkVector.y);
}

//----------------------------------------------------------------------------

inline Vector2& Vector2::operator+= (const Vector2& rkVector) {
    x += rkVector.x;
    y += rkVector.y;
    return *this;
}

//----------------------------------------------------------------------------

inline Vector2& Vector2::operator-= (const Vector2& rkVector) {
    x -= rkVector.x;
    y -= rkVector.y;
    return *this;
}

//----------------------------------------------------------------------------

inline Vector2& Vector2::operator*= (float fScalar) {
    x *= fScalar;
    y *= fScalar;
    return *this;
}

//----------------------------------------------------------------------------

inline Vector2& Vector2::operator*= (const Vector2& rkVector) {
    x *= rkVector.x;
    y *= rkVector.y;
    return *this;
}

//----------------------------------------------------------------------------

inline Vector2& Vector2::operator/= (const Vector2& rkVector) {
    x /= rkVector.x;
    y /= rkVector.y;
    return *this;
}

//----------------------------------------------------------------------------

inline Vector2 Vector2::operator* (const Vector2& rkVector) const {
    return Vector2(x * rkVector.x, y * rkVector.y);
}

//----------------------------------------------------------------------------

inline Vector2 Vector2::operator/ (const Vector2& rkVector) const {
    return Vector2(x / rkVector.x, y / rkVector.y);
}

//----------------------------------------------------------------------------
inline float Vector2::squaredLength () const {
    return x*x + y*y;
}

//----------------------------------------------------------------------------
inline float Vector2::length () const {
    return sqrt(x*x + y*y);
}

//----------------------------------------------------------------------------
inline Vector2 Vector2::direction () const {
    float lenSquared = x * x + y * y;

    if (lenSquared != 1.0) {
        return *this / sqrt(lenSquared);
    } else {
        return *this;
    }
}

//----------------------------------------------------------------------------

inline float Vector2::dot (const Vector2& rkVector) const {
    return x*rkVector.x + y*rkVector.y;
}

//----------------------------------------------------------------------------

inline Vector2 Vector2::min(const Vector2 &v) const {
    return Vector2(G3D::min(v.x, x), G3D::min(v.y, y));
}

//----------------------------------------------------------------------------

inline Vector2 Vector2::max(const Vector2 &v) const {
    return Vector2(G3D::max(v.x, x), G3D::max(v.y, y));
}

//----------------------------------------------------------------------------

inline bool Vector2::fuzzyEq(const Vector2& other) const {
    return G3D::fuzzyEq((*this - other).squaredLength(), 0);
}

//----------------------------------------------------------------------------

inline bool Vector2::fuzzyNe(const Vector2& other) const {
    return G3D::fuzzyNe((*this - other).squaredLength(), 0);
}

//----------------------------------------------------------------------------

inline bool Vector2::isFinite() const {
    return G3D::isFinite(x) && G3D::isFinite(y);
}

//----------------------------------------------------------------------------

inline bool Vector2::isZero() const {
    return (x == 0.0) && (y == 0.0);
}

//----------------------------------------------------------------------------

inline bool Vector2::isUnit() const {
    return squaredLength() == 1.0;
}

}

