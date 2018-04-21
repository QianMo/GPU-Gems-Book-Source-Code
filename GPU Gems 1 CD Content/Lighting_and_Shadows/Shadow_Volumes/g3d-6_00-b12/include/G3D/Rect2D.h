/**
  @file Rect2D.h
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2003-11-13
  @created 2003-11-25

  Copyright 2000-2003, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_RECT2D_H
#define G3D_RECT2D_H

#include "G3D/Vector2.h"

namespace G3D {

/**
 If you are using this class for pixel rectangles, keep in mind that the last
 pixel you can draw to is at x0() + width() - 1.
 */
class Rect2D {
private:
    Vector2 min, max;

public:

    inline double width() const {
        return max.x - min.x;
    }

    inline double height() const {
        return max.y - min.y;
    }

    inline double x0() const {
        return min.x;
    }

    inline double x1() const {
        return max.x;
    }

    inline double y0() const {
        return min.y;
    }

    inline double y1() const {
        return max.y;
    }

    inline Rect2D() : min(0, 0), max(0, 0) {}

    inline static Rect2D xyxy(double x0, double y0, double x1, double y1) {
        Rect2D r;
        
        r.min.x = G3D::min(x0, x1);
        r.min.y = G3D::min(y0, y1);
        r.max.x = G3D::max(x0, x1);
        r.max.y = G3D::max(y0, y1);

        return r;
    }

    inline static Rect2D xyxy(const Vector2& v0, const Vector2& v1) {
        Rect2D r;

        r.min = v0.min(v1);
        r.max = v0.max(v1);

        return r;
    }

    inline static Rect2D xywh(double x, double y, double w, double h) {
        return xyxy(x, y, x + w, y + h);
    }

    inline bool contains(const Vector2& v) {
        return (v.x >= min.x) && (v.y >= min.y) && (v.x <= max.x) && (v.y <= max.y);
    }

    inline Rect2D operator*(double s) {
        return xyxy(min.x * s, min.y * s, max.x * s, max.y * s);
    }

    inline Rect2D operator/(double s) {
        return xyxy(min * s, max * s);
    }

    inline Rect2D operator+(const Vector2& v) {
        return xyxy(min + v, max + v);
    }

    inline Rect2D operator-(const Vector2& v) {
        return xyxy(min - v, max - v);
    }

    inline bool operator==(const Rect2D& other) {
        return (min == other.min) && (max == other.max);
    }

    inline bool operator!=(const Rect2D& other) {
        return (min != other.min) || (max != other.max);
    }
};

}

#endif
