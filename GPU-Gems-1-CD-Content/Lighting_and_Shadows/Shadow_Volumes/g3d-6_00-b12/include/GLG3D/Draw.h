/**
 @file Draw.h
  
 @maintainer Morgan McGuire, matrix@graphics3d.com
 
 @created 2003-10-29
 @edited  2004-01-10
 */

#ifndef G3D_DRAW_H
#define G3D_DRAW_H

#include "G3D/Color3.h"
#include "G3D/Color4.h"
#include "G3D/Vector3.h"
#include "G3D/MeshAlg.h"

namespace G3D {

class RenderDevice;
class Sphere;
class LineSegment;
class Box;
class AABox;
class Line;
class Capsule;

/**
 When rendering translucent bounding objects, depth write is automatically
 disabled.  Render from back to front for proper transparency.
 */
class Draw {
private:

    static const int WIRE_SPHERE_SECTIONS;
    static const int SPHERE_SECTIONS;

    /** Called from wireSphere, wireCapsule */
    static void wireSphereSection(
        const Sphere&               sphere,
        class RenderDevice*         renderDevice,
        const class Color4&         color,
        bool                        top,
        bool                        bottom);


    static void sphereSection(
        const Sphere&       sphere,
        RenderDevice*       renderDevice,
        const Color4&       color,
        bool                top,
        bool                bottom);

    /**
     Returns the scale due to perspective at
     a point for a line.
     */
    static double perspectiveLineThickness(
        RenderDevice*       rd,
        const class Vector3&      pt);

public:

    /**
     Set the solid color or wire color to Color4::CLEAR to
     prevent rendering of surfaces or lines.
     */
    static void box(
        const Box&          box,
        RenderDevice*       rd,
        const Color4&       solidColor = Color4(1,.2,.2,.5),
        const Color4&       wireColor  = Color3::BLACK);

    static void box(
        const AABox&        box,
        RenderDevice*       rd,
        const Color4&       solidColor = Color4(1,.2,.2,.5),
        const Color4&       wireColor  = Color3::BLACK);

    static void sphere(
        const Sphere&       sphere,
        RenderDevice*       rd,
        const Color4&       solidColor = Color4(1, 1, 0, .5),
        const Color4&       wireColor  = Color3::BLACK);

    static void line(
        const Line&         line,
        RenderDevice*       rd,
        const Color4&       color = Color3::BLACK);

    static void lineSegment(
        const LineSegment&  lineSegment,
        RenderDevice*       rd,
        const Color4&       color = Color3::BLACK);

    /**
     Renders per-vertex normals as thin arrows.  The length
     of the normals is scaled inversely to the number of normals
     rendered.
     */
    static void vertexNormals(
        const G3D::MeshAlg::Geometry&    geometry,
        RenderDevice*               renderDevice,
        const Color4&               color = Color3::GREEN * .5,
        double                      scale = 1);

    static void capsule(
       const Capsule&       capsule, 
       RenderDevice*        renderDevice,
       const Color4&        solidColor = Color4(1,0,1,.5),
       const Color4&        wireColor = Color3::BLACK);

    static void ray(
        const class Ray&          ray,
        RenderDevice*       renderDevice,
        const Color4&       color = Color3::ORANGE,
        double              scale = 1);
    
    static void arrow(
        const Vector3&      start,
        const Vector3&      direction,
        RenderDevice*       renderDevice,
        const Color4&       color = Color3::ORANGE,
        double              scale = 1.0);

    static void axes(
        const class CoordinateFrame& cframe,
        RenderDevice*       renderDevice,
        const Color4&       xColor = Color3::RED,
        const Color4&       yColor = Color3::GREEN,
        const Color4&       zColor = Color3::BLUE,
        double              scale = 1.0);

    static void axes(
        RenderDevice*       renderDevice,
        const Color4&       xColor = Color3::RED,
        const Color4&       yColor = Color3::GREEN,
        const Color4&       zColor = Color3::BLUE,
        double              scale = 1.0);

    /**
     This method is slow and is recommended for 2D GUI rendering only.
     It is much faster to make a rectangular texture and draw using that.
     */
    static void fullScreenImage(
        const class GImage& im,
        RenderDevice*       renderDevice);
};

}

#endif
