/**
  @file GLight.h

  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2003-11-12
  @edited  2003-11-14
*/

#ifndef G3D_GLIGHT_H
#define G3D_GLIGHT_H

#include "G3D/Vector4.h"
#include "G3D/Vector3.h"
#include "G3D/Color4.h"

namespace G3D {

/**
 A light representation that closely follows the OpenGL light format.
 */
class GLight  {
public:
    /** World space position (for a directional light, w = 0 */
    Vector4             position;

    Vector3             spotDirection;

    /** In <B>degrees</B>.  180 = no cutoff (point/dir) >90 = spot light */
    double              spotCutoff;

    /** Constant, linear, quadratic */
    double              attenuation[3];

    Color3              color;

    /** If false, this light is ignored */
    bool                enabled;

    GLight();

    static GLight directional(const Vector3& toLight, const Color3& color);
    static GLight point(const Vector3& pos, const Color3& color, double constAtt = 1, double linAtt = 0, double quadAtt = 0);
    static GLight spot(const Vector3& pos, const Vector3& pointDirection, double cutOffAngleDegrees, const Color3& color, double constAtt = 1, double linAtt = 0, double quadAtt = 0);

    bool operator==(const GLight& other);
    bool operator!=(const GLight& other);
};

} // namespace
#endif

