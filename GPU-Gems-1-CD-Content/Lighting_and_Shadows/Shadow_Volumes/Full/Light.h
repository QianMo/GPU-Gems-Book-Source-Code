/**
  @file Light.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)

*/

#ifndef LIGHT_H
#define LIGHT_H

#include <G3DAll.h>

class Light
{
public:
    Light();
    Light(
            const Vector4&      position,
            const Color3&       color, 
            bool                areaLight,
            const Vector3&      attenuation);

    virtual ~Light() { }

    /**
     * light positions with a homogenous coordinate exactly equal to 0
     * are considered directional lights
     */
    static bool staticIsDirectionalLight(
            const Vector4&      position);


    /**
     * calculate light intensity according to GL's rules
     */
    static float staticCalculateIntensity(
            float               distance,
            const Vector3&      attenuation);


    /**
     * calculate the effective radius for a point light given
     * the attenuation factor.  Test radii will be taken in
     * testIncrement steps away from each other.
     */
    static float staticCalculateRadius(
            float               testIncrement,
            const Vector3&      attenuation);

    /**
     * calculate the radius for this light
     */
    void calculateRadius(
            float               testIncrement);


    bool                m_on;
    Vector4             m_position;
    Color4              m_color;
    bool                m_areaLight;
    Vector3             m_attenuation;
    float               m_radius;

};

#endif



