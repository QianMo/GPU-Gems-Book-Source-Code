/**
  @file Light.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)

*/

#include "Light.h"

Light::Light()
{
}

Light::Light(
            const Vector4&      position,
            const Color3&       color, 
            bool                areaLight,
            const Vector3&      attenuation)
{
    m_position = position;
    m_color = color;
    m_areaLight = areaLight;
    m_attenuation = attenuation;
}


bool Light::staticIsDirectionalLight(
        const Vector4&                      light)
{
    return (light[3] == 0.0);
}


float Light::staticCalculateIntensity(
            float               distance,
            const Vector3&      attenuation)
{
    // definition GL has for light intensity
    // http://shiva.missouri.edu:88/SGI_Developer/OpenGL_PG/8107
    return 1.0 / (attenuation[0] + attenuation[1] * distance +
            attenuation[2] * distance * distance);
}


float Light::staticCalculateRadius(
            float               testIncrement,
            const Vector3&      attenuation)
{
    float testRadius = testIncrement;
    static const float minIntensity = 10.0 / 256.0;

    // make sure the parameters are reasonable and that
    // the algorithm will terminate
    if ((attenuation[0] < 0.0) ||
        (attenuation[1] < 0.0) ||
        (attenuation[2] < 0.0) ||
        ((attenuation[0] < (1.0 / minIntensity)) &&
             (attenuation[1] == 0.0) && (attenuation[2] == 0.0))) {

        return -1.0;
    }

    while (staticCalculateIntensity(testRadius, attenuation) > minIntensity) {
        testRadius += testIncrement;
    }

    return testRadius;
}


void Light::calculateRadius(
            float               testIncrement)
{
    if (staticIsDirectionalLight(m_position)) {
        m_radius = -1.0;
    } else {
        m_radius = staticCalculateRadius(testIncrement, m_attenuation);
    }
}


