/**
  @file BasicCamera.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#include <G3DAll.h>
#include <SDL.h>
#include "BasicCamera.h"
#include "DemoSettings.h"

BasicCamera::BasicCamera()
{
    m_centerX       = 400;
    m_centerY       = 300;

    m_yaw           = -G3D_PI/2;
    m_pitch         = 0;
    m_transformation.translation = G3D::Vector3::ZERO;
    m_maxMoveRate   = 3;
    m_maxTurnRate   = G3D_PI/30;
    m_prevMouseX    = m_centerX;
    m_prevMouseY    = m_centerY;
    
    if (appHasFocus()) {
        SDL_WarpMouse(m_centerX, m_centerY);
    }
}


Vector3 BasicCamera::getViewVector() const
{
    return m_transformation.getLookVector();
}


G3D::CoordinateFrame BasicCamera::getWorldToCamera() const
{
    return m_transformation.inverse();
}


void BasicCamera::get3DViewportCorners(
        double                      nearX,
        double                      nearY,
        double                      nearZ,
        Vector3&                    outUR,
        Vector3&                    outLR,
        Vector3&                    outLL,
        Vector3&                    outUL) const
{
    // compensate for BasicCamera's compensation of different z look vector
    // directions
    double cameraZ = nearZ * CoordinateFrame::zLookDirection;

    // Take to world space
    outUR = m_transformation.pointToWorldSpace(
            Vector3( nearX,  nearY, cameraZ));
    outLR = m_transformation.pointToWorldSpace(
            Vector3( nearX, -nearY, cameraZ));
    outLL = m_transformation.pointToWorldSpace(
            Vector3(-nearX, -nearY, cameraZ));
    outUL = m_transformation.pointToWorldSpace(
            Vector3(-nearX,  nearY, cameraZ));
}


void BasicCamera::updateCamera(
        int                         xDirection,
        int                         zDirection,
        int                         mouseX,
        int                         mouseY)
{
    int newMouseX = mouseX;
    int newMouseY = mouseY;

    if (!appHasFocus()) {
        return;
    }

    m_transformation.translation = m_transformation.translation + 
        zDirection * m_transformation.getLookVector() * m_maxMoveRate;
    m_transformation.translation = m_transformation.translation + 
        xDirection * m_transformation.getRightVector() * m_maxMoveRate;

    if((mouseX < 0) || (mouseX > 10000)) {
        newMouseX = m_centerX;
        newMouseY = m_centerY;
    }

    double dx = (newMouseX - m_prevMouseX)/100;
    double dy = (newMouseY - m_prevMouseY)/100;

    m_prevMouseX = m_centerX;
    m_prevMouseY = m_centerY;
    SDL_WarpMouse(m_centerX, m_centerY);

    if(G3D::abs(dx) > m_maxTurnRate) {
        dx = m_maxTurnRate * G3D::sign(dx);
    }
    if(G3D::abs(dy) > m_maxTurnRate) {
        dy = m_maxTurnRate * G3D::sign(dy);
    }

    m_yaw += dx;
    m_pitch += dy;

    if(m_pitch < -G3D_PI/2) {
        m_pitch = -G3D_PI/2;
    } else if(m_pitch > G3D_PI/2) {
        m_pitch = G3D_PI/2;
    }

    m_transformation.rotation.fromEulerAnglesZYX(0, -m_yaw, -m_pitch);
}


void BasicCamera::orient(
        const Vector3&              eye,
        const Vector3&              look)
{
    m_yaw = atan2(look[0], look[2] * CoordinateFrame::zLookDirection);
    m_pitch = -atan2(look[1], sqrt(look[0] * look[0] + look[2] * look[2]));
    m_transformation.rotation.fromEulerAnglesZYX(0, -m_yaw, -m_pitch);

    m_transformation.translation = eye;
}

