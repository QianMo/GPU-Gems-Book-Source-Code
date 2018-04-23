// Magic Software, Inc.
// http://www.magic-software.com
// Copyright (c) 2000-2002.  All Rights Reserved
//
// Source code from Magic Software is supplied under the terms of a license
// agreement and may not be copied or disclosed except in accordance with the
// terms of that agreement.  The various license agreements may be found at
// the Magic Software web site.  This file is subject to the license
//
// FREE SOURCE CODE
// http://www.magic-software.com/License/free.pdf

#include "MgcIntr3DBoxBox.h"
using namespace Mgc;

//----------------------------------------------------------------------------
bool Mgc::TestIntersection (const Box3& rkBox0, const Box3& rkBox1)
{
    // convenience variables
    const Vector3* akA = rkBox0.Axes();
    const Vector3* akB = rkBox1.Axes();
    const Real* afEA = rkBox0.Extents();
    const Real* afEB = rkBox1.Extents();

    // compute difference of box centers, D = C1-C0
    Vector3 kD = rkBox1.Center() - rkBox0.Center();

    Real aafC[3][3];     // matrix C = A^T B, c_{ij} = Dot(A_i,B_j)
    Real aafAbsC[3][3];  // |c_{ij}|
    Real afAD[3];        // Dot(A_i,D)
    Real fR0, fR1, fR;   // interval radii and distance between centers
    Real fR01;           // = R0 + R1
    
    // axis C0+t*A0
    aafC[0][0] = akA[0].Dot(akB[0]);
    aafC[0][1] = akA[0].Dot(akB[1]);
    aafC[0][2] = akA[0].Dot(akB[2]);
    afAD[0] = akA[0].Dot(kD);
    aafAbsC[0][0] = Math::FAbs(aafC[0][0]);
    aafAbsC[0][1] = Math::FAbs(aafC[0][1]);
    aafAbsC[0][2] = Math::FAbs(aafC[0][2]);
    fR = Math::FAbs(afAD[0]);
    fR1 = afEB[0]*aafAbsC[0][0]+afEB[1]*aafAbsC[0][1]+afEB[2]*aafAbsC[0][2];
    fR01 = afEA[0] + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A1
    aafC[1][0] = akA[1].Dot(akB[0]);
    aafC[1][1] = akA[1].Dot(akB[1]);
    aafC[1][2] = akA[1].Dot(akB[2]);
    afAD[1] = akA[1].Dot(kD);
    aafAbsC[1][0] = Math::FAbs(aafC[1][0]);
    aafAbsC[1][1] = Math::FAbs(aafC[1][1]);
    aafAbsC[1][2] = Math::FAbs(aafC[1][2]);
    fR = Math::FAbs(afAD[1]);
    fR1 = afEB[0]*aafAbsC[1][0]+afEB[1]*aafAbsC[1][1]+afEB[2]*aafAbsC[1][2];
    fR01 = afEA[1] + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A2
    aafC[2][0] = akA[2].Dot(akB[0]);
    aafC[2][1] = akA[2].Dot(akB[1]);
    aafC[2][2] = akA[2].Dot(akB[2]);
    afAD[2] = akA[2].Dot(kD);
    aafAbsC[2][0] = Math::FAbs(aafC[2][0]);
    aafAbsC[2][1] = Math::FAbs(aafC[2][1]);
    aafAbsC[2][2] = Math::FAbs(aafC[2][2]);
    fR = Math::FAbs(afAD[2]);
    fR1 = afEB[0]*aafAbsC[2][0]+afEB[1]*aafAbsC[2][1]+afEB[2]*aafAbsC[2][2];
    fR01 = afEA[2] + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*B0
    fR = Math::FAbs(akB[0].Dot(kD));
    fR0 = afEA[0]*aafAbsC[0][0]+afEA[1]*aafAbsC[1][0]+afEA[2]*aafAbsC[2][0];
    fR01 = fR0 + afEB[0];
    if ( fR > fR01 )
        return false;

    // axis C0+t*B1
    fR = Math::FAbs(akB[1].Dot(kD));
    fR0 = afEA[0]*aafAbsC[0][1]+afEA[1]*aafAbsC[1][1]+afEA[2]*aafAbsC[2][1];
    fR01 = fR0 + afEB[1];
    if ( fR > fR01 )
        return false;

    // axis C0+t*B2
    fR = Math::FAbs(akB[2].Dot(kD));
    fR0 = afEA[0]*aafAbsC[0][2]+afEA[1]*aafAbsC[1][2]+afEA[2]*aafAbsC[2][2];
    fR01 = fR0 + afEB[2];
    if ( fR > fR01 )
        return false;

    // axis C0+t*A0xB0
    fR = Math::FAbs(afAD[2]*aafC[1][0]-afAD[1]*aafC[2][0]);
    fR0 = afEA[1]*aafAbsC[2][0] + afEA[2]*aafAbsC[1][0];
    fR1 = afEB[1]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][1];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A0xB1
    fR = Math::FAbs(afAD[2]*aafC[1][1]-afAD[1]*aafC[2][1]);
    fR0 = afEA[1]*aafAbsC[2][1] + afEA[2]*aafAbsC[1][1];
    fR1 = afEB[0]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A0xB2
    fR = Math::FAbs(afAD[2]*aafC[1][2]-afAD[1]*aafC[2][2]);
    fR0 = afEA[1]*aafAbsC[2][2] + afEA[2]*aafAbsC[1][2];
    fR1 = afEB[0]*aafAbsC[0][1] + afEB[1]*aafAbsC[0][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A1xB0
    fR = Math::FAbs(afAD[0]*aafC[2][0]-afAD[2]*aafC[0][0]);
    fR0 = afEA[0]*aafAbsC[2][0] + afEA[2]*aafAbsC[0][0];
    fR1 = afEB[1]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][1];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A1xB1
    fR = Math::FAbs(afAD[0]*aafC[2][1]-afAD[2]*aafC[0][1]);
    fR0 = afEA[0]*aafAbsC[2][1] + afEA[2]*aafAbsC[0][1];
    fR1 = afEB[0]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A1xB2
    fR = Math::FAbs(afAD[0]*aafC[2][2]-afAD[2]*aafC[0][2]);
    fR0 = afEA[0]*aafAbsC[2][2] + afEA[2]*aafAbsC[0][2];
    fR1 = afEB[0]*aafAbsC[1][1] + afEB[1]*aafAbsC[1][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A2xB0
    fR = Math::FAbs(afAD[1]*aafC[0][0]-afAD[0]*aafC[1][0]);
    fR0 = afEA[0]*aafAbsC[1][0] + afEA[1]*aafAbsC[0][0];
    fR1 = afEB[1]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][1];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A2xB1
    fR = Math::FAbs(afAD[1]*aafC[0][1]-afAD[0]*aafC[1][1]);
    fR0 = afEA[0]*aafAbsC[1][1] + afEA[1]*aafAbsC[0][1];
    fR1 = afEB[0]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0+t*A2xB2
    fR = Math::FAbs(afAD[1]*aafC[0][2]-afAD[0]*aafC[1][2]);
    fR0 = afEA[0]*aafAbsC[1][2] + afEA[1]*aafAbsC[0][2];
    fR1 = afEB[0]*aafAbsC[2][1] + afEB[1]*aafAbsC[2][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    return true;
}
//----------------------------------------------------------------------------
bool Mgc::TestIntersection (Real fTime, const Box3& rkBox0,
    const Vector3& rkVel0, const Box3& rkBox1, const Vector3& rkVel1)
{
    // convenience variables
    const Vector3* akA = rkBox0.Axes();
    const Vector3* akB = rkBox1.Axes();
    const Real* afEA = rkBox0.Extents();
    const Real* afEB = rkBox1.Extents();

    // Compute relative velocity of box1 with respect to box0 so that box0
    // may as well be stationary.
    Vector3 kW = rkVel1 - rkVel0;

    // Compute difference of box centers at time 0 and time 'fTime'.
    Vector3 kD0 = rkBox1.Center() - rkBox0.Center();
    Vector3 kD1 = kD0 + fTime*kW;

    Real aafC[3][3];     // matrix C = A^T B, c_{ij} = Dot(A_i,B_j)
    Real aafAbsC[3][3];  // |c_{ij}|
    Real afAD0[3];       // Dot(A_i,D0)
    Real afAD1[3];       // Dot(A_i,D1)
    Real fR0, fR1, fR;   // interval radii and distance between centers
    Real fR01;           // = R0 + R1
    
    // axis C0+t*A0
    aafC[0][0] = akA[0].Dot(akB[0]);
    aafC[0][1] = akA[0].Dot(akB[1]);
    aafC[0][2] = akA[0].Dot(akB[2]);
    afAD0[0] = akA[0].Dot(kD0);
    afAD1[0] = akA[0].Dot(kD1);
    aafAbsC[0][0] = Math::FAbs(aafC[0][0]);
    aafAbsC[0][1] = Math::FAbs(aafC[0][1]);
    aafAbsC[0][2] = Math::FAbs(aafC[0][2]);
    fR1 = afEB[0]*aafAbsC[0][0]+afEB[1]*aafAbsC[0][1]+afEB[2]*aafAbsC[0][2];
    fR01 = afEA[0] + fR1;
    if ( afAD0[0] > fR01 )
    {
        if ( afAD1[0] > fR01 )
            return false;
    }
    else if ( afAD0[0] < -fR01 )
    {
        if ( afAD1[0] < -fR01 )
            return false;
    }

    // axis C0+t*A1
    aafC[1][0] = akA[1].Dot(akB[0]);
    aafC[1][1] = akA[1].Dot(akB[1]);
    aafC[1][2] = akA[1].Dot(akB[2]);
    afAD0[1] = akA[1].Dot(kD0);
    afAD1[1] = akA[1].Dot(kD1);
    aafAbsC[1][0] = Math::FAbs(aafC[1][0]);
    aafAbsC[1][1] = Math::FAbs(aafC[1][1]);
    aafAbsC[1][2] = Math::FAbs(aafC[1][2]);
    fR1 = afEB[0]*aafAbsC[1][0]+afEB[1]*aafAbsC[1][1]+afEB[2]*aafAbsC[1][2];
    fR01 = afEA[1] + fR1;
    if ( afAD0[1] > fR01 )
    {
        if ( afAD1[1] > fR01 )
            return false;
    }
    else if ( afAD0[1] < -fR01 )
    {
        if ( afAD1[1] < -fR01 )
            return false;
    }

    // axis C0+t*A2
    aafC[2][0] = akA[2].Dot(akB[0]);
    aafC[2][1] = akA[2].Dot(akB[1]);
    aafC[2][2] = akA[2].Dot(akB[2]);
    afAD0[2] = akA[2].Dot(kD0);
    afAD1[2] = akA[2].Dot(kD1);
    aafAbsC[2][0] = Math::FAbs(aafC[2][0]);
    aafAbsC[2][1] = Math::FAbs(aafC[2][1]);
    aafAbsC[2][2] = Math::FAbs(aafC[2][2]);
    fR1 = afEB[0]*aafAbsC[2][0]+afEB[1]*aafAbsC[2][1]+afEB[2]*aafAbsC[2][2];
    fR01 = afEA[2] + fR1;
    if ( afAD0[2] > fR01 )
    {
        if ( afAD1[2] > fR01 )
            return false;
    }
    else if ( afAD0[2] < -fR01 )
    {
        if ( afAD1[2] < -fR01 )
            return false;
    }

    // axis C0+t*B0
    fR = akB[0].Dot(kD0);
    fR0 = afEA[0]*aafAbsC[0][0]+afEA[1]*aafAbsC[1][0]+afEA[2]*aafAbsC[2][0];
    fR01 = fR0 + afEB[0];
    if ( fR > fR01 )
    {
        fR = akB[0].Dot(kD1);
        if ( fR > fR01)
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = akB[0].Dot(kD1);
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*B1
    fR = akB[1].Dot(kD0);
    fR0 = afEA[0]*aafAbsC[0][1]+afEA[1]*aafAbsC[1][1]+afEA[2]*aafAbsC[2][1];
    fR01 = fR0 + afEB[1];
    if ( fR > fR01 )
    {
        fR = akB[1].Dot(kD1);
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = akB[1].Dot(kD1);
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*B2
    fR = akB[2].Dot(kD0);
    fR0 = afEA[0]*aafAbsC[0][2]+afEA[1]*aafAbsC[1][2]+afEA[2]*aafAbsC[2][2];
    fR01 = fR0 + afEB[2];
    if ( fR > fR01 )
    {
        fR = akB[2].Dot(kD1);
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = akB[2].Dot(kD1);
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A0xB0
    fR = afAD0[2]*aafC[1][0]-afAD0[1]*aafC[2][0];
    fR0 = afEA[1]*aafAbsC[2][0] + afEA[2]*aafAbsC[1][0];
    fR1 = afEB[1]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][1];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[2]*aafC[1][0]-afAD1[1]*aafC[2][0];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[2]*aafC[1][0]-afAD1[1]*aafC[2][0];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A0xB1
    fR = afAD0[2]*aafC[1][1]-afAD0[1]*aafC[2][1];
    fR0 = afEA[1]*aafAbsC[2][1] + afEA[2]*aafAbsC[1][1];
    fR1 = afEB[0]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[2]*aafC[1][1]-afAD1[1]*aafC[2][1];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[2]*aafC[1][1]-afAD1[1]*aafC[2][1];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A0xB2
    fR = afAD0[2]*aafC[1][2]-afAD0[1]*aafC[2][2];
    fR0 = afEA[1]*aafAbsC[2][2] + afEA[2]*aafAbsC[1][2];
    fR1 = afEB[0]*aafAbsC[0][1] + afEB[1]*aafAbsC[0][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[2]*aafC[1][2]-afAD1[1]*aafC[2][2];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[2]*aafC[1][2]-afAD1[1]*aafC[2][2];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A1xB0
    fR = afAD0[0]*aafC[2][0]-afAD0[2]*aafC[0][0];
    fR0 = afEA[0]*aafAbsC[2][0] + afEA[2]*aafAbsC[0][0];
    fR1 = afEB[1]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][1];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[0]*aafC[2][0]-afAD1[2]*aafC[0][0];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[0]*aafC[2][0]-afAD1[2]*aafC[0][0];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A1xB1
    fR = afAD0[0]*aafC[2][1]-afAD0[2]*aafC[0][1];
    fR0 = afEA[0]*aafAbsC[2][1] + afEA[2]*aafAbsC[0][1];
    fR1 = afEB[0]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[0]*aafC[2][1]-afAD1[2]*aafC[0][1];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[0]*aafC[2][1]-afAD1[2]*aafC[0][1];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A1xB2
    fR = afAD0[0]*aafC[2][2]-afAD0[2]*aafC[0][2];
    fR0 = afEA[0]*aafAbsC[2][2] + afEA[2]*aafAbsC[0][2];
    fR1 = afEB[0]*aafAbsC[1][1] + afEB[1]*aafAbsC[1][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[0]*aafC[2][2]-afAD1[2]*aafC[0][2];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[0]*aafC[2][2]-afAD1[2]*aafC[0][2];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A2xB0
    fR = afAD0[1]*aafC[0][0]-afAD0[0]*aafC[1][0];
    fR0 = afEA[0]*aafAbsC[1][0] + afEA[1]*aafAbsC[0][0];
    fR1 = afEB[1]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][1];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[1]*aafC[0][0]-afAD1[0]*aafC[1][0];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[1]*aafC[0][0]-afAD1[0]*aafC[1][0];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A2xB1
    fR = afAD0[1]*aafC[0][1]-afAD0[0]*aafC[1][1];
    fR0 = afEA[0]*aafAbsC[1][1] + afEA[1]*aafAbsC[0][1];
    fR1 = afEB[0]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[1]*aafC[0][1]-afAD1[0]*aafC[1][1];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[1]*aafC[0][1]-afAD1[0]*aafC[1][1];
        if ( fR < -fR01 )
            return false;
    }

    // axis C0+t*A2xB2
    fR = afAD0[1]*aafC[0][2]-afAD0[0]*aafC[1][2];
    fR0 = afEA[0]*aafAbsC[1][2] + afEA[1]*aafAbsC[0][2];
    fR1 = afEB[0]*aafAbsC[2][1] + afEB[1]*aafAbsC[2][0];
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
    {
        fR = afAD1[1]*aafC[0][2]-afAD1[0]*aafC[1][2];
        if ( fR > fR01 )
            return false;
    }
    else if ( fR < -fR01 )
    {
        fR = afAD1[1]*aafC[0][2]-afAD1[0]*aafC[1][2];
        if ( fR < -fR01 )
            return false;
    }

    // At this point none of the 15 axes separate the boxes.  It is still
    // possible that they are separated as viewed in any plane orthogonal
    // to the relative direction of motion W.  In the worst case, the two
    // projected boxes are hexagons.  This requires three separating axis
    // tests per box.
    Vector3 kWxD0 = kW.Cross(kD0);
    Real afWA[3], afWB[3];

    // axis C0 + t*WxA0
    afWA[1] = kW.Dot(akA[1]);
    afWA[2] = kW.Dot(akA[2]);
    fR = Math::FAbs(akA[0].Dot(kWxD0));
    fR0 = afEA[1]*Math::FAbs(afWA[2]) + afEA[2]*Math::FAbs(afWA[1]);
    fR1 =
        afEB[0]*Math::FAbs(aafC[1][0]*afWA[2] - aafC[2][0]*afWA[1]) +
        afEB[1]*Math::FAbs(aafC[1][1]*afWA[2] - aafC[2][1]*afWA[1]) +
        afEB[2]*Math::FAbs(aafC[1][2]*afWA[2] - aafC[2][2]*afWA[1]);
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0 + t*WxA1
    afWA[0] = kW.Dot(akA[0]);
    fR = Math::FAbs(akA[1].Dot(kWxD0));
    fR0 = afEA[2]*Math::FAbs(afWA[0]) + afEA[0]*Math::FAbs(afWA[2]);
    fR1 =
        afEB[0]*Math::FAbs(aafC[2][0]*afWA[0] - aafC[0][0]*afWA[2]) +
        afEB[1]*Math::FAbs(aafC[2][1]*afWA[0] - aafC[0][1]*afWA[2]) +
        afEB[2]*Math::FAbs(aafC[2][2]*afWA[0] - aafC[0][2]*afWA[2]);
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0 + t*WxA2
    fR = Math::FAbs(akA[2].Dot(kWxD0));
    fR0 = afEA[0]*Math::FAbs(afWA[1]) + afEA[1]*Math::FAbs(afWA[0]);
    fR1 =
        afEB[0]*Math::FAbs(aafC[0][0]*afWA[1] - aafC[1][0]*afWA[0]) +
        afEB[1]*Math::FAbs(aafC[0][1]*afWA[1] - aafC[1][1]*afWA[0]) +
        afEB[2]*Math::FAbs(aafC[0][2]*afWA[1] - aafC[1][2]*afWA[0]);
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0 + t*WxB0
    afWB[1] = kW.Dot(akB[1]);
    afWB[2] = kW.Dot(akB[2]);
    fR = Math::FAbs(akB[0].Dot(kWxD0));
    fR0 =
        afEA[0]*Math::FAbs(aafC[0][1]*afWB[2] - aafC[0][2]*afWB[1]) +
        afEA[1]*Math::FAbs(aafC[1][1]*afWB[2] - aafC[1][2]*afWB[1]) +
        afEA[2]*Math::FAbs(aafC[2][1]*afWB[2] - aafC[2][2]*afWB[1]);
    fR1 = afEB[1]*Math::FAbs(afWB[2]) + afEB[2]*Math::FAbs(afWB[1]);
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0 + t*WxB1
    afWB[0] = kW.Dot(akB[0]);
    fR = Math::FAbs(akB[1].Dot(kWxD0));
    fR0 =
        afEA[0]*Math::FAbs(aafC[0][2]*afWB[0] - aafC[0][0]*afWB[2]) +
        afEA[1]*Math::FAbs(aafC[1][2]*afWB[0] - aafC[1][0]*afWB[2]) +
        afEA[2]*Math::FAbs(aafC[2][2]*afWB[0] - aafC[2][0]*afWB[2]);
    fR1 = afEB[2]*Math::FAbs(afWB[0]) + afEB[0]*Math::FAbs(afWB[2]);
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    // axis C0 + t*WxB2
    fR = Math::FAbs(akB[2].Dot(kWxD0));
    fR0 =
        afEA[0]*Math::FAbs(aafC[0][0]*afWB[1] - aafC[0][1]*afWB[0]) +
        afEA[1]*Math::FAbs(aafC[1][0]*afWB[1] - aafC[1][1]*afWB[0]) +
        afEA[2]*Math::FAbs(aafC[2][0]*afWB[1] - aafC[2][1]*afWB[0]);
    fR1 = afEB[0]*Math::FAbs(afWB[1]) + afEB[1]*Math::FAbs(afWB[0]);
    fR01 = fR0 + fR1;
    if ( fR > fR01 )
        return false;

    return true;
}
//----------------------------------------------------------------------------
bool Mgc::TestIntersection (Real fTime, int iNumSteps, const Box3& rkBox0,
    const Vector3& rkVel0, const Vector3& rkRotCen0,
    const Vector3& rkRotAxis0, const Box3& rkBox1, const Vector3& rkVel1,
    const Vector3& rkRotCen1, const Vector3& rkRotAxis1)
{
    // time step for the integration
    Real fStep = fTime/Real(iNumSteps);

    // initialize subinterval boxes
    Box3 kSubBox0, kSubBox1;
    kSubBox0.Center() = rkBox0.Center();
    kSubBox1.Center() = rkBox1.Center();
    int i;
    for (i = 0; i < 3; i++)
    {
        kSubBox0.Axis(i) = rkBox0.Axis(i);
        kSubBox1.Axis(i) = rkBox1.Axis(i);
    }

    // integrate the differential equations using Euler's method
    for (int iStep = 1; iStep <= iNumSteps; iStep++)
    {
        // compute box velocities and test boxes for intersection
        Real fSubTime = fStep*Real(iStep);
        Vector3 kNewRotCen0 = rkRotCen0 + fSubTime*rkVel0;
        Vector3 kNewRotCen1 = rkRotCen1 + fSubTime*rkVel1;
        Vector3 kDiff0 = kSubBox0.Center() - kNewRotCen0;
        Vector3 kDiff1 = kSubBox1.Center() - kNewRotCen1;
        Vector3 kSubVel0 = fStep*(rkVel0 + rkRotAxis0.Cross(kDiff0));
        Vector3 kSubVel1 = fStep*(rkVel1 + rkRotAxis1.Cross(kDiff1));
        if ( TestIntersection(fStep,kSubBox0,kSubVel0,kSubBox1,kSubVel1) )
            return true;

        // update the box centers
        kSubBox0.Center() = kSubBox0.Center() + kSubVel0;
        kSubBox1.Center() = kSubBox1.Center() + kSubVel1;

        // update the box axes
        for (i = 0; i < 3; i++)
        {
            kSubBox0.Axis(i) = kSubBox0.Axis(i) +
                fStep*rkRotAxis0.Cross(kSubBox0.Axis(i));

            kSubBox1.Axis(i) = kSubBox1.Axis(i) +
                fStep*rkRotAxis1.Cross(kSubBox1.Axis(i));
        }

        // Use Gram-Schmidt to orthonormalize the updated axes.  NOTE:  If
        // T/N is small and N is small, you can remove this expensive step
        // with the assumption that the updated axes are nearly orthonormal.
        Vector3::Orthonormalize(kSubBox0.Axes());
        Vector3::Orthonormalize(kSubBox1.Axes());
    }

    // NOTE:  If the boxes do not intersect, then the application might
    // want to move/rotate the boxes to their new locations.  In this case
    // you want to return the final values of kSubBox0 and kSubBox1 so that
    // the application can set rkBox0 <- kSubBox0 and rkBox1 <- kSubBox1.
    // Otherwise, the application would have to solve the differential
    // equation again or compute the new box locations using the closed form
    // solution for the rigid motion.

    return false;
}
//----------------------------------------------------------------------------


