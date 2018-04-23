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

#ifndef MGCINTR3DBOXBOX_H
#define MGCINTR3DBOXBOX_H

#include "MgcBox3.h"

namespace Mgc {

// boxes are stationary
MAGICFM bool TestIntersection (const Box3& rkBox0, const Box3& rkBox1);

// boxes have constant linear velocity
MAGICFM bool TestIntersection (Real fTime, const Box3& rkBox0,
    const Vector3& rkVel0, const Box3& rkBox1, const Vector3& rkVel1);

// boxes have constant linear velocities and angular velocities
MAGICFM bool TestIntersection (Real fTime, int iNumSteps, const Box3& rkBox0,
    const Vector3& rkVel0, const Vector3& rkRotCen0,
    const Vector3& rkRotAxis0, const Box3& rkBox1, const Vector3& rkVel1,
    const Vector3& rkRotCen1, const Vector3& rkRotAxis1);

} // namespace Mgc

#endif


