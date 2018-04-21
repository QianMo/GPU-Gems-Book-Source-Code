/**
 @file graphics3D.h

 This header includes all of the graphics3D libraries in
 appropriate namespaces.

 @maintainer Morgan McGuire, matrix@graphics3d.com

 @created 2001-08-25
 @edited  2004-01-10

 Copyright 2001-2003, Morgan McGuire.
 All rights reserved.
*/


#ifndef G3D_GRAPHICS3D_H
#define G3D_GRAPHICS3D_H

#define NOMINMAX
#ifdef min
    #undef min
#endif
#ifdef max
    #undef max
#endif

#include "G3D/platform.h"
#include "G3D/Array.h"
#include "G3D/Queue.h"
#include "G3D/format.h"
#include "G3D/Vector2.h"
#include "G3D/Vector3.h"
#include "G3D/Vector4.h"
#include "G3D/Color3.h"
#include "G3D/Color4.h"
#include "G3D/Matrix3.h"
#include "G3D/CoordinateFrame.h"
#include "G3D/PhysicsFrame.h"
#include "G3D/Plane.h"
#include "G3D/Line.h"
#include "G3D/Ray.h"
#include "G3D/Sphere.h"
#include "G3D/Box.h"
#include "G3D/AABox.h"
#include "G3D/Cone.h"
#include "G3D/Quat.h"
#include "G3D/stringutils.h"
#include "G3D/prompt.h"
#include "G3D/g3derror.h"
#include "G3D/Table.h"
#include "G3D/Set.h"
#include "G3D/BinaryInput.h"
#include "G3D/BinaryOutput.h"
#include "G3D/debug.h"
#include "G3D/G3DGameUnits.h"
#include "G3D/g3dmath.h"
#include "G3D/fileutils.h"
#include "G3D/ReferenceCount.h"
#include "G3D/GImage.h"
#include "G3D/CollisionDetection.h"
#include "G3D/Log.h"
#include "G3D/TextInput.h"
#include "G3D/NetworkDevice.h"
#include "G3D/System.h"
#include "G3D/LineSegment.h"
#include "G3D/Capsule.h"
#include "G3D/Triangle.h"
#include "G3D/Color3uint8.h"
#include "G3D/Color4uint8.h"
#include "G3D/Vector2int16.h"
#include "G3D/Vector3int16.h"
#include "G3D/ConvexPolyhedron.h"
#include "G3D/Discovery.h"
#include "G3D/MeshAlg.h"
#include "G3D/vectorMath.h"
#include "G3D/Rect2D.h"
#include "G3D/GCamera.h"
#include "G3D/GLight.h"
#include "G3D/KDTreeSet.h"

#endif

