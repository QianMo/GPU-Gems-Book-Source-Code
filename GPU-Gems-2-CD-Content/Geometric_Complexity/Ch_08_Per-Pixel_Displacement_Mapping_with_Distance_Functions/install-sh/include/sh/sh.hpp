// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
// 
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
// 
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
// 
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
// 
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
#ifndef SH_HPP
#define SH_HPP

#include "ShDllExport.hpp"
#include "ShException.hpp"
#include "ShContext.hpp"
#include "ShProgram.hpp"
#include "ShSyntax.hpp"
#include "ShInterval.hpp"
#include "ShHalf.hpp"
#include "ShFraction.hpp"
#include "ShAttrib.hpp"
#include "ShMatrix.hpp"
#include "ShVector.hpp"
#include "ShPoint.hpp"
#include "ShColor.hpp"
#include "ShTexCoord.hpp"
#include "ShNormal.hpp"
#include "ShPosition.hpp"
#include "ShLib.hpp"
#include "ShLibAttrib.hpp"
#include "ShLibVector.hpp"
#include "ShLibPoint.hpp"
#include "ShLibColor.hpp"
#include "ShLibTexCoord.hpp"
#include "ShLibNormal.hpp"
#include "ShLibPosition.hpp"
#include "ShMemory.hpp"
#include "ShArray.hpp"
#include "ShTable.hpp"
#include "ShTexture.hpp"
#include "ShWrap.hpp"
#include "ShClamping.hpp"
#include "ShInterp.hpp"
#include "ShBackend.hpp"
#include "ShImage.hpp"
#include "ShImage3D.hpp"
#include "ShAlgebra.hpp"
#include "ShNibbles.hpp"
#include "ShManipulator.hpp"
#include "ShFixedManipulator.hpp"
#include "ShChannelNode.hpp"
#include "ShChannel.hpp"
#include "ShStream.hpp"
#include "ShQuaternion.hpp"
#include "ShVariant.hpp"
#include "ShTypeInfo.hpp"
#include "ShMemoryDep.hpp"
#include "ShPalette.hpp"

/** \namespace SH
 * \brief The main Sh namespace.
 *
 * All Sh classes, functions and objects reside within this namespace.
 */

/** \file sh.hpp
 * \brief The main Sh include file.
 * You should only have to include this file to use Sh.
 */


#endif
