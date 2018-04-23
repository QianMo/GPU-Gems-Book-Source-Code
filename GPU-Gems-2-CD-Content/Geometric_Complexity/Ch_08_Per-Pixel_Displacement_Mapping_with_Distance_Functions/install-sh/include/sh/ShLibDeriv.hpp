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
#ifndef SHLIBDERIV_HPP
#define SHLIBDERIV_HPP

#include "ShGeneric.hpp"
#include "ShLib.hpp"
#include "ShMatrix.hpp"

#ifndef WIN32
namespace SH {

/** \defgroup lib_deriv Derivatives
 * @ingroup library
 * Operations related to derivatives of values.
 * @{
 */

/** Screen-space x derivatives
 */
template<int N, typename T>
ShGeneric<N, T> dx(const ShGeneric<N, T>& var);

/** Screen-space y derivatives
 */
template<int N, typename T>
ShGeneric<N, T> dy(const ShGeneric<N, T>& var);

/** Maximum value of absolute derivatives
 */
template<int N, typename T>
ShGeneric<N, T> fwidth(const ShGeneric<N, T>& var);

/** Pair of screen-space derivatives
 */
template<typename T>
ShGeneric<2, T> gradient(const ShGeneric<1, T>& var);

/** Jacobian matrix
 */
template<int N, typename T>
ShMatrix<2, N, SH_TEMP, T> jacobian(const ShGeneric<N, T>& var);

/*@}*/

}
#endif

#include "ShLibDerivImpl.hpp"

#endif
