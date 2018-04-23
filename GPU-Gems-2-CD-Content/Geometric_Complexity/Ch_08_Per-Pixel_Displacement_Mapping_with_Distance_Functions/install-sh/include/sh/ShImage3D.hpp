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
#ifndef SHIMAGE3D_HPP
#define SHIMAGE3D_HPP

#include <string>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShMemory.hpp"

namespace SH {

/** A 3D image.
 * Consists of a grid of floating-point elements.   Stores data
 * in a memory object that can be shared with a 3D array, table, or
 * texture.
 * @see ShImage
 */
class
SH_DLLEXPORT ShImage3D : public ShRefCountable {
public:
  ShImage3D(); ///< Construct an empty image
  ShImage3D(int width, int height, int depth, int elements); ///< Construct a black
                                             ///image at the given width/height/elements
  ShImage3D(const ShImage3D& other); ///< Copy an image

  ~ShImage3D();

  ShImage3D& operator=(const ShImage3D& other); ///< Copy the data from
                                            ///one image to another

  int width() const; ///< Determine the width of the image
  int height() const; ///< Determine the height of the image
  int depth() const; ///< Determine the depth of the image
  int elements() const; ///< Determine the elements (floats per pixel) of
                     ///the image

  float operator()(int x, int y, int z, int i) const; ///< Retrieve a
                                               ///particular component
                                               ///from the image.
  float& operator()(int x, int y, int z, int i);  ///< Retrieve a
                                               ///particular component
                                               ///from the image.

  const float* data() const;
  float* data();
  
  ShMemoryPtr memory();
  ShPointer<const ShMemory> memory() const;
  
private:
  int m_width, m_height, m_depth;
  int m_elements;
  ShHostMemoryPtr m_memory;
};

}

#endif
