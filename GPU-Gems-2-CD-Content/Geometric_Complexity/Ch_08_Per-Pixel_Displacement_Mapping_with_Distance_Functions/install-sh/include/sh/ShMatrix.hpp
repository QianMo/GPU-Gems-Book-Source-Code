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
#ifndef SH_SHMATRIX_HPP
#define SH_SHMATRIX_HPP

#include "ShMeta.hpp"
#include "ShVariable.hpp"
#include "ShAttrib.hpp"
#include "ShRefCount.hpp"

namespace SH {

template<int Rows, int Cols, ShBindingType Binding, typename T>
class ShMatrix;

template<int Rows, int Cols, typename T>
class ShMatrixRows;

/** A matrix of Rows by Cols elements.
 * A matrix is a representation of a linear operator.  In Sh, this
 * class represents SMALL matrices that will fit in registers.   For
 * large amounts of data, use an ShArray.
 * @see ShArray
 */
template<int Rows, int Cols, ShBindingType Binding, typename T>
class ShMatrix: public virtual ShMeta {
public:
  typedef typename ShHostType<T>::type H; 
  typedef H HostType; 
    
  /** \brief Identity constructor.
   *
   * Constructs an identity matrix.
   * For non-square matrices, the largest square upper-left submatrix
   * possible is made an identity matrix, with the rest left zero.
   *
   */
  ShMatrix();

  /** \brief Copy constructor.
   *
   * Construct a matrix with the same contents as the given matrix.
   */
  ShMatrix(const ShMatrix<Rows, Cols, Binding, T>& other);


  /** \brief Copy constructor.
   *
   * Construct a matrix with the same contents as the given matrix.
   */
  template<ShBindingType Binding2>
  ShMatrix(const ShMatrix<Rows, Cols, Binding2, T>& other);
  
  ~ShMatrix();

  /** \brief Assignment.
   *
   * Replace each entry in this matrix with the corresponding entry of
   * the given matrix.
   */
  ShMatrix& operator=(const ShMatrix<Rows, Cols, Binding, T>& other);

  /** \brief Assignment.
   *
   * Replace each entry in this matrix with the corresponding entry of
   * the given matrix.
   */
  template<ShBindingType Binding2>
  ShMatrix& operator=(const ShMatrix<Rows, Cols, Binding2, T>& other);

  /** \brief Attribute row access.
   *
   * Return a reference to the given row as an ShAttrib.
   */
  ShAttrib<Cols, Binding, T>& operator[](int i);

  /** \brief Attribute row access.
   *
   * Return a reference to the given row as an ShAttrib.
   */
  const ShAttrib<Cols, Binding, T>& operator[](int i) const;

  /** \brief Modifying componentwise addition
   *
   * Add each entry in the given matrix to each entry in this
   * matrix matching its row and column index.
   */
  template<ShBindingType Binding2>
  ShMatrix& operator+=(const ShMatrix<Rows, Cols, Binding2, T>& other);

  /** \brief Modifying componentwise subtraction
   *
   * Subtract each entry in the given matrix from each entry in this
   * matrix matching its row and column index.
   */
  template<ShBindingType Binding2>
  ShMatrix& operator-=(const ShMatrix<Rows, Cols, Binding2, T>& other);


  /** \brief Modifying componentwise division
   *
   * Divide each entry in this matrix by the entry in the given matrix
   * matching its row and column index.
   *
   */
  template<ShBindingType Binding2>
  ShMatrix& operator/=(const ShMatrix<Rows, Cols, Binding2, T>& other);    
   

  /** \brief Obtain a submatrix of this matrix.
   *
   * Return a copy of this matrix not containing the given row and column.
   */
  ShMatrix<Rows - 1, Cols -1, SH_TEMP, T> subMatrix(int,int) const;

  void setTranslation(const ShGeneric<Rows-1, T>& trans);
  
  void setScaling(const ShGeneric<Rows-1, T>& scale);


  /** \brief Modifying scalar multiplicatoin
   *
   * Multiply the given scalar attribute with each component of this
   * matrix.
   */
  ShMatrix& operator*=(const ShGeneric<1, T>& a);
    
  /** \brief Modifying scalar division
   *
   * Divide each component of this matrix by the given attribute.
   */
  ShMatrix& operator/=(const ShGeneric<1, T>& a);

  /**@name Swizzling
   * Retrieve some set of rows from this matrix. These must be
   * swizzled again to operate on them.
   */
  //@{
  ShMatrixRows<Rows, Cols, T> operator()() const; ///< Identity swizzle
  ShMatrixRows<1, Cols, T> operator()(int) const;
  ShMatrixRows<2, Cols, T> operator()(int, int) const;
  ShMatrixRows<3, Cols, T> operator()(int, int, int) const;
  ShMatrixRows<4, Cols, T> operator()(int, int, int, int) const;
  //@}
  
  /**@name Metadata
   * This data is useful for various things, including asset
   * management.
   */
  //@{

  /// Set a range of values for this matrix
  void range(H low, H high);

  //@}
  
  virtual std::string name() const;
  virtual void name(const std::string& n);
  virtual bool has_name() const;
  
  virtual bool internal() const;
  virtual void internal(bool);

  virtual std::string title() const;
  virtual void title(const std::string& t);

  virtual std::string description() const;
  virtual void description(const std::string& d);

  virtual std::string meta(const std::string& key) const;
  virtual void meta(const std::string& key, const std::string& value);

private:
  /** \brief The Matrix itself
   *
   * The data structure that hold the data
   */
  ShAttrib<Cols, Binding, T> m_data[Rows];
  
};
/** \brief Matrix output operator
 *
 * Print a representation of the given (uniform or constant) matrix's
 * values to the given stream.
 */

template<int R, int C, ShBindingType B, typename Ty>
std::ostream& operator<<(std::ostream& out,
                         const ShMatrix<R, C, B, Ty>& m);

/** A few rows from a matrix.
 * This is an intermediate structure representing some rows that have
 * just been swizzled from a matrix. The only useful operation is to
 * swizzle these rows again to obtain a submatrix with the given rows
 * and columns.
 */
template<int Rows, int Cols, typename T>
class ShMatrixRows {
public:
  template<ShBindingType Binding>
  ShMatrixRows(const ShMatrix<Rows, Cols, Binding, T>& source);
  
  template<int OR, ShBindingType Binding>
  ShMatrixRows(const ShMatrix<OR, Cols, Binding, T>& source,
               int idx0);
  template<int OR, ShBindingType Binding>
  ShMatrixRows(const ShMatrix<OR, Cols, Binding, T>& source,
               int idx0, int idx1);
  template<int OR, ShBindingType Binding>
  ShMatrixRows(const ShMatrix<OR, Cols, Binding, T>& source,
               int idx0, int idx1, int idx2);
  template<int OR, ShBindingType Binding>
  ShMatrixRows(const ShMatrix<OR, Cols, Binding, T>& source,
               int idx0, int idx1, int idx2, int idx3);

  ShMatrixRows(const ShMatrixRows<Rows, Cols, T>& other);
  
  ShMatrixRows& operator=(const ShMatrixRows<Rows, Cols, T>& other);

  /**@name Swizzling
   * Retrieve a matrix consisting of these rows and the given columns.
   */
  //@{
  ShMatrix<Rows, Cols, SH_TEMP, T> operator()() const; ///< Identity swizzle
  ShMatrix<Rows, 1, SH_TEMP, T> operator()(int) const;
  ShMatrix<Rows, 2, SH_TEMP, T> operator()(int, int) const;
  ShMatrix<Rows, 3, SH_TEMP, T> operator()(int, int, int) const;
  ShMatrix<Rows, 4, SH_TEMP, T> operator()(int, int, int, int) const;
  //@}
private:
  ShAttrib<Cols, SH_TEMP, T> m_data[Rows];
};


typedef ShMatrix<1, 1, SH_INPUT, float> ShInputMatrix1x1f;
typedef ShMatrix<1, 1, SH_OUTPUT, float> ShOutputMatrix1x1f;
typedef ShMatrix<1, 1, SH_INOUT, float> ShInOutMatrix1x1f;
typedef ShMatrix<1, 1, SH_TEMP, float> ShMatrix1x1f;
typedef ShMatrix<1, 2, SH_INPUT, float> ShInputMatrix1x2f;
typedef ShMatrix<1, 2, SH_OUTPUT, float> ShOutputMatrix1x2f;
typedef ShMatrix<1, 2, SH_INOUT, float> ShInOutMatrix1x2f;
typedef ShMatrix<1, 2, SH_TEMP, float> ShMatrix1x2f;
typedef ShMatrix<1, 3, SH_INPUT, float> ShInputMatrix1x3f;
typedef ShMatrix<1, 3, SH_OUTPUT, float> ShOutputMatrix1x3f;
typedef ShMatrix<1, 3, SH_INOUT, float> ShInOutMatrix1x3f;
typedef ShMatrix<1, 3, SH_TEMP, float> ShMatrix1x3f;
typedef ShMatrix<1, 4, SH_INPUT, float> ShInputMatrix1x4f;
typedef ShMatrix<1, 4, SH_OUTPUT, float> ShOutputMatrix1x4f;
typedef ShMatrix<1, 4, SH_INOUT, float> ShInOutMatrix1x4f;
typedef ShMatrix<1, 4, SH_TEMP, float> ShMatrix1x4f;

typedef ShMatrix<2, 1, SH_INPUT, float> ShInputMatrix2x1f;
typedef ShMatrix<2, 1, SH_OUTPUT, float> ShOutputMatrix2x1f;
typedef ShMatrix<2, 1, SH_INOUT, float> ShInOutMatrix2x1f;
typedef ShMatrix<2, 1, SH_TEMP, float> ShMatrix2x1f;
typedef ShMatrix<2, 2, SH_INPUT, float> ShInputMatrix2x2f;
typedef ShMatrix<2, 2, SH_OUTPUT, float> ShOutputMatrix2x2f;
typedef ShMatrix<2, 2, SH_INOUT, float> ShInOutMatrix2x2f;
typedef ShMatrix<2, 2, SH_TEMP, float> ShMatrix2x2f;
typedef ShMatrix<2, 3, SH_INPUT, float> ShInputMatrix2x3f;
typedef ShMatrix<2, 3, SH_OUTPUT, float> ShOutputMatrix2x3f;
typedef ShMatrix<2, 3, SH_INOUT, float> ShInOutMatrix2x3f;
typedef ShMatrix<2, 3, SH_TEMP, float> ShMatrix2x3f;
typedef ShMatrix<2, 4, SH_INPUT, float> ShInputMatrix2x4f;
typedef ShMatrix<2, 4, SH_OUTPUT, float> ShOutputMatrix2x4f;
typedef ShMatrix<2, 4, SH_INOUT, float> ShInOutMatrix2x4f;
typedef ShMatrix<2, 4, SH_TEMP, float> ShMatrix2x4f;

typedef ShMatrix<3, 1, SH_INPUT, float> ShInputMatrix3x1f;
typedef ShMatrix<3, 1, SH_OUTPUT, float> ShOutputMatrix3x1f;
typedef ShMatrix<3, 1, SH_INOUT, float> ShInOutMatrix3x1f;
typedef ShMatrix<3, 1, SH_TEMP, float> ShMatrix3x1f;
typedef ShMatrix<3, 2, SH_INPUT, float> ShInputMatrix3x2f;
typedef ShMatrix<3, 2, SH_OUTPUT, float> ShOutputMatrix3x2f;
typedef ShMatrix<3, 2, SH_INOUT, float> ShInOutMatrix3x2f;
typedef ShMatrix<3, 2, SH_TEMP, float> ShMatrix3x2f;
typedef ShMatrix<3, 3, SH_INPUT, float> ShInputMatrix3x3f;
typedef ShMatrix<3, 3, SH_OUTPUT, float> ShOutputMatrix3x3f;
typedef ShMatrix<3, 3, SH_INOUT, float> ShInOutMatrix3x3f;
typedef ShMatrix<3, 3, SH_TEMP, float> ShMatrix3x3f;
typedef ShMatrix<3, 4, SH_INPUT, float> ShInputMatrix3x4f;
typedef ShMatrix<3, 4, SH_OUTPUT, float> ShOutputMatrix3x4f;
typedef ShMatrix<3, 4, SH_INOUT, float> ShInOutMatrix3x4f;
typedef ShMatrix<3, 4, SH_TEMP, float> ShMatrix3x4f;

typedef ShMatrix<4, 1, SH_INPUT, float> ShInputMatrix4x1f;
typedef ShMatrix<4, 1, SH_OUTPUT, float> ShOutputMatrix4x1f;
typedef ShMatrix<4, 1, SH_INOUT, float> ShInOutMatrix4x1f;
typedef ShMatrix<4, 1, SH_TEMP, float> ShMatrix4x1f;
typedef ShMatrix<4, 2, SH_INPUT, float> ShInputMatrix4x2f;
typedef ShMatrix<4, 2, SH_OUTPUT, float> ShOutputMatrix4x2f;
typedef ShMatrix<4, 2, SH_INOUT, float> ShInOutMatrix4x2f;
typedef ShMatrix<4, 2, SH_TEMP, float> ShMatrix4x2f;
typedef ShMatrix<4, 3, SH_INPUT, float> ShInputMatrix4x3f;
typedef ShMatrix<4, 3, SH_OUTPUT, float> ShOutputMatrix4x3f;
typedef ShMatrix<4, 3, SH_INOUT, float> ShInOutMatrix4x3f;
typedef ShMatrix<4, 3, SH_TEMP, float> ShMatrix4x3f;
typedef ShMatrix<4, 4, SH_INPUT, float> ShInputMatrix4x4f;
typedef ShMatrix<4, 4, SH_OUTPUT, float> ShOutputMatrix4x4f;
typedef ShMatrix<4, 4, SH_INOUT, float> ShInOutMatrix4x4f;
typedef ShMatrix<4, 4, SH_TEMP, float> ShMatrix4x4f;

}

#include "ShMatrixImpl.hpp"

#endif

