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
#ifndef SHMANIPULATOR_HPP
#define SHMANIPULATOR_HPP

#include "ShProgram.hpp"
#include <vector>
#include <string>
#include <sstream>

namespace SH {

/** \brief A type trait class that determines storage type used for T in 
 * a ShManipulator.  There must be an automatic conversion from T
 * to StorageType or an applicable copy constructor. The storage type also 
 * must be responsible for its own memory management as the code often
 * uses shallow copies of ranges/range vectors.
 *
 * The only function that needs to be added after adding a new storage type
 * is the OffsetRange::absIndex function in ShManipulator.cpp.
 * This converts the storage type index into an absolute integer index.
 */
template<typename T> 
struct storage_trait {
  typedef T StorageType;
};

template<>
struct storage_trait<const char*> {
  typedef std::string StorageType; 
};

enum OffsetRangeErrors {
  OFFSET_RANGE_BAD_OFFSET = -1,
  OFFSET_RANGE_BAD_INDEX = -2
};

template<typename T>
class OffsetRange {
  public:
    OffsetRange(); 
    OffsetRange( T start, T end );
    OffsetRange( T start, int startOffset, T end, int endOffset );

    /** \brief returns an absolute position for start 
     * or BAD_OFFSET/BAD_INDEX
     * The offset is only for internal use, so those indicate the
     * range should be ignored, but user does not need feedback.
     * If the use provided index is invalid, then there should be an error message.
     */
    int absStartIndex( const ShProgramNode::VarList vars ) const; 
    int absEndIndex( const ShProgramNode::VarList vars ) const; 

    std::string toString() const; 

  private:
    T start, end;
    int startOffset, endOffset;
    int absIndex( T index, int offset, const ShProgramNode::VarList &vars ) const;
};


/** \brief A ShManipulator is a ShProgram output manipulator. 
 * This kind of manipulator permutes the outputs of 
 * a ShProgram based on given integer indices. 
 *
 * Currently, two types are supported - T = int and T = char*.
 * T = int references channels by position (negative numbers are position from end)
 * T = char* references channels by name.
 * (The Manipulator<char*> operator()(??) methods store copies of the string) 
 * (negative indices -k mean k^th channel from the end, with -1
 * being the last output channel) 
 */
template<typename T>
class ShManipulator {
  public:
    typedef typename storage_trait<T>::StorageType StorageType;
    typedef OffsetRange<StorageType> IndexRange;
    typedef std::vector<IndexRange> IndexRangeVector;

    /** \brief Creates empty manipulator of given size
     */
    ShManipulator();
    ~ShManipulator();

    ShManipulator<T>& operator()(T i);
    ShManipulator<T>& operator()(T start, T end);
    ShManipulator<T>& operator()(const IndexRange &range); 

    // converts ranges to a sequence of integer ranges using given var list 
    IndexRangeVector getRanges() const; 

    // converts to string for debugging/error messages
    std::string toString() const;
    
  protected:
    IndexRangeVector m_ranges; 

    // converts indices to positive integer indices. 
    // If it cannot be found, raises AlgebraException.
    // if the index has an offset that makes it invalid, then valid is set to false
    OffsetRange<int> convertRange(IndexRange range, const ShProgramNode::VarList &v) const;
};

/** \brief Applies a manipulator to inputs of a ShProgram
 *
 * The permutation ranges are more restrictive than output manipulation
 * since inputs cannot be repeated, and should not be discarded.
 *
 * This means that ranges in the manipulator must not overlap, and any inputs not 
 * in a range are given a default value of 0. 
 */
template<typename T>
ShProgram operator<<(const ShProgram &p, const ShManipulator<T> &m); 

/** \brief Applies a manipulator to the outputs of a ShProgram
 *
 * This makes sense since >> is left associative, so 
 *    p >> m >> q
 * looks like manipulating p's output channels to use as q's inputs.
 */
template<typename T>
ShProgram operator<<(const ShManipulator<T> &m, const ShProgram &p);


/// permute(a1, ...) is a manipulator that permutes 
// shader outputs based on given indices
//
/** \brief creates a permutation manipulator which
 * gives outputSize outputs when applied to a ShProgram
 *
 * Empty permutes are not allowed because the compiler would not
 * be able to resolve ambiguity. 
 *
 * if an index >= 0, then uses index'th output
 * if index < 0, then uses program size + index'th output
 */
template<typename T>
ShManipulator<T> shSwizzle(T i0);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3, T i4);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3, T i4, T i5);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3, T i4, T i5, T i6);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7, T i8);

template<typename T>
ShManipulator<T> shSwizzle(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7, T i8, T i9);

template<typename T>
ShManipulator<T> shSwizzle(std::vector<T> indices);

/// range manipulator that permutes ranges of shader
// outputs based on given indices
template<typename T>
ShManipulator<T> shRange(T i);

template<typename T>
ShManipulator<T> shRange(T start, T end);

/** extract is a manipulator that removes the kth output
 * and appends it before all other outputs. 
 *
 * int version:
 * if k >= 0, then take element k (indices start at 0) 
 * if k < 0, take element outputs.size + k
 *
 * string version:
 * extracts given name to beginning
 */
template<typename T>
ShManipulator<T> shExtract(T k); 

/** insert is a manipulator that does the opposite of extract.
 * It moves the first output to the kth output and shifts
 * the rest of the outputs accordingly.
 *
 * int version:
 * if k >= 0, then move to element k (indices start at 0) 
 * if k < 0, move to element outputs.size + k
 *
 * string version:
 * inserts first output to the position of the given name 
 */
template<typename T>
ShManipulator<T> shInsert(T k); 

/** drop is a manipulator that discards the k outputs.
 *
 * int version:
 * discards k'th output
 *
 * string version:
 * drops given name to beginning
 */
template<typename T>
ShManipulator<T> shDrop(T k);

typedef ShManipulator<int> ShPositionManipulator;
typedef ShManipulator<char*> ShNameManipulator;

}

#include "ShManipulatorImpl.hpp"

#endif
