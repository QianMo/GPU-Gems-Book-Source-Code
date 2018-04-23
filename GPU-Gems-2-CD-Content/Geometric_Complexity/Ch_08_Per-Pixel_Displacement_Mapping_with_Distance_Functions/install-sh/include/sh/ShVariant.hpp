/// Sh: A GPU metaprogramming language.
//
/// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
/// Project administrator: Michael D. McCool
/// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
///          Michael D. McCool
/// 
/// This software is provided 'as-is', without any express or implied
/// warranty. In no event will the authors be held liable for any damages
/// arising from the use of this software.
/// 
/// Permission is granted to anyone to use this software for any purpose,
/// including commercial applications, and to alter it and redistribute it
/// freely, subject to the following restrictions:
/// 
/// 1. The origin of this software must not be misrepresented; you must
/// not claim that you wrote the original software. If you use this
/// software in a product, an acknowledgment in the product documentation
/// would be appreciated but is not required.
/// 
/// 2. Altered source versions must be plainly marked as such, and must
/// not be misrepresented as being the original software.
/// 
/// 3. This notice may not be removed or altered from any source
/// distribution.
//////////////////////////////////////////////////////////////////////////////
#ifndef SHVARIANT_HPP
#define SHVARIANT_HPP

#include <string>
#include <vector>
#include <ostream>
#include "ShDllExport.hpp"
#include "ShPool.hpp"
#include "ShSwizzle.hpp"
#include "ShVariableType.hpp"
#include "ShDataType.hpp"
#include "ShRefCount.hpp"

namespace SH {


/** An ShVariant is a wrapper around a fixed-size array of data
 * of a data type.  It is used internally for
 * holding tuple data for ShVariableNodes, and occasionally as temporaries
 * for larger data arrays.
 *
 * There are ShDataVariant<V> implementations that hold a T array. 
 *
 * @see ShDataVariant
 */
class 
SH_DLLEXPORT 
ShVariant: public ShRefCountable {
  public:
    ShVariant();
    virtual ~ShVariant();


    //// Gives the associated type identifier string mapped to a unique integer index 
    /// (Note this is here only for convenience and the value is cached
    /// somewhere better like in the ShVariableNode)
    virtual ShValueType valueType() const = 0; 

    /// Gives the data type held by this variant 
    virtual ShDataType dataType() const = 0; 

    /// Returns whether the value/data type in this variant match the given ones.
    virtual bool typeMatches(ShValueType valueType, ShDataType dataType) const = 0;

    //// Gives the associated type name 
    virtual const char* typeName() const = 0; 

    /// Gives the number of elements stored in this data array 
    virtual int size() const = 0;

    /// Gives the number of bytes per element stored in this data array 
    virtual int datasize() const = 0;

    /// Returns true if the array held in the ShVariant was allocated
    /// by this.
    virtual bool managed() const = 0;

    /// The only two required operations on data values - negation and copy
    /// assignment. 
    virtual void negate() = 0;


    /** Sets the values of this from other.  
     * size() must equal other.size() 
     * @{ */
    virtual void set(ShPointer<const ShVariant> other) = 0;
    virtual void set(const ShVariant* other) = 0;
    // @}

    /** Sets the value of the indexed element in this from the first element of other 
     * @{ */
    virtual void set(ShPointer<const ShVariant> other, int index) = 0;
    virtual void set(const ShVariant* other, int index) = 0;
    // @}

    /** Sets the value of this from other using the given negation and writemask
     * on this.  other.size() must equal writemask.size()
     * @{ */
    virtual void set(ShPointer<const ShVariant> other, bool neg, const ShSwizzle &writemask) = 0;
    virtual void set(const ShVariant* other, bool neg, const ShSwizzle &writemask) = 0;
    // @}

    /// Creates a copy of this ShVariant.
    virtual ShPointer<ShVariant> get() const = 0;

    /// Creates single element ShVariant with the indexed element from this.  
    virtual ShPointer<ShVariant> get(int index) const = 0;

    /// Creates a copy of the Variant with the given swizzle and negation
    /// swizzle.m_srcSize must equal size()
    virtual ShPointer<ShVariant> get(bool neg, const ShSwizzle &swizzle) const = 0; 

    /// Returns true iff other is the same size, type, and has the same values
    // This uses shDataTypeEquals
    //
    // @see ShDataType.hpp
    //@{
    virtual bool equals(ShPointer<const ShVariant> other) const = 0;
    virtual bool equals(const ShVariant* other) const = 0;
    //@}

    /// Returns whether every tuple element is positive 
    // @see ShDataType.hpp
    virtual bool isTrue() const = 0;

    /// Returns a pointer to the beginning of the array 
    //@{
    virtual void* array() = 0;
    virtual const void* array() const = 0;
    // @}


    /// Encodes the data value as a string
    virtual std::string encode() const = 0;
    virtual std::string encode(int index, int repeats=1) const = 0;
    virtual std::string encode(bool neg, const ShSwizzle &swizzle) const = 0;

    /// C++ array declaration compatible encoding.
    /// Generates a string that can be used as an array initializer 
    /// i.e. T foo[size] = { encodeArray() };
    /// @todo type may want to put this somewhere else, it doesn't really belong
    virtual std::string encodeArray() const = 0;
};

typedef ShPointer<ShVariant> ShVariantPtr;
typedef ShPointer<const ShVariant> ShVariantCPtr;

/* A fixed-size array of a specific data type that can act as an ShVariant 
 *
 * This is different from ShMemory objects which hold arbitrary typed
 * data in byte arrays (that eventually may include some 
 * unordered collection of several types)
 *
 * @see ShMemory 
 **/ 
template<typename T, ShDataType DT>
class ShDataVariant: public ShVariant {
  public:
    static const ShValueType value_type = ShStorageTypeInfo<T>::value_type;
    typedef ShPointer<ShDataVariant<T, DT> > PtrType;
    typedef ShPointer<const ShDataVariant<T, DT> > CPtrType;
    typedef typename ShDataTypeCppType<T, DT>::type DataType;
    typedef DataType* iterator;
    typedef const DataType* const_iterator;

    /// Constructs a data array and sets the value to a default value
    /// (typically zero)
    ShDataVariant(int N); 

    /// Constructs a data array and sets the value to a given value 
    ShDataVariant(int N, const DataType &value); 

    /// Constructs a data array that reads its size and values from
    /// a string encoding (must be from the encode() method of a ShDataVariant
    /// of the same type)
    ShDataVariant(std::string encodedValue);

    /// Constructs a data array from an existing array of type T
    /// of the given size.  This uses the given array internally
    /// iff managed = false, otherwise it allocates a new array 
    /// and makes a copy.
    ShDataVariant(void *data, int N, bool managed = true);

    /// Constructs a data array using values from another array
    /// swizzled and negated as requested. 
    ShDataVariant(const ShDataVariant<T, DT> &other);
    ShDataVariant(const ShDataVariant<T, DT> &other, bool neg, const ShSwizzle &swizzle); 

    ~ShDataVariant();

    ShValueType valueType() const; 
    ShDataType dataType() const; 
    bool typeMatches(ShValueType valueType, ShDataType dataType) const; 

    //// Gives the associated type name 
    const char* typeName() const; 

    //std::string typeName() const; 
    
    int size() const; 
    int datasize() const; 

    bool managed() const;

    void negate();

    void set(ShVariantCPtr other);
    void set(const ShVariant* other);
    void set(ShVariantCPtr other, int index);
    void set(const ShVariant* other, int index);
    void set(ShVariantCPtr other, bool neg, const ShSwizzle &writemask);
    void set(const ShVariant* other, bool neg, const ShSwizzle &writemask);

    ShVariantPtr get() const; 
    ShVariantPtr get(int index) const; 
    ShVariantPtr get(bool neg, const ShSwizzle &swizzle) const; 

    bool equals(ShVariantCPtr other) const; 
    bool equals(const ShVariant* other) const; 

    bool isTrue() const;

    void* array(); 
    const void* array() const; 

    DataType& operator[](int index);
    const DataType& operator[](int index) const;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    //// Encodes the tuple values into a string
    //// For now, the encoding cannot contain the character '$' 
    //// and the istream >> T function cannot read $'s
    std::string encode() const;
    std::string encode(int index, int repeats=1) const; 
    std::string encode(bool neg, const ShSwizzle &swizzle) const; 

    /// TODO switch to fixed byte-length encodings so we don't
    /// need these whacky special characters 
    //
    /// For now, make all encodings human-readable so they will
    /// still be useful if we switch to byte encodings
    
    std::string encodeArray() const;

#ifdef SH_USE_MEMORY_POOL
    // Memory pool stuff.
    void* operator new(std::size_t size);
    void operator delete(void* d, std::size_t size);
#endif
    
  protected:
    DataType *m_begin; ///< Start of the data array
    DataType *m_end; ///< One after the end of the array

    bool m_managed; ///< true iff we are responsible for array alloc/delete 

    /// allocates an array of size N and sets m_begin, m_end
    void alloc(int N);

#ifdef SH_USE_MEMORY_POOL
    static ShPool* m_pool;
#endif
};

/// utility functions

// Cast to the specified data variant using dynamic_cast
//
// Refcounted and non-refcounted versions
//@{
template<typename T, ShDataType DT>
ShPointer<ShDataVariant<T, DT> > variant_cast(ShVariantPtr c);

template<typename T, ShDataType DT>
ShPointer<const ShDataVariant<T, DT> > variant_cast(ShVariantCPtr c);

template<typename T, ShDataType DT>
ShDataVariant<T, DT>* variant_cast(ShVariant* c);

template<typename T, ShDataType DT>
const ShDataVariant<T, DT>* variant_cast(const ShVariant* c);
// @}

// Make a copy of c cast to the requested type 
//@{
template<typename T, ShDataType DT>
ShPointer<ShDataVariant<T, DT> > variant_convert(ShVariantCPtr c);
// @}


}

#include "ShVariantImpl.hpp"

#endif
