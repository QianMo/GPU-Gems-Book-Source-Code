/** 
  @file Array.h
 
  @maintainer Morgan McGuire, graphics3d.com
  @cite Portions written by Aaron Orenstein, a@orenstein.name
 
  @created 2001-03-11
  @edited  2004-01-13

  Copyright 2000-2004, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_ARRAY_H
#define G3D_ARRAY_H

#include "G3D/platform.h"
#include "G3D/debug.h"
#include "G3D/System.h"
#include <algorithm>

#ifdef G3D_WIN32
    #include <new.h>
#endif


namespace G3D {

/**
 Constant for passing to Array::resize
 */
const bool DONT_SHRINK_UNDERLYING_ARRAY = false;

/** Constant for Array::sort */
const int SORT_INCREASING = 1;
/** Constant for Array::sort */
const int SORT_DECREASING = -1;

/**
 Dynamic 1D array.  

 Objects must have a default constructor (constructor that
 takes no arguments) in order to be used with this template.
 You will get the error "no appropriate default constructor found"
 if they do not.

 If SSE is defined Arrays allocate the first element aligned to
 16 bytes.

 Unlike std::vector, Array is optimized for graphics use.  The default
 array takes up zero heap space.  The first resize (or append)
 operation grows it to a reasonable internal size so it is efficient
 to append to small arrays.  Memory is allocated using
 System::alignedMalloc, which produces pointers aligned to 16-byte
 boundaries for use with SSE instructions.  When Array needs to copy
 data internally on a resize operation it correctly invokes copy
 constructors of the elements (the Microsoft implementation of
 std::vector uses realloc, which can create memory leaks for classes
 containing references and pointers).  Array provides a guaranteed
 safe way to access the underlying data as a flat C array --
 Array::getCArray.  Although (T*)std::vector::begin() can be used for
 this purpose, it is not guaranteed to succeed on all platforms.


 Do not subclass an Array.
 */
template <class T>
class Array {
private:
    /** 0...num-1 are initialized elements, num...numAllocated-1 are not */
    T*              data;

    int             num;
    int             numAllocated;

    void init(int n, int a) {
        debugAssert(n <= a);
        debugAssert(n >= 0);
        this->num = 0;
        this->numAllocated = 0;
        data = NULL;
        if (a > 0) {
            resize(n);
        } else {
            data = NULL;
        }
    }

    void _copy(const Array &other) {
        init(other.num, other.num);
        for (int i = 0; i < num; i++) {
            data[i] = other.data[i];
        }
    }

    /**
     Returns true iff address points to an element of this array.
     Used by append.
     */
    inline bool inArray(const T* address) {
        return (address >= data) && (address < data + num);
    }


    /** Only compiled if you use the sort procedure. */
    static bool __cdecl compareGT(const T& a, const T& b) {
        return a > b;
    }


    /**
     Allocates a new array of size numAllocated and copys at most
     oldNum elements from the old array to it.  Destructors are
     called for oldNum elements of the old array.
     */
    void realloc(int oldNum) {
         T* oldData = data;
         
         data = (T*)System::alignedMalloc(sizeof(T) * numAllocated, 16);
         // Call the copy constructors
         int i;
         for (i = iMin(oldNum, numAllocated) - 1; i >= 0; --i) {
             new (data + i) T(oldData[i]);
         }

         // Call destructors
         for (i = oldNum - 1; i >= 0; --i) {
            (oldData + i)->~T();
         }

         System::alignedFree(oldData);
    }

public:

    /**
     C++ STL style iterator variable.  Call begin() to get 
     the first iterator, pre-increment (++i) the iterator to get to
     the next value.  Use dereference (*i) to access the element.
     */
    typedef T* Iterator;
    typedef const T* ConstIterator;

    /**
     C++ STL style iterator method.  Returns the first iterator element.
     Do not change the size of the array while iterating.
     */
    Iterator begin() {
        return data;
    }

    ConstIterator begin() const {
        return data;
    }
    /**
     C++ STL style iterator method.  Returns one after the last iterator
     element.
     */
    ConstIterator end() const {
        return data + num;
    }

    Iterator end() {
        return data + num;
    }

   /**
    The array returned is only valid until the next append() or resize call, or 
	the Array is deallocated.
    */
   T* getCArray() {
       return data;
   }

   /**
    The array returned is only valid until the next append() or resize call, or 
	the Array is deallocated.
    */
   const T* getCArray() const {
       return data;
   }

   /** Creates a zero length array (no heap allocation occurs until resize). */
   Array() {
       init(0, 0);
   }

   /**
    Creates an array of size.
    */
   Array(int size) {
       init(size, size);
   }

   /**
    Copy constructor
    */
   Array(const Array& other) {
       _copy(other);
   }

   /**
    Destructor does not delete() the objects if T is a pointer type
    (e.g. T = int*) instead, it deletes the pointers themselves and 
    leaves the objects.  Call deleteAll if you want to dealocate
    the objects referenced.
    */
   ~Array() {
       // Invoke the destructors on the elements
       for (int i = 0; i < num; i++) {
           (data + i)->~T();
       }
       
       System::alignedFree(data);
       data = NULL;
   }


   /**
    Removes all elements.  Use resize(0, false) if you want to 
    remove all elements without deallocating the underlying array
    so that future append() calls will be faster.
    */
   void clear() {
       resize(0);
   }

   /**
    Assignment operator.
    */
   Array& operator=(const Array& other) {
       resize(other.num);
       for (int i = 0; i < num; i++) {
           data[i] = other[i];
       }
       return *this;
   }

   /**
    Number of elements in the array.
    */
   inline int size() const {
      return num;
   }

   /**
    Number of elements in the array.
    */
   inline int length() const {
      return size();
   }

   /**
    Swaps element index with the last element in the array then
    shrinks the array by one.
    */
   void fastRemove(int index) {
       debugAssert(index >= 0);
       debugAssert(index < num);
       data[index] = data[num - 1];
       resize(size() - 1);
   }

   /**
    Resizes, calling the default constructor for 
    newly created objects and shrinking the underlying
    array as needed (and calling destructors as needed).
    */
   void resize(int n) {
      resize(n, true);
   }


   void resize(int n, bool shrinkIfNecessary) {
      int oldNum = num;
      num = n;

      if (num < oldNum) {
          // Call the destructors on newly hidden elements
          for (int i = num; i < oldNum; i++) {
             (data + i)->~T();
          }
      }

      // Allocate 8 elements or 32 bytes, whichever is higher.
      const int minSize = iMax(8, 32 / sizeof(T));

      if (num > numAllocated) {
         
         // Increase the underlying size of the array
         numAllocated = (num - numAllocated) + (int)(numAllocated * 1.4) + 16;

         if (numAllocated < minSize) {
             numAllocated = minSize;
         }

         realloc(oldNum);

      } else if ((num <= numAllocated / 2) && shrinkIfNecessary && (numAllocated > minSize)) {

          // Only copy over old elements that still remain after resizing
          // (destructors were called for others if we're shrinking)
          realloc(iMin(num, oldNum));

      }

      // Call the constructors on newly revealed elements.
      for (int i = oldNum; i < num; i++) {
          new (data + i) T();
      }
   }

    /**
     Add an element to the end of the array.  Will not shrink the underlying array
     under any circumstances.  It is safe to append an element that is already
     in the array.
     */
    inline void append(const T& value) {
        
        if (num < numAllocated) {
            // This is a simple situation; just stick it in the next free slot using
            // the copy constructor.
            new (data + num) T(value);
            ++num;
        } else if (inArray(&value)) {
            // The value was in the original array; resizing
            // is dangerous because it may move the value
            // we have a reference to.
            T tmp = value;
            append(tmp);
        } else {
            resize(num + 1, DONT_SHRINK_UNDERLYING_ARRAY);
            data[num - 1] = value;
        }
    }


    inline void append(const T& v1, const T& v2) {
        if (inArray(&v1) || inArray(&v2)) {
            T t1 = v1;
            T t2 = v2;
            append(t1, t2);
        } else {
            resize(num + 2, DONT_SHRINK_UNDERLYING_ARRAY);
            data[num - 2] = v1;
            data[num - 1] = v2;
        }
    }


    inline void append(const T& v1, const T& v2, const T& v3) {
        if (inArray(&v1) || inArray(&v2) || inArray(&v3)) {
            T t1 = v1;
            T t2 = v2;
            T t3 = v3;
            append(t1, t2, t3);
        } else {
            resize(num + 3, DONT_SHRINK_UNDERLYING_ARRAY);
            data[num - 3] = v1;
            data[num - 2] = v2;
            data[num - 1] = v3;
        }
    }


    inline void append(const T& v1, const T& v2, const T& v3, const T& v4) {
        if (inArray(&v1) || inArray(&v2) || inArray(&v3) || inArray(&v4)) {
            T t1 = v1;
            T t2 = v2;
            T t3 = v3;
            T t4 = v4;
            append(t1, t2, t3, t4);
        } else {
            resize(num + 4, DONT_SHRINK_UNDERLYING_ARRAY);
            data[num - 4] = v1;
            data[num - 3] = v2;
            data[num - 2] = v3;
            data[num - 1] = v4;
        }
    }

    /**
     Returns true if the given element is in the array.
     */
    bool contains(const T& e) const {
        for (int i = 0; i < size(); ++i) {
            if ((*this)[i] == e) {
                return true;
            }
        }

        return false;
    }

   /**
    Append the elements of array.  Cannot be called with this array
    as an argument.
    */
   void append(const Array<T>& array) {
       debugAssert(this != &array);
       int oldNum = num;
       int arrayLength = array.length();

       resize(num + arrayLength, false);

       for (int i = 0; i < arrayLength; i++) {
           data[oldNum + i] = array.data[i];
       }
   }

   /**
    Pushes a new element onto the end and returns its address.
    This is the same as A.resize(A.size() + 1); A.last()
    */
   inline T& next() {
       resize(num + 1);
       return last();
   }

   /**
    Pushes an element onto the end (appends)
    */
   inline void push(const T& value) {
       append(value);
   }

   inline void push(const Array<T>& array) {
       append(array);
   }

   /**
    Removes the last element and returns it.
    */
   inline T pop(bool shrinkUnderlyingArrayIfNecessary = false) {
       debugAssert(num > 0);
       T temp = data[num - 1];
       resize(num - 1, shrinkUnderlyingArrayIfNecessary);
       return temp;
   }

   /**
    Performs bounds checks in debug mode
    */
   inline T& operator[](int n) {
      debugAssert((n >= 0) && (n < num));
      return data[n];
   }

   /**
    Performs bounds checks in debug mode
    */
    inline const T& operator[](int n) const {
        debugAssert((n >= 0) && (n < num));
        return data[n];
    }

   /**
    Returns the last element, performing a check in
    debug mode that there is at least one element.
    */
    inline const T& last() const {
        debugAssert(num > 0);
        return data[num - 1];
    }

    inline T& last() {
        debugAssert(num > 0);
        return data[num - 1];
    }

   /**
    Calls delete on all objects[0...size-1]
    and sets the size to zero.
    */
    void deleteAll() {
        for (int i = 0; i < num; i++) {
            delete(data[i]);
        }
        resize(0);
    }

    /**
     Returns the index of (the first occurance of) an index or -1 if
     not found.
     */
    int findIndex(const T& value) const {
        for (int i = 0; i < num; ++i) {
            if (data[i] == value) {
                return i;
            }
        }
        return -1;
    }

    /**
     Finds an element and returns the iterator to it.  If the element
     isn't found then returns end().
     */
    Iterator find(const T& value) {
        for (int i = 0; i < num; ++i) {
            if (data[i] == value) {
                return data + i;
            }
        }
        return end();
    }

    ConstIterator find(const T& value) const {
        for (int i = 0; i < num; ++i) {
            if (data[i] == value) {
                return data + i;
            }
        }
        return end();
    }

    /**
     Removes count elements from the array
     referenced either by index or Iterator.
     */
    void remove(Iterator element, int count = 1) {
        debugAssert((element >= begin()) && (element < end()));
        debugAssert((count > 0) && (element + count) <= end());
        Iterator last = end() - count;

        while(element < last) {
            element[0] = element[count];
            ++element;
        }
        
        resize(num - count);
    }

    void remove(int index, int count = 1) {
        debugAssert((index >= 0) && (index < num));
        debugAssert((count > 0) && (index + count <= num));
        
        remove(begin() + index, count);
    }

    /**
     Reverse the elements of the array in place.
     */
    void reverse() {
        T temp;
        
        int n2 = num / 2;
        for (int i = 0; i < n2; ++i) {
            temp = data[num - 1 - i];
            data[num - 1 - i] = data[i];
            data[i] = temp;
        }
    }

    void sort(bool (__cdecl *lessThan)(const T& elem1, const T& elem2)) {
        std::sort(data, data + num, lessThan);
    }


    /**
     Sorts the array in increasing order using the > or < operator.  To 
     invoke this method on Array<T>, T must override those operator.
     You can overide these operators as follows:
     <code>
        bool T::operator>(const T& other) const {
           return ...;
        }
        bool T::operator<(const T& other) const {
           return ...;
        }
     </code>
     */
    void sort(int direction=SORT_INCREASING) {
        if (direction == SORT_INCREASING) {
            std::sort(data, data + num);
        } else {
            std::sort(data, data + num, compareGT);
        }
    }

    /**
     Sorts elements beginIndex through and including endIndex.
     */
    void sortSubArray(int beginIndex, int endIndex, int direction=SORT_INCREASING) {
        if (direction == SORT_INCREASING) {
            std::sort(data + beginIndex, data + endIndex + 1);
        } else {
            std::sort(data + beginIndex, data + endIndex + 1, compareGT);
        }
    }

    void sortSubArray(int beginIndex, int endIndex, bool (__cdecl *lessThan)(const T& elem1, const T& elem2)) {
        std::sort(data + beginIndex, data + endIndex + 1, lessThan);
    }

    /** Redistributes the elements so that the new order is statistically independent
        of the original order. O(n) time.*/
    void randomize() {
        Array<T> original = *this;
        for (int i = size() - 1; i >= 0; --i) {
            int x = iRandom(0, i);
            data[i] = original[x];
            original.fastRemove(x);
        }
    }

};

} // namespace

#endif

