/**
  @file ReferenceCount.h

  Reference Counting Garbage Collector for C++

  @maintainer Morgan McGuire, matrix@graphics3d.com
  @cite Adapted from Justin Miller's "RGC" class, as appeared in BYTE magazine.

  @created 2001-10-23
  @edited  2003-11-17

Example:

  <PRE>

class Foo : public G3D::ReferenceCountedObject {
public:
    int x;
    ~Foo() {
        printf("Deallocating 0x%x\n", this);
    }
};

typedef G3D::ReferenceCountedPointer<Foo> FooRef;

int main(int argc, char *argv[]) {

    FooRef a = new Foo();

    {
        FooRef b = a;
    }
    return 0
}
</PRE>
*/

#ifndef G3D_RGC_H
#define G3D_RGC_H

#include "G3D/debug.h"

namespace G3D {


/**
 Objects that are reference counted inherit from this.  Subclasses 
 <B>must</B> have a public destructor (the default destructor is fine)
 and publicly inherit ReferenceCountedObject.
 */
class ReferenceCountedObject {
public:

    /**
     The long name is to keep this from accidentally conflicting with
     a subclass's variable name.  Do not explicitly manipulate this value.
     */
    int ReferenceCountedObject_refCount;

protected:

    ReferenceCountedObject() : ReferenceCountedObject_refCount(0) {
        debugAssertM(isValidHeapPointer(this), 
            "Reference counted objects must be allocated on the heap.");
    }

public:

    virtual ~ReferenceCountedObject() {}

    /**
      Note: copies will initially start out with 0 
      references like any other object.
     */
    ReferenceCountedObject(const ReferenceCountedObject& notUsed) : 
        ReferenceCountedObject_refCount(0) {
        debugAssertM(isValidHeapPointer(this), 
            "Reference counted objects must be allocated on the heap.");
    }
};



/**
 Use ReferenceCountedPointer<T> in place of T* in your program.
 T must subclass ReferenceCountedObject.
 */
template <class T>
class ReferenceCountedPointer {
private:

    T*           pointer;

public:

    inline T* getPointer() const {
        return pointer;
    }

private:

    void registerReference() { 
        pointer->ReferenceCountedObject_refCount += 1;
    }


    int deregisterReference() {
        if (pointer->ReferenceCountedObject_refCount > 0) {
            pointer->ReferenceCountedObject_refCount -= 1;
        }

        return pointer->ReferenceCountedObject_refCount;
    }


    void zeroPointer() {
        if (pointer != NULL) {

            debugAssert(isValidHeapPointer(pointer));

            if (deregisterReference() <= 0) {
                // We held the last reference, so delete the object
                delete pointer;
            }

            pointer = NULL;
        }
    }


    void setPointer(T* x) {
        if (x != pointer) {
            if (pointer != NULL) {
                zeroPointer();
            }

            if (x != NULL) {
                debugAssert(isValidHeapPointer(x));

		        pointer = x;
		        registerReference();
            }
        }
    }

public:      

    /**
      Allow subtyping rule RCP&lt;<I>T</I>&gt; &lt;: RCP&lt;<I>S</I>&gt; if <I>T</I> &lt;: <I>S</I>
       (this could fail at runtime if the subtype relation is incorrect)
     */
    template <class S>
    inline ReferenceCountedPointer(const ReferenceCountedPointer<S>& p) : pointer(NULL) {
        setPointer(dynamic_cast<T*>(p.getPointer()));
    }


    inline ReferenceCountedPointer() : pointer(NULL) {}


    inline ReferenceCountedPointer(T* p) : pointer(NULL) { 
        setPointer(p); 
    }
    

    inline ReferenceCountedPointer(const ReferenceCountedPointer<T>& r) : pointer(NULL) { 
        setPointer(r.getPointer());
    }


    inline ~ReferenceCountedPointer() {
        zeroPointer();
    }
  

    inline ReferenceCountedPointer<T>& operator= (const ReferenceCountedPointer<T>& p) { 
        setPointer(p.getPointer());
        return *this; 
    }


    inline ReferenceCountedPointer<T>& operator= (T* x) {
        setPointer(x);
        return *this;
    }

    inline int operator== (const ReferenceCountedPointer<T> &y) const { 
        return (pointer == y.pointer); 
    }


    inline T& operator*() const {
        return (*pointer);
    }


    inline T* operator->() const {
        return pointer;
    }


    inline operator T*() const {
        return pointer;
    }


    inline bool isNull() const {
        return (pointer == NULL);
    }


    /**
     Returns true if this is the last reference to an object.
     Useful for flushing memoization caches-- a cache that holds the last
     reference is unnecessarily keeping an object alive.
     */
    inline int isLastReference() const {
        return (pointer->ReferenceCountedObject_refCount == 1);
    }
};

} // namespace

#endif





