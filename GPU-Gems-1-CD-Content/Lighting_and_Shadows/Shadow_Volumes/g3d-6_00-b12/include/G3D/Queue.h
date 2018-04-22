/** 
  @file Queue.h
 
  @maintainer Morgan McGuire, graphics3d.com
 
  @created 2002-07-09
  @edited  2003-06-07
 */

#ifndef G3D_QUEUE_H
#define G3D_QUEUE_H

#include "G3D/debug.h"

namespace G3D {

/**
 Locate the ends of the two sections of the circular queue.
 */
#define FIND_ENDS \
    int firstEnd  = head + num;\
    int secondEnd = 0;\
    if (firstEnd > numAllocated) {\
       secondEnd = numAllocated - firstEnd;\
       firstEnd  = numAllocated;\
    }

/**
 Dynamic queue.
 */
template <class T>
class Queue {
private:
    //                
    //             |<----  num  ---->|
    // [  |  |  |  |  |  |  |  |  |  |  |  |  |  ]
    //              ^ 
    //              |
    //             head
    //
    //

    /**
     Only num elements are initialized.
     */
    T*  data;

    /**
     Index of the next element to be deque-d.
     */
    int head;

    /**
     Number of elements (including head) that are visible and initialized.
     */
    int num;
    
    /**
     Size of data.
     */
    int numAllocated;

    void _copy(const Queue& other) {
        data = (T*)malloc(sizeof(T) * other.numAllocated);
        debugAssert(data);

        FIND_ENDS;
	int i;
        for (i = head; i < firstEnd; ++i) {
            new (data + i)T();
            data[i] = other.data[i];
        }
        for (i = 0; i < secondEnd; ++i) {
            new (data + i)T();
            data[i] = other.data[i];
        }
    }


    /**
     Computes an array index from a queue position.
     */
    inline int index(int i) const {
        return (head + i + numAllocated) % numAllocated;
    }


    /**
     Allocates newSize elements and repacks the array.
     */
    void repackAndRealloc(int newSize) {
        data = (T*)realloc(data, newSize * sizeof(T));
        debugAssert(data != NULL);

        FIND_ENDS;

        if (secondEnd > 0) {
            int shift = newSize - numAllocated;
            // Move over the elements at the end.  Work
            // backwards so we don't overwrite the values
            // we're reading.  Must allow copy constructor
            // so explicitly iterate instead of memcpy-ing.

            // Call constructors on the exposed elements.
            int i;

            for (i = firstEnd; i < firstEnd + shift; ++i) {
                new (data + i)T();
            }

            for (i = firstEnd - 1; i >= head; --i) {
                data[i + shift] = data[i];
            }

            // Call the destructors of the now-unused elements.
            for (i = head; i < head + shift; ++i) {
                 (data + i)->~T();
            }

            // Shift the head (note num is unchanged)
            head += shift;
        }

        numAllocated = newSize;
    }

public:

    Queue() {
        numAllocated = 0;
        head         = 0;
        num          = 0;
        data         = NULL;
    }


    /**
    Copy constructor
    */
    Queue(const Queue& other) {
       _copy(other);
    }


   /**
    Destructor does not delete() the objects if T is a pointer type
    (e.g. T = int*) instead, it deletes the pointers themselves and 
    leaves the objects.  Call deleteAll if you want to dealocate
    the objects referenced.
    */
    virtual ~Queue() {
        clear();
    }

    /**
     Insert a new element into the front of the queue
     (a typical queue only uses pushBack).
     */
    inline void pushFront(const T& e) {
        if (num == numAllocated) {
            repackAndRealloc(numAllocated * 1.5 + 2);
        }

        int i = index(-1);
        head = i;
        // Call the constructor on the newly exposed element.
        new (data + i)T();
        data[head] = e;
        num++;
    }

    /**
    Insert a new element at the end of the queue.
    */
    inline void pushBack(const T& e) {
        if (num == numAllocated) {
            repackAndRealloc(numAllocated * 1.5 + 2);
        }

        int i = index(num);
        new (data + i)T();
        data[index(num)] = e;
        num++;
    }

    /**
     pushBack
     */
    inline void enqueue(const T& e) {
        pushBack(e);
    }


    /**
     Remove the last element from the queue.  The queue will never
     shrink in size.  (A typical queue only uses popFront).
     */
    inline T popBack() {
        int tail = index(num - 1);
        T result = data[tail];

        // Call the destructor
        data[tail].~T();
        --num;

        return result;
    }

    /**
    Remove the next element from the head of the queue.  The queue will never
    shrink in size. */
    inline T popFront() {
        T result = data[head];
        // Call the destructor
        data[head].~T();
        head = (head + 1) % numAllocated;
        --num;
        return result;
    }


   /**
    popFront
    */
   inline T dequeue() {
       return popFront();
   }

   /**
    Removes all elements.
    */
   void clear() {

       FIND_ENDS;
       
       // Invoke the destructors on the elements
       int i;
       for (i = head; i < firstEnd; ++i) {
           (data + i)->~T();
       }
       for (i = 0; i < secondEnd; ++i) {
           (data + i)->~T();
       }
       
       num = 0;
       numAllocated = 0;
       head = 0;
       free(data);
       data = NULL;
   }

   /**
    Assignment operator.
    */
   Queue& operator=(const Queue& other) {
       _copy(other);
       return *this;
   }

   /**
    Number of elements in the queue.
    */
   inline int size() const {
      return num;
   }

   /**
    Number of elements in the queue.
    */
   inline int length() const {
      return size();
   }

   /**
    Performs bounds checks in debug mode
    */
   inline T& operator[](int n) {
      debugAssert((n >= 0) && (n < num));
      return data[index(n)];
   }

   /**
    Performs bounds checks in debug mode
    */
    inline const T& operator[](int n) const {
        debugAssert((n >= 0) && (n < num));
        return data[index(n)];
    }


    /**
     Returns true if the given element is in the queue.
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
    Calls delete on all objects[0...size-1]
    and sets the size to zero.
    */
    void deleteAll() {
        FIND_ENDS;
        int i;
    	for (i = 0; i < secondEnd; ++i) {
            delete data[i];
        }
        for (i = head; i < firstEnd; ++i) {
            delete data[i];
        }
        clear();
    }
};

#undef FIND_ENDS

}; // namespace

#endif
