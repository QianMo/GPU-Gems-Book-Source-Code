/*
    Crystal Space utility library: vector class interface
    Copyright (C) 1998,1999,2000 by Andrew Zabolotny <bit@eltech.ru>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#ifndef __CS_GARRAY_H__
#define __CS_GARRAY_H__

// Common macro for declarations below
#define CS_TYPEDEF_GROWING_ARRAY_EXT(Name, Type, ExtraConstructor, Extra) \
  class Name								\
  {									\
    typedef Type ga_type;						\
    ga_type *root;							\
    int limit;								\
    int length;								\
  public:								\
    int Limit () const							\
    { return limit; }							\
    void SetLimit (int iLimit)						\
    {									\
      if (limit == iLimit) return;					\
      if ((limit = iLimit)!=0)						\
        root = (ga_type *)realloc (root, limit * sizeof (ga_type));	\
      else								\
      { if (root) { free (root); root = NULL; } }			\
    }									\
    Name ()								\
    { limit = length = 0; root = NULL; ExtraConstructor; }		\
    ~Name ()								\
    { SetLimit (0); }							\
    int Length () const							\
    { return length; }							\
    void SetLength (int iLength, int iGrowStep = 8)			\
    {									\
      length = iLength;							\
      int newlimit = ((length + (iGrowStep - 1)) / iGrowStep) * iGrowStep;\
      if (newlimit != limit) SetLimit (newlimit);			\
    }									\
    ga_type &operator [] (int n)					\
    { CS_ASSERT (n >= 0 && n < limit); return root [n]; }		\
    const ga_type &operator [] (int n) const				\
    { CS_ASSERT (n >= 0 && n < limit); return root [n]; }		\
    ga_type &Get (int n)						\
    { CS_ASSERT (n >= 0 && n < limit); return root [n]; }		\
    void Delete (int n)							\
    { CS_ASSERT (n >= 0 && n < limit);					\
      memmove (root + n, root + n + 1, (limit - n - 1) * sizeof (ga_type)); \
      SetLength (length-1); }						\
    ga_type *GetArray ()						\
    { return root; }							\
    int Push (const ga_type &val, int iGrowStep = 8)			\
    {									\
      SetLength (length + 1, iGrowStep);				\
      memcpy (root + length - 1, &val, sizeof (ga_type));		\
      return length-1;							\
    }									\
    void Insert (int pos, const ga_type &val, int iGrowStep = 8)	\
    {									\
      CS_ASSERT (pos>=0 && pos<=length);				\
      SetLength (length + 1, iGrowStep);				\
      memmove (root+pos+1, root+pos, sizeof(ga_type) * (length-pos-1)); \
      memcpy (root + pos, &val, sizeof (ga_type));			\
    }									\
    Extra								\
  }

/**
 * This is a macro that will declare a growable array variable that is able to 
 * contain a number of elements of given type.<p>
 * Methods:
 * <ul>
 * <li>void SetLimit (int) - set max number of values the array can hold
 * <li>int Limit () - query max number of values the array can hold
 * <li>void SetLength (int) - set the amount of elements that are actually used
 * <li>int Length () - query the amount of elements that are actually used
 * <li>operator [] (int) - return a reference to Nth element of the array
 * </ul>
 * Usage examples:
 * <pre>
 * CS_TYPEDEF_GROWING_ARRAY (csLightArray, csLight*);
 * CS_TYPEDEF_GROWING_ARRAY (csIntArray, int);
 * static csLightArray la;
 * static csIntArray ia;
 * </pre>
 */
#define CS_TYPEDEF_GROWING_ARRAY(Name, Type)				\
  CS_TYPEDEF_GROWING_ARRAY_EXT (Name, Type, ;,  ;)

/**
 * Same as TYPEDEF_GROWING_ARRAY but contains additionally an reference
 * counter, so that the object can be shared among different clients.
 * If you do an IncRef each time you make use of it and an DecRef when you're 
 * done, the array will be automatically freed when there are no more 
 * references to it.<p> 
 * Methods:
 * <ul>
 * <li>void IncRef ()/void DecRef () - Reference counter management
 * </ul>
 */
#define CS_TYPEDEF_GROWING_ARRAY_REF(Name, Type)			\
  CS_TYPEDEF_GROWING_ARRAY_EXT (Name, Type, RefCount = 0,		\
    int RefCount;							\
    void IncRef ()							\
    { RefCount++; }							\
    void DecRef ()							\
    {									\
      if (RefCount == 1) SetLimit (0);					\
      RefCount--;							\
    })

/**
 * This is a shortcut for above to declare a dummy class and a single
 * instance of that class.
 * <p>
 * Usage examples:
 * <pre>
 * CS_DECLARE_GROWING_ARRAY (la, csLight*);
 * CS_DECLARE_GROWING_ARRAY (ia, int);
 * </pre>
 */
#ifdef CS_DECLARE_GROWING_ARRAY_REF
#undef CS_DECLARE_GROWING_ARRAY_REF
#endif

#define CS_DECLARE_GROWING_ARRAY(Name, Type)				\
  CS_TYPEDEF_GROWING_ARRAY(__##Name##_##Type,Type) Name

/**
 * Same as above but declares an object which has a reference counter.
 */
#define CS_DECLARE_GROWING_ARRAY_REF(Name, Type)			\
  CS_TYPEDEF_GROWING_ARRAY_REF(__##Name,Type) Name

#endif // __CS_GARRAY_H__
