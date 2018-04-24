/*! \file KeyEvent.inl
 *  \author Jared Hoberock
 *  \brief Inline file for KeyEvent.h.
 */

#include "KeyEvent.h"

KeyEvent
  ::KeyEvent(const int k, const int m)
    :mKey(k),mModifiers(m)
{
  ;
} // end KeyEvent::KeyEvent()

int KeyEvent
  ::key(void) const
{
  return mKey;
} // end KeyEvent::key()

int KeyEvent
  ::modifiers(void) const
{
  return mModifiers;
} // end KeyEvent::modifiers()

