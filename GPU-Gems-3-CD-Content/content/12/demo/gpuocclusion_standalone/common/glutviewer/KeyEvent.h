/*! \file KeyEvent.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a KeyEvent
 *         class providing a similar interface to Qt's.
 */

#ifndef KEY_EVENT_H
#define KEY_EVENT_H

class KeyEvent
{
  public:
    /*! Constructor accepts the ascii code of a key
     *  and a bitfield describing the modifiers.
     *  \param k Sets mKey.
     *  \param m Sets mModifiers.
     */
    inline KeyEvent(const int k, const int m);

    /*! This method returns mKey.
     *  \return mKey.
     */
    inline int key(void) const;

    /*! This method returns mModifiers
     *  \return mModifiers
     */
    inline int modifiers(void) const;

  protected:
    int mKey;
    int mModifiers;
}; // end KeyEvent

#include "KeyEvent.inl"

#endif // KEY_EVENT_H

