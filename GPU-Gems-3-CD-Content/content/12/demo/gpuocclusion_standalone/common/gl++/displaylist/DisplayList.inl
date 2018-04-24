/*! \file DisplayList.inl
 *  \author Jared Hoberock
 *  \brief Inline file for DisplayList.h.
 */

DisplayList::DisplayList(void):Parent()
{
  setTarget(GL_COMPILE);
} // end DisplayList::DisplayList()

void DisplayList::call(void) const
{
  glCallList(getIdentifier());
} // end DisplayList::call()

void DisplayList::operator()(void) const
{
  call();
} // end DisplayList::operator()()

