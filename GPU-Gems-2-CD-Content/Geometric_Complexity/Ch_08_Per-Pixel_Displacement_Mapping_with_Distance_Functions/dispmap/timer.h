/* 
  Copyright (C) 2003, Kevin Moule (krmoule@cgl.uwaterloo.ca)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef __COMMON_TIMER_H__
#define __COMMON_TIMER_H__

#ifdef WIN32
#include <windows.h>
#else /* WIN32 */
#include <sys/time.h>
#endif /* WIN32 */

#include <iostream>

//-------------------------------------------------------------------
// mtimer_t 
//-------------------------------------------------------------------
class mtimer_t
  {
  public:
    mtimer_t(void);
    ~mtimer_t(void);

    static mtimer_t zero(void);
    static mtimer_t now(void);

    float value() const;

  public:
    friend mtimer_t operator+(const mtimer_t& a, const mtimer_t& b);
    friend mtimer_t operator-(const mtimer_t& a, const mtimer_t& b);
    friend std::ostream& operator<<(std::ostream& s, const mtimer_t& t);

  private:
#ifdef WIN32
    LARGE_INTEGER t;
#else
    struct timeval t;
#endif /* WIN32 */
  };

#endif /* __COMMON_TIMER_H__ */
