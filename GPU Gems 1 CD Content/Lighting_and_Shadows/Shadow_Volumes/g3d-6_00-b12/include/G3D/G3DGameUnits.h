/**
 @file G3DGameUnits.h

 @maintainer Morgan McGuire, matrix@graphics3d.com
 @created 2002-10-05
 @edited  2002-11-26
 */

#ifndef G3D_GAMEUNITS_H
#define G3D_GAMEUNITS_H

namespace G3D {
/**
 Time, in seconds.
 */
typedef double GameTime;
typedef double SimTime;

/**
 Actual wall clock time in seconds.
 */
typedef double RealTime;

enum AMPM {AM, PM};

#define SECOND              (1.0)
#define MINUTE              (60 * SECOND)
#define HOUR                (60 * MINUTE)
#define DAY                 (24 * HOUR)
#define SUNRISE             (DAY / 4)
#define SUNSET              (DAY * 3 / 4)
#define MIDNIGHT            (0)

#define CENTIMETER          (0.01)
#define DECIMETER           (0.1)
#define METER               (1.0)
#define KILOMETER       (1000.0)

/**
 Converts a 12 hour clock time into the number of seconds since 
 midnight.  Note that 12:00 PM is noon and 12:00 AM is midnight.

 Example: <CODE>toSeconds(10, 00, AM)</CODE>
 */
SimTime toSeconds(int hour, int minute, double seconds, AMPM ap);
SimTime toSeconds(int hour, int minute, AMPM ap);

}

using G3D::toSeconds;
#endif
