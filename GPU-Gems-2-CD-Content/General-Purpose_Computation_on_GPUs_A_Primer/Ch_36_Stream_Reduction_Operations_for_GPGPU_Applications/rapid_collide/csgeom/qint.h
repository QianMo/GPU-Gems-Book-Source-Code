/*
    Copyright (C) 1998-2000 by Andrew Zabolotny
  
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

#ifndef __QINT_H__
#define __QINT_H__

#if defined (CS_IEEE_DOUBLE_FORMAT)

/**
    Quick floating point to integer conversions.
    <p>
    There is a general trick that can be used on all FPUs that uses IEEE
    double-precision floating-point number format (theoretically it should
    cover all CPUs that have a "double" data type of 64 bits; it is applicable
    for other types as well, but you have to figure out yourself how - it's
    easy if you understand everything that is written below) to quickly
    convert numbers from floating-point format to integer and fixed-point
    formats. Let's take the "double" type format (on x86, e.g. little-endian):

<pre>
 bit 0        8        16       24       32       40       48       56   sign
    +--------+--------+--------+--------+--------+--------+--------+-------|+
    |mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmeeee|eeeeeees|
    +|-------+--------+--------+--------+--------+--------+---||---+------|-+
     \------------------------- mantissa ---------------------/\-exponent-/
</pre>

    In the following we'll picture numbers as (s)m^e, e.g. (0)2^1 is the
    number '2', (1)2^5 means number -32.
    This means that the number "127" will be represented as

<pre>
    (0)2^7 + 2^6 + 2^5 + ... + 2^0, or

    00000000.00000000.00000000.00000000.00000000.00000111.1111|1110.00000000
</pre>

    (well, in general exponent is stored + 0x3ff but we will not take this
    into account... consider we'll always subtract 0x3ff from exponent).

    So observe several things. First, the exponent is 111b (e.g. 7) - the
    most significant power of two in above series. Next, observe that the
    mantissa contains just 7 '1' bits while 255 contains 8 '1' bits. Well,
    the topmost bit is always considered '1' except for the case of 'zero'.

    Now what if we add, say, 2^20 to the above number (i.e. 255+2^20)?
    The most significant power of two is 20 then, so all bits with
    lower significance will just shift left by 20 bits:

<pre>
    (0)2^20 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0, or

    00000000.00000000.00000000.00000000.11111111.00000000.0000|0010.10000000
</pre>

    So what we see? Our beloved 255 has moved to the left and we can just pick
    it by taking the fifth byte of the floating-point number. So if we need
    to convert any number from 0 to 2^20-1 from floaring-point format to
    integer format, we can just add 2^20 to him and pick the respective
    bytes from the "double" variable. If we would need a fixed-point value,
    we also can take the bytes prior to integer value (they will contain
    the fractional part). So if we would need a 16.16 format, we can add
    2^20 and take bits 16-31 (fractional part) and 23-48 (integer part).

    Now what we will do with negative numbers? The problem is that when
    you 'shift' left an negative number by adding, say, 2^20, the most
    significant bits are filled with '1' rather than with '0' because
    your number is subtracted from 2^20 rather than added. Thus 2^20-1
    will look like this:

<pre>
    00000000.00000000.00000000.00000000.01111111.11111111.1111|1100.10000000
</pre>

    Thus if we'll get the fifth byte as before, we'll get '-2' rather than
    '-1'. So we would have to check whenever we should shift the result
    or not depending on his sign. If rounding number (instead of truncating
    the fractional part) is acceptable, we can do another trick: add
    2^20 + 2^8, and after taking the required subpart subtract 2^8 back.

    Measurements show that this trick is even faster (at least on P5 and
    above) than FISTP command (even if we use the (correct) presumption
    that FPU is in rounding mode by default, e.g. no need to toggle FPU
    control word). QRound executes 2 clocks on Celeron while FISTP
    executes 13 clocks, i.e. about 6 times slower.

    All said above is true for all CPU types that uses standard IEEE
    format for "double" type, with the respective corrections for endianess.
*/

#ifdef CS_BIG_ENDIAN
#  define CS_LOWER_WORD_BYTE	4
#else
#  define CS_LOWER_WORD_BYTE	0
#endif

#define CS_LONG_AT_BYTE(x,b)	*(long *)(((char *)&x) + b)

/**
 * We'll add 2^36 to create a 32.16 fixed-point value and then will pick
 * just the integer part from that. This will allow us to work correctly
 * with numbers up to 1 - 1/2^16 fractional part; numbers above will be
 * rounded up (because by default FPU is in rounding mode).
 */
#define FIST_MAGIC_QINT   (65536.0 * 65536.0 * 16.0)

/// Truncate the fractional part of a floating-point value and convert to integer
/**
 * well, for QInt gcc 2.96 and above compilers (we tested up to 3.0.1) seems to
 * be buggy, we've got a workaround which seems to be buggy in vc :) so we have
 * to differentiate between compilers here (Matze)
 */
#ifdef CS_QINT_WORKAROUND
static inline long QInt (double inval)
{
  union { double dtemp; long result; } x;

  x.dtemp = FIST_MAGIC_QINT + inval;
  x.result = CS_LONG_AT_BYTE (x.dtemp, 2);
  return x.result < 0 ? (x.result >> 1) + 1 : x.result;
}
#else
static inline long QInt (double inval)
{
  double dtemp = FIST_MAGIC_QINT + inval;
  // Note that on both low-endian (x86) and big-endian (m68k) we have
  // to shift by two bytes. So no need for an #ifdef.
  long result = CS_LONG_AT_BYTE (dtemp, 2);
  return result < 0 ? (result >> 1) + 1 : result;
}
#endif

/**
 * To round an floating-point value we'll add 2^52+0.5*2^32 to it,
 * and will subtract 2^32 back after taking the required bits out.
 */
#define FIST_MAGIC_QROUND (((65536.0 * 65536.0 * 16.0) + (65536.0 * 0.5)) * 65536.0)

/// Round a floating-point value and convert to integer
static inline long QRound (double inval)
{
  double dtemp = FIST_MAGIC_QROUND + inval;
  return CS_LONG_AT_BYTE (dtemp, CS_LOWER_WORD_BYTE) - 0x80000000;
}

/**
 * Add this constant to convert a floating-point value to 24.8
 * fixed-point value.
 */
#define FIST_MAGIC_QINT8 (((65536.0 * 16.0) + 0.5) * 65536.0 * 256.0)

/// Convert a floating-point number to 24.8 fixed-point value.
inline long QInt8 (float inval)
{
  double dtemp = FIST_MAGIC_QINT8 + inval;
  return CS_LONG_AT_BYTE (dtemp, CS_LOWER_WORD_BYTE) - 0x80000000;
}

/**
 * Add this constant to convert a floating-point value to 16.16
 * fixed-point value.
 */
#define FIST_MAGIC_QINT16 (((65536.0 * 16.0) + 0.5) * 65536.0)

/// Convert a floating-point number to 16.16 fixed-point value.
inline long QInt16 (float inval)
{
  double dtemp = FIST_MAGIC_QINT16 + inval;
  return CS_LONG_AT_BYTE (dtemp, CS_LOWER_WORD_BYTE) - 0x80000000;
}

/**
 * Add this constant to convert a floating-point value to 8.24
 * fixed-point value.
 */
#define FIST_MAGIC_QINT24 (((65536.0 * 16.0) + 0.5) * 256.0)

inline long QInt24 (float inval)
{
  double dtemp = FIST_MAGIC_QINT24 + inval;
  return CS_LONG_AT_BYTE (dtemp, CS_LOWER_WORD_BYTE) - 0x80000000;
}
    
#else /* not CS_IEEE_DOUBLE_FORMAT */

#define QRound(x) (int ((x) + ((x < 0) ? -0.5 : +0.5)))
#define QInt(x)   (int (x))
#define QInt8(x)  (int ((x)*256.))
#define QInt16(x) (int ((x)*65536.))
#define QInt24(x) (int ((x)*16777216.))

#endif /* CS_IEEE_DOUBLE_FORMAT */

#endif // __QINT_H__
