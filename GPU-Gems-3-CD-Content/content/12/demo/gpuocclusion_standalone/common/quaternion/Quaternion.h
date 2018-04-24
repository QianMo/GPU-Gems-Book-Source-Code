/*! \file Quaternion.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing
 *         a quaternion.
 */

#ifndef QUATERNION_H
#define QUATERNION_H

#include <gpcpu/Vector.h>
#include <gpcpu/floatmxn.h>

class Quaternion
  : public float4
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef float4 Parent;

    /*! Null constructor calls the Parent.
     */
    inline Quaternion(void);

    /*! Constructor interprets a float4 as a Quaternion.
     *  \param v The quaternion to set this Quaternion to.
     */
    inline Quaternion(const float4 &v);

    /*! Constructor accepts an angle in radians, and an axis of rotation.
     *  \param angle The rotation angle in radians.
     *  \param axis The axis of rotation.
     */
    inline Quaternion(const float angle, const float3 &axis);

    /*! This method returns the representation of this Quaternion
     *  as a 4x4 rotation matrix.
     *  \return The 4x4 matrix representing this Quaternion.
     */
    inline float4x4 getRotationMatrix(void) const;

    /*! Quaternion multiply equal.
     *  \param rhs The right hand side of the product.
     *  \return *this
     */
    inline Quaternion &operator*=(const Quaternion &rhs);

    /*! Quaternion product.
     *  \param rhs The right hand side of the product.
     *  \return (*this) * rhs
     */
    inline Quaternion operator*(const Quaternion &rhs) const;
}; // end Quaternion

#include "Quaternion.inl"

#endif // QUATERNION_H

