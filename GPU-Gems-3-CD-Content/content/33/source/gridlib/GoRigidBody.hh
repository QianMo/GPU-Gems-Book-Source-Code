/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/

#ifndef  GORIGIDBODY_HH
#define  GORIGIDBODY_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbMatrix3.hh"
#include "GbQuaternion.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

template <class T>
class GRIDLIB_API GoRigidBody
{
public:
    // Construction and destruction.  The rigid body state is uninitialized.
    // Use the set functions to initialize the state before starting the
    // simulation.
    GoRigidBody ();
    virtual ~GoRigidBody ();

    // set rigid body state
    INLINE void setMass (T fMass);
    INLINE void setBodyInertia (const GbMatrix3<T>& rkInertia);
    INLINE void setPosition (const GbVec3<T>& rkPos);
    INLINE void setQOrientation (const GbQuaternion<T>& rkQOrient);
    INLINE void setLinearMomentum (const GbVec3<T>& rkLinMom);
    INLINE void setAngularMomentum (const GbVec3<T>& rkAngMom);
    INLINE void setROrientation (const GbMatrix3<T>& rkROrient);
    INLINE void setLinearVelocity (const GbVec3<T>& rkLinVel);
    INLINE void setAngularVelocity (const GbVec3<T>& rkAngVel);

    // get rigid body state
    INLINE T getMass () const;
    INLINE T getInverseMass () const;
    INLINE const GbMatrix3<T>& getBodyInertia () const;
    INLINE const GbMatrix3<T>& getBodyInverseInertia () const;
    INLINE GbMatrix3<T> getWorldInertia () const;
    INLINE GbMatrix3<T> getWorldInverseInertia () const;
    INLINE const GbVec3<T>& getPosition () const;
    INLINE const GbQuaternion<T>& getQOrientation () const;
    INLINE const GbVec3<T>& getLinearMomentum () const;
    INLINE const GbVec3<T>& getAngularMomentum () const;
    INLINE const GbMatrix3<T>& getROrientation () const;
    INLINE const GbVec3<T>& getLinearVelocity () const;
    INLINE const GbVec3<T>& getAngularVelocity () const;

    // force/torque function format
    typedef GbVec3<T> (*Function)
	(
	    T,                    // time of application
	    T,                    // mass
	    const GbVec3<T>&,    // position
	    const GbQuaternion<T>&, // orientation
	    const GbVec3<T>&,    // linear momentum
	    const GbVec3<T>&,    // angular momentum
	    const GbMatrix3<T>&,    // orientation
	    const GbVec3<T>&,    // linear velocity
	    const GbVec3<T>&     // angular velocity
	    );

    // force and torque functions
    Function Force;
    Function Torque;

    // Runge-Kutta fourth-order differential equation solver with fixed step size
    virtual void integrate(T fT, T fDT);

protected:
    // constant quantities (matrices in body coordinates)
    T mass_, invMass_;
    GbMatrix3<T> inertia_, invInertia_;

    // state variables
    GbVec3<T> position_;
    GbQuaternion<T> orientation_;
    GbVec3<T> linearMomentum_;
    GbVec3<T> angularMomentum_;

    // derived state variables
    GbMatrix3<T> worldTransform_;
    GbVec3<T> linearVelocity_;
    GbVec3<T> angularVelocity_;
};

#ifndef OUTLINE

#include "GoRigidBody.in"
#include "GoRigidBody.T" 

#else

INSTANTIATE( GoRigidBody<float> );
INSTANTIATE( GoRigidBody<double> ); 

#endif  // OUTLINE

#endif  // GORIGIDBODY_HH
