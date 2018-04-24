// -*- C++ -*- automatisch in C++mode wechseln (emacs)

template <class T>
GoRigidBody<T>::GoRigidBody ()
    : mass_(std::numeric_limits<T>::max())
    , invMass_(T(0.0))
    , inertia_(GbMatrix3<T>::IDENTITY)
    , invInertia_(GbMatrix3<T>::ZERO)
    , position_(GbVec3<T>::ZERO)
    , orientation_(GbQuaternion<T>::IDENTITY)
    , linearMomentum_(GbVec3<T>::ZERO)
    , angularMomentum_(GbVec3<T>::ZERO)
    , worldTransform_(GbMatrix3<T>::IDENTITY)
    , linearVelocity_(GbVec3<T>::ZERO)
    , angularVelocity_(GbVec3<T>::ZERO)
{
    // default body is immovable
}

template <class T>
GoRigidBody<T>::~GoRigidBody ()
{
}

// this is a RK4 fixed step size integrator
template <class T>
void 
GoRigidBody<T>::integrate(T fT, T fDT)
{
    T fHalfDT = T(0.5)*fDT;
    T fSixthDT = fDT/T(6.0);
    T fTpHalfDT = fT + fHalfDT;
    T fTpDT = fT + fDT;

    GbMatrix3<T> kNewROrient;

    // A1 = G(T,S0), B1 = S0 + (DT/2)*A1
    GbVec3<T> kA1DXDT(linearVelocity_);
    GbQuaternion<T> kW(T(0.0),angularVelocity_[0],angularVelocity_[1],angularVelocity_[2]);
    GbQuaternion<T> kA1DQDT(T(0.5)*kW*orientation_);
    GbVec3<T> kA1DPDT(Force(fT,mass_,position_,orientation_,linearMomentum_,angularMomentum_,worldTransform_,linearVelocity_,angularVelocity_));
    GbVec3<T> kA1DLDT(Torque(fT,mass_,position_,orientation_,linearMomentum_,angularMomentum_,worldTransform_,linearVelocity_,angularVelocity_));
    GbVec3<T> kNewPos(position_ + fHalfDT*kA1DXDT);
//     debugmsg("new pos "<<kNewPos);
    GbQuaternion<T> kNewQOrient(orientation_ + fHalfDT*kA1DQDT);
    GbVec3<T> kNewLinMom(linearMomentum_ + fHalfDT*kA1DPDT);
    GbVec3<T> kNewAngMom(angularMomentum_ + fHalfDT*kA1DLDT);
    kNewQOrient.toRotationMatrix(kNewROrient);
    GbVec3<T> kNewLinVel(invMass_*kNewLinMom);
    GbVec3<T> kNewAngVel(kNewROrient*invInertia_*kNewROrient.transpose()*kNewAngMom);

    // A2 = G(T+DT/2,B1), B2 = S0 + (DT/2)*A2
    GbVec3<T> kA2DXDT(kNewLinVel);
    kW = GbQuaternion<T>(T(0.0),kNewAngVel[0],kNewAngVel[1],kNewAngVel[2]);
    GbQuaternion<T> kA2DQDT(T(0.5)*kW*kNewQOrient);
    GbVec3<T> kA2DPDT(Force(fTpHalfDT,mass_,kNewPos,kNewQOrient,kNewLinMom,kNewAngMom,kNewROrient,kNewLinVel,kNewAngVel));
    GbVec3<T> kA2DLDT(Torque(fTpHalfDT,mass_,kNewPos,kNewQOrient,kNewLinMom,kNewAngMom,kNewROrient,kNewLinVel,kNewAngVel));
    kNewPos = position_ + fHalfDT*kA2DXDT;
    kNewQOrient = orientation_ + fHalfDT*kA2DQDT;
    kNewLinMom = linearMomentum_ + fHalfDT*kA2DPDT;
    kNewAngMom = angularMomentum_ + fHalfDT*kA2DLDT;
    kNewQOrient.toRotationMatrix(kNewROrient);
    kNewLinVel = invMass_*kNewLinMom;
    kNewAngVel = kNewROrient*invInertia_*kNewROrient.transpose()*kNewAngMom;

    // A3 = G(T+DT/2,B2), B3 = S0 + DT*A3
    GbVec3<T> kA3DXDT(kNewLinVel);
    kW = GbQuaternion<T>(T(0.0),kNewAngVel[0],kNewAngVel[1],kNewAngVel[2]);
    GbQuaternion<T> kA3DQDT(T(0.5)*kW*kNewQOrient);
    GbVec3<T> kA3DPDT(Force(fTpHalfDT,mass_,kNewPos,kNewQOrient,kNewLinMom,kNewAngMom,kNewROrient,kNewLinVel,kNewAngVel));
    GbVec3<T> kA3DLDT(Torque(fTpHalfDT,mass_,kNewPos,kNewQOrient,kNewLinMom,kNewAngMom,kNewROrient,kNewLinVel,kNewAngVel));
    kNewPos = position_ + fDT*kA3DXDT;
    kNewQOrient = orientation_ + fDT*kA3DQDT;
    kNewLinMom = linearMomentum_ + fDT*kA3DPDT;
    kNewAngMom = angularMomentum_ + fDT*kA3DLDT;
    kNewQOrient.toRotationMatrix(kNewROrient);
    kNewLinVel = invMass_*kNewLinMom;
    kNewAngVel = kNewROrient*invInertia_*kNewROrient.transpose()*kNewAngMom;

    // A4 = G(T+DT,B3), S1 = S0 + (DT/6)*(A1+2*(A2+A3)+A4)
    GbVec3<T> kA4DXDT(kNewLinVel);
    kW = GbQuaternion<T>(T(0.0),kNewAngVel[0],kNewAngVel[1],kNewAngVel[2]);
    GbQuaternion<T> kA4DQDT(T(0.5)*kW*kNewQOrient);
    GbVec3<T> kA4DPDT(Force(fTpDT,mass_,kNewPos,kNewQOrient,kNewLinMom,kNewAngMom,kNewROrient,kNewLinVel,kNewAngVel));
    GbVec3<T> kA4DLDT(Torque(fTpDT,mass_,kNewPos,kNewQOrient,kNewLinMom,kNewAngMom,kNewROrient,kNewLinVel,kNewAngVel));
    position_ = position_ + fSixthDT*(kA1DXDT + T(2.0)*(kA2DXDT + kA3DXDT) + kA4DXDT);
    orientation_ = orientation_ + fSixthDT*(kA1DQDT + T(2.0)*(kA2DQDT + kA3DQDT) + kA4DQDT);
    linearMomentum_ = linearMomentum_ + fSixthDT*(kA1DPDT + T(2.0)*(kA2DPDT + kA3DPDT) + kA4DPDT);
    angularMomentum_ = angularMomentum_ + fSixthDT*(kA1DLDT + T(2.0)*(kA2DLDT + kA3DLDT) + kA4DLDT);
    orientation_.toRotationMatrix(worldTransform_);
    linearVelocity_ = invMass_*linearMomentum_;
    angularVelocity_ = worldTransform_*invInertia_*worldTransform_.transpose()*angularMomentum_;
}
