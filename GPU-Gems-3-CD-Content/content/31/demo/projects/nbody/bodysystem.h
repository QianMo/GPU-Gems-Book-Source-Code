#ifndef __BODYSYSTEM_H__
#define __BODYSYSTEM_H__

enum NBodyConfig
{
	NBODY_CONFIG_RANDOM,
    NBODY_CONFIG_SHELL,
    NBODY_CONFIG_EXPAND,
	NBODY_NUM_CONFIGS
};

// utility function
void randomizeBodies(NBodyConfig config, float* pos, float* vel, float* color, float clusterScale, float velocityScale, int numBodies);

// BodySystem abstract base class
class BodySystem
{
public: // methods
    BodySystem(int numBodies) : m_numBodies(numBodies), m_bInitialized(false) {}
    virtual ~BodySystem()         {}

    virtual void update(float deltaTime) = 0;

    enum BodyArray 
    {
        BODYSYSTEM_POSITION,
        BODYSYSTEM_VELOCITY,
    };

    virtual void setSoftening(float softening) = 0;
    virtual void setDamping(float damping) = 0;

    virtual float* getArray(BodyArray array) = 0;
    virtual void   setArray(BodyArray array, const float* data) = 0;
 
    virtual unsigned int getCurrentReadBuffer() const = 0;

    virtual int    getNumBodies() const { return m_numBodies; }

    

protected: // methods
    BodySystem() {} // default constructor

    virtual void _initialize(int numBodies) = 0;
    virtual void _finalize() = 0;

protected: // data
    int m_numBodies;

    bool m_bInitialized;

};

#endif // __BODYSYSTEM_H__
