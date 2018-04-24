// little physics demo using GLUT + GLEW to demonstrate
// the CUDA LCP solver

// author: Peter Kipfer <peter@kipfer.de>

// the rigid body system is based on code from David Eberly's excellent
// book "Game Physics", Morgan Kaufmann, 2004

// handy stuff
#include "GbDefines.hh"
#include "GbTypes.hh"

// system stuff
#include <GL/glew.h>
#include <GL/glut.h>

// application helpers
#include "GLCamera.hh"
#include "GbPlane.hh"

// the LCP solver
#include "CUDASolver.hh"

// rigid body stuff
#include "RigidShape.hh"
#include "RigidTetrahedron.hh"
#include "RigidHexahedron.hh"
#include "RigidObject.hh"


// application state
GLCamera camera_;
int lastx_ = 0;
int lasty_ = 0; // last mouse click screen position
unsigned int mouseButtons_ = 0;
float rotX_ = 0.0f;
float rotY_ = 0.0f;
float rotZ_ = 0.0f;

GbVec3<float> worldUp(GbVec3<float>::UNIT_Y);


// rigid bodies setup
enum { MAX_NUM_VERTICES = 16, MAX_NUM_FACES = 12, NUM_BODIES_EACH_TYPE = 48, NUM_TYPES = 3 };
#define NUM_BODIES_IN_SIM (NUM_BODIES_EACH_TYPE*NUM_TYPES)
std::vector<RigidShape*> rigidBodyShape_;

// representation of boundaries (floor, ceiling, walls)
GoRigidBody<float> boundary_[6];
GbVec3<float> boundaryLocation_[6];
GbVec3<float> boundaryNormal_[6];
typedef struct
{
    GbVec3<float> vertices[4];
    int indices[6];
    GbVec3<float> colorRGB;
} BoxFace;
std::vector<BoxFace> boxFaces_;
BoxFace movableFace_;
float movableWallPosition_ = 16.0f;

// contact points for the collision detection system
typedef struct
{
    GoRigidBody<float>* A;    // object containing face
    GoRigidBody<float>* B;    // object containing vertex
    GbVec3<float> PA;         // contact point for object A
    GbVec3<float> PB;         // contact point for object B
    GbVec3<float> N;          // outward unit-length normal to face
    GbBool isBoundaryContact; // true if this is a boundary contact (used for rendering)
} ContactPoint;

// data used during one pass of the physics simulation
std::vector<ContactPoint> contactPoints_;
GbVec3<float>* solutionPoints_;
CUDASolver::CollisionPair* pairList_; unsigned int pairListSize_;
CUDASolver::CollisionPair* reduceList_;
CUDASolver::StatusCode*    statusList_;

// simulated clock
float simulationTime_ = 0.0f;
float simulationTimeDelta_ = 0.01f;

// error tolerance used for interpenetration calculations
float compareTolerance_ = std::numeric_limits<float>::epsilon()*1000.0f;

// kinetic energy of the system
float totalKineticEnergy_ = 0.0f;

// render state
GbBool drawWireframe_ = false;
GbBool drawBroadphase_ = false;
GbBool drawNormals_ = false;
GbBool runSimulation_ = true;

// finally: the LCP solver
CUDASolver LCPDist_;



void 
shutdownAndDeallocate()
{
    staticdebugmsg("shutdown","cleaning up");
    for (unsigned int i=0; i<rigidBodyShape_.size(); ++i)
    {
	delete rigidBodyShape_[i];
    }
}

// force function for rigid body integration
GbVec3<float> 
Force (float, float fMass, const GbVec3<float>&,
       const GbQuaternion<float>&, const GbVec3<float>&, const GbVec3<float>&,
       const GbMatrix3<float>&, const GbVec3<float>&, const GbVec3<float>&)
{
    const float fGravityConstant = 9.81f;
    const GbVec3<float> kGravityDirection( -worldUp );
    return (fGravityConstant*fMass)*kGravityDirection;
}

// torque function for rigid body integration
GbVec3<float> 
Torque (float, float, const GbVec3<float>&,
	const GbQuaternion<float>&, const GbVec3<float>&, const GbVec3<float>&,
	const GbMatrix3<float>&, const GbVec3<float>&, const GbVec3<float>&)
{
    return GbVec3<float>::ZERO;
}

// add some rigid bodies to the world
void 
createRigidBodies ()
{
    GbVec3<float> akVertex[MAX_NUM_VERTICES];

    rigidBodyShape_.resize(NUM_BODIES_IN_SIM);

    int bodyType = -1;

    for (int i = 0; i < NUM_BODIES_IN_SIM; i++)
    {
        float fSize = 1.0f + 0.8f * GbMath<float>::SymmetricRandom();
        float fMass = 8.0f*fSize;
	float ir = 4.0f*GbMath<float>::UnitRandom();
        GbVec3<float> kPos = GbVec3<float>(3.0f,3.0f+3.0f*ir,14.0f-3.0f*ir);
        GbVec3<float> kLinMom = 0.01f*GbVec3<float>( 2.0f + 100.0f*GbMath<float>::SymmetricRandom(),
						     2.0f + 100.0f*GbMath<float>::SymmetricRandom(),
						     -1.2f + 100.0f*GbMath<float>::SymmetricRandom());
        GbVec3<float> kAngMom = 0.01f*GbVec3<float>( 1.0f + 100.0f*GbMath<float>::SymmetricRandom(),
						     2.0f + 100.0f*GbMath<float>::SymmetricRandom(),
						     3.0f + 100.0f*GbMath<float>::SymmetricRandom());

	if (!(i%NUM_BODIES_EACH_TYPE)) bodyType++;

	switch (bodyType)
	{
	    case 0:
		rigidBodyShape_[i] = new RigidTetrahedron(fSize,fMass,kPos,kLinMom,kAngMom);
		break;
	    case 1:
		rigidBodyShape_[i] = new RigidHexahedron(fSize,fMass,kPos,kLinMom,kAngMom);
		break;
	    case 2:
		rigidBodyShape_[i] = new RigidObject(fSize/500.0f,fMass,kPos,kLinMom,kAngMom);
		break;

	    default:
		staticdebugmsg("createRigidBodies","skipping unknown object type");
		break;
	}
		
        rigidBodyShape_[i]->Force = Force;
        rigidBodyShape_[i]->Torque = Torque;

	assert(rigidBodyShape_[i]->getNumHullVertices()<=MAX_NUM_VERTICES);
        rigidBodyShape_[i]->getWorldSpaceHullVertices(akVertex);
	const GbVec3i<int>* akIndex = rigidBodyShape_[i]->getHullFaces();
	LCPDist_.storeVertices(i,rigidBodyShape_[i]->getNumHullVertices(),akVertex);
	LCPDist_.storeIndices(i,rigidBodyShape_[i]->getNumHullFaces(),akIndex);
    }	

    LCPDist_.commitVertices();
    LCPDist_.commitIndices();
}

// build the scene
void 
createBox()
{
    BoxFace boxFace;

    // floor
    boxFace.vertices[0] = GbVec3<float>(1.0f,1.0f,1.0f);
    boxFace.vertices[1] = GbVec3<float>(17.0f,1.0f,1.0f);
    boxFace.vertices[2] = GbVec3<float>(17.0f,20.0f,1.0f);
    boxFace.vertices[3] = GbVec3<float>(1.0f,20.0f,1.0f);

    boxFace.indices[0] = 0;  boxFace.indices[1] = 1;  boxFace.indices[2] = 2;
    boxFace.indices[3] = 0;  boxFace.indices[4] = 2;  boxFace.indices[5] = 3;

    boxFace.colorRGB  = GbVec3<float>(155.0f/255.0f,177.0f/255.0f,164.0f/255.0f);

    boxFaces_.push_back(boxFace);

    // ceiling
    boxFace.vertices[0] = GbVec3<float>(1.0f,1.0f,17.0f);
    boxFace.vertices[1] = GbVec3<float>(17.0f,1.0f,17.0f);
    boxFace.vertices[2] = GbVec3<float>(17.0f,20.0f,17.0f);
    boxFace.vertices[3] = GbVec3<float>(1.0f,20.0f,17.0f);

    boxFaces_.push_back(boxFace);

    // right wall
    boxFace.vertices[0] = GbVec3<float>(1.0f,15.0f,1.0f);
    boxFace.vertices[1] = GbVec3<float>(17.0f,15.0f,1.0f);
    boxFace.vertices[2] = GbVec3<float>(17.0f,15.0f,17.0f);
    boxFace.vertices[3] = GbVec3<float>(1.0f,15.0f,17.0f);

    boxFace.colorRGB = GbVec3<float>(170.0f/255.0f,187.0f/255.0f,219.0f/255.0f);

    boxFaces_.push_back(boxFace);

    // left wall
    boxFace.vertices[0] = GbVec3<float>(17.0f,2.0f,1.0f);
    boxFace.vertices[1] = GbVec3<float>(1.0f,2.0f,1.0f);
    boxFace.vertices[2] = GbVec3<float>(1.0f,2.0f,17.0f);
    boxFace.vertices[3] = GbVec3<float>(17.0f,2.0f,17.0f);

    boxFaces_.push_back(boxFace);

    // back wall
    boxFace.vertices[0] = GbVec3<float>(1.0f,1.0f,1.0f);
    boxFace.vertices[1] = GbVec3<float>(1.0f,20.0f,1.0f);
    boxFace.vertices[2] = GbVec3<float>(1.0f,20.0f,17.0f);
    boxFace.vertices[3] = GbVec3<float>(1.0f,1.0f,17.0f);

    boxFace.colorRGB = GbVec3<float>(209.0f/255.0f,204.0f/255.0f,180.0f/255.0f);

    boxFaces_.push_back(boxFace);

    // front wall (movable)
    movableFace_.vertices[0] = GbVec3<float>(movableWallPosition_,1.0f,1.0f);
    movableFace_.vertices[1] = GbVec3<float>(movableWallPosition_,20.0f,1.0f);
    movableFace_.vertices[2] = GbVec3<float>(movableWallPosition_,20.0f,17.0f);
    movableFace_.vertices[3] = GbVec3<float>(movableWallPosition_,1.0f,17.0f);

    movableFace_.indices[0] = 0;  movableFace_.indices[1] = 1;  movableFace_.indices[2] = 2;
    movableFace_.indices[3] = 0;  movableFace_.indices[4] = 2;  movableFace_.indices[5] = 3;

    movableFace_.colorRGB = GbVec3<float>(209.0f/255.0f,204.0f/255.0f,180.0f/255.0f);
}


// this is the broadphase for immovable objects
GbBool 
farFromBoundary(const RigidShape& shape)
{
    // The tests are arranged so that the most likely to be encountered
    // (the floor) is tested first and the least likely to be encountered
    // (the ceiling) is tested last.
    // works only however if you don't turn the scene upside down with the mouse

    const GbVec3<float>& position = shape.getPosition();
    const float radius = shape.getRadius();

    return position[2]-radius >= boundaryLocation_[2][2]
        && position[0]-radius >= boundaryLocation_[0][0]
        && position[0]+radius <= boundaryLocation_[5][0]
        && position[1]-radius >= boundaryLocation_[1][1]
        && position[1]+radius <= boundaryLocation_[3][1]
        && position[2]+radius <= boundaryLocation_[4][2];
}

// this is the broadphase for moving objects
GbBool 
farApart(int indexA, int indexB)
{
    const GbVec3<float>& positionA = rigidBodyShape_[indexA]->getPosition();
    const GbVec3<float>& positionB = rigidBodyShape_[indexB]->getPosition();
    const float radiusSum = rigidBodyShape_[indexA]->getRadius() + rigidBodyShape_[indexB]->getRadius();

    return (positionA - positionB).getSquareNorm() >= (radiusSum * radiusSum);
}


// create a contact point when a moving object hits an immovable object
GbBool 
boundaryIntersection (int shapeIndex, int boundaryIndex, const float* vertexBoundaryDistance)
{
    int hitIndex = -1;
    float maxDepth = 0.0f;
    for (unsigned int j = 0; j < rigidBodyShape_[shapeIndex]->getNumHullVertices(); j++)
    {
        const float depth = vertexBoundaryDistance[j];
        if (depth < maxDepth)
        {
            maxDepth = depth;
            hitIndex = j;
        }
    }
    if (hitIndex != -1)
    {
	ContactPoint contact;

	RigidShape* shape = rigidBodyShape_[shapeIndex];

	contact.A = &boundary_[boundaryIndex];
	contact.B = shape;
	contact.N = boundaryNormal_[boundaryIndex];
	contact.PA = GbVec3<float>::ZERO;
	contact.PB = shape->getWorldSpaceHullVertex(hitIndex);
	contact.isBoundaryContact = true;

	// ignore if there are previous movements, always move intersecting shapes to surface of boundary
	shape->setPosition(shape->getPosition() - maxDepth * contact.N);
	shape->moved = true;

        contactPoints_.push_back(contact);
        return true;
    }
    return false;
}


// helper to see if contact point is actually a hull vertex
GbBool 
isVertex (const GbVec3<float>* vertices, int numVertices, const GbVec3<float>& contactPoint)
{
    // this can fail because shape may have moved
    const float cmpSquared = compareTolerance_ * compareTolerance_;
    for (int i = 0; i < numVertices; i++)
    {
        const GbVec3<float> relPos(vertices[i] - contactPoint);
        if (relPos.getSquareNorm() < cmpSquared)
        {
            return true;
        }
    }
    return false;
}

// calculate the normal of the hull plane with shortest distance to the contact point
void 
calculateNormal (const GbVec3<float>* vertices, const GbVec3i<int>* indices, int numFaces, 
		 const GbVec3<float>& contactPoint, ContactPoint& contact)
{
    float distance = std::numeric_limits<float>::max();
    for (int i = 0; i < numFaces; i++ )
    {
	const GbVec3i<int> idx(indices[i]);
        const GbPlane<float> plane(vertices[idx[0]],
				   vertices[idx[1]],
				   vertices[idx[2]]);
        const float temp = GbMath<float>::Abs(plane.distanceTo(contactPoint));
        if (temp < distance)
        {
            contact.N = plane.getNormal();
            distance = temp;
        }
    }
}

// Find the edge of the rigid body nearest to the contact point.
// If otherVertex is ZERO, then closestEdge skips the calculation of
// an unneeded other vertex for the second body.
GbVec3<float> 
closestEdge (const GbVec3<float>* vertices, int numVertices, 
	     const GbVec3<float>& contactPoint, GbVec3<float>& otherVertex)
{
    // this routine can be optimized. it currently also checks interior edges.

    GbVec3<float> closestEdge(0.0f);
    float minDistance = std::numeric_limits<float>::max();
    for (int i = 0; i < (numVertices-1); i++)
    {
        for (int j = i+1; j < numVertices; j++)
        {
            const GbVec3<float> edge(vertices[j] - vertices[i]);
            const GbVec3<float> relPosition(contactPoint - vertices[i]);
            const float posDotEdge = relPosition.dot(edge);
            const float edgeLengthSquare = edge.getSquareNorm();
            const float relPosSquare = relPosition.getSquareNorm();
            const float distance = GbMath<float>::Abs((posDotEdge*posDotEdge)/(edgeLengthSquare*relPosSquare)-1.0f);

            if (distance < minDistance)
            {
                minDistance = distance;
                closestEdge = edge;
                for (int k = 0; otherVertex != GbVec3<float>::ZERO, k < (numVertices-1); k++)
                {
                    if (k != i && k != j)
                    {
                        otherVertex = vertices[k];
                        continue;
                    }
                }
            }
        }
    }

    return GbVec3<float>(closestEdge);
}


// reposition two rigid bodies according to the computed closest points
// and create a contact point from this collision
void
reposition (int pairIndex, const GbVec3<float>& solverResult0, const GbVec3<float>& solverResult1, ContactPoint& contact)
{
    int indexA = pairList_[pairIndex].a;
    int indexB = pairList_[pairIndex].b;

    RigidShape* shapeA = rigidBodyShape_[indexA];
    RigidShape* shapeB = rigidBodyShape_[indexB];

    contact.isBoundaryContact = false;

    // get object data
    GbVec3<float> verticesA[MAX_NUM_VERTICES], verticesB[MAX_NUM_VERTICES];
    const GbVec3i<int>* indicesA = shapeA->getHullFaces();
    const GbVec3i<int>* indicesB = shapeB->getHullFaces();
    shapeA->getWorldSpaceHullVertices(verticesA);
    shapeB->getWorldSpaceHullVertices(verticesB);
    const GbVec3<float>& centroidA = shapeA->getCenter();
    const GbVec3<float>& centroidB = shapeB->getCenter();

    // calculate the theoretical points of contact from the reduced solver points
    GbVec3<float> solverResult[2];
#ifdef DEBUG
    if (statusList_[pairIndex] != CUDASolver::SC_FOUND_SOLUTION)
    { 
	staticerrormsg("reposition","solver error");
    }
#endif
    solverResult[0] = centroidA+(solverResult0-centroidA)/pairList_[pairIndex].reduce_a;
    solverResult[1] = centroidB+(solverResult1-centroidB)/pairList_[pairIndex].reduce_b;
    GbVec3<float> solverResultDifference(solverResult[0] - solverResult[1]);
    float bodyDistance = solverResultDifference.getNorm();

    // In theory, the LCP solver should always find a valid distance, but just
    // in case numerical round-off errors cause problems, let us trap it here
    assert(bodyDistance >= 0.0f);

    // Apply the separation distance along the line containing the centroids
    // of the convex hulls
    const GbVec3<float> centroidDifference(centroidB - centroidA);
    solverResultDifference = centroidDifference.getNormalized() * bodyDistance;

    // Move each object by half of solverResultDifference and apply some solver damping
    const float damping = 0.5f;
    const GbVec3<float> scaledDistance(damping*0.5f*solverResultDifference);

    // undo the interpenetration
    if (shapeA->moved && !shapeB->moved)
    {
        // indexA has been moved but indexB has not
	shapeB->setPosition(shapeB->getPosition() + 2.0f * scaledDistance);
        shapeB->moved = true;
    }
    else if (!shapeA->moved && shapeB->moved)
    {
        // indexB has been moved but indexA has not
	shapeA->setPosition(shapeA->getPosition() - 2.0f * scaledDistance);
        shapeA->moved = true;
    }
    else
    {
        // both moved or both did not move
	shapeA->setPosition(shapeA->getPosition() - scaledDistance);
        shapeA->moved = true;
	shapeB->setPosition(shapeB->getPosition() + scaledDistance);
        shapeB->moved = true;
    }


    // test if the two objects intersect in a vertex-face configuration
    GbBool isVFContact = isVertex(verticesA,shapeA->getNumHullVertices(),solverResult[0]);
    if (isVFContact)
    {
        contact.A = shapeB;
        contact.B = shapeA;
	calculateNormal(verticesB,indicesB,shapeB->getNumHullFaces(),solverResult[1],contact);
    }
    else
    {
        isVFContact = isVertex(verticesB,shapeB->getNumHullVertices(),solverResult[1]);
        if (isVFContact)
        {
            contact.A = shapeA;
            contact.B = shapeB;
	    calculateNormal(verticesA,indicesA,shapeA->getNumHullFaces(),solverResult[0],contact);
        }
    }

    // test if the two objects intersect in an edge-edge configuration
    if (!isVFContact)
    {
        contact.A = shapeA;
        contact.B = shapeB;
        GbVec3<float> otherVertexA(GbVec3<float>::UNIT_X);
        GbVec3<float> otherVertexB(GbVec3<float>::ZERO);
	GbVec3<float> EA = closestEdge(verticesA,shapeA->getNumHullVertices(),solverResult[0],otherVertexA);
	GbVec3<float> EB = closestEdge(verticesB,shapeB->getNumHullVertices(),solverResult[1],otherVertexB);
        const GbVec3<float> normal(EA.unitCross(EB));
        if (normal.dot(otherVertexA - solverResult[0]) < 0.0f)
        {
            contact.N = normal;
        }
        else
        {
            contact.N = -normal;
        }
    }

    // position contact point to correspond to relocaton of objects
    contact.PA = solverResult[0] - scaledDistance;
    contact.PB = solverResult[1] + scaledDistance;
}


// resolve real collisions of moving convex objects
// as the LCP gives us the distance between the objects, simply shrink the objects
// to 95% of their size to get stable results. This also accounts for slight 
// interpenetrations that may be present because of forced repositioning by
// boundaries or similar
void 
resolveCollisions()
{
    staticdebugmsg("resolveCollisions","resolving "<<pairListSize_<<" collisions");

    // Randomly perturb the shrinking factor by a small amount.  This is
    // done to help prevent the LCP solver from getting into cycles and
    // degenerate cases.
    const float reduction = 0.95f;
    for (unsigned int i=0; i<pairListSize_; ++i)
    {
	CUDASolver::CollisionPair& pair = pairList_[i];
	pair.reduce_a = reduction*GbMath<float>::IntervalRandom(0.9999f,1.0001f);
	pair.reduce_b = reduction*GbMath<float>::IntervalRandom(0.9999f,1.0001f);
    }

    // do it
    LCPDist_.solve(pairList_, pairListSize_, solutionPoints_, statusList_);

    // add successfull results to the contact points list
    for (unsigned int i=0; i<pairListSize_; ++i)
    {
 	if (statusList_[i] != CUDASolver::SC_FOUND_SOLUTION)
 	{ 
  	    staticdebugmsg("resolveCollisions","status "<<i<<": "<<int(statusList_[i]));
 	}
 	else
 	{
	    ContactPoint contact;
	    reposition(i,solutionPoints_[i*2],solutionPoints_[i*2+1],contact);
	    contactPoints_.push_back(contact);
	}
    }
}

// narrowphase processing
// check the collision pair list obtained from the broadphase for really
// colliding objects. remove clear non-contacts from the list.
void 
reducePairList()
{
    staticdebugmsg("reducePairList","checking list with "<<pairListSize_<<" collisions");

    // do a copy - don't want to destroy pairList_ memory allocation
    std::memcpy(reduceList_,pairList_,pairListSize_*sizeof(CUDASolver::CollisionPair));
    unsigned int reduceListSize = pairListSize_;

    // do it
    LCPDist_.solve(reduceList_, reduceListSize, solutionPoints_, statusList_);

    // check which object pairs are really close or touch
    pairListSize_ = 0;
    for (unsigned int i=0; i<reduceListSize; ++i)
    {
 	if (statusList_[i] != CUDASolver::SC_FOUND_SOLUTION)
 	{ 
  	    staticdebugmsg("reducePairList","status "<<i<<": "<<int(statusList_[i]));
 	}
 	else
 	{
	    const GbVec3<float> kDiff( solutionPoints_[i*2] - solutionPoints_[i*2+1] );
 	    if (kDiff.getSquareNorm() <= (compareTolerance_*compareTolerance_))
		pairList_[pairListSize_++] = reduceList_[i];
 	}
    }

    staticdebugmsg("reducePairList","reduced list has "<<pairListSize_<<" collisions");
}


// this is the primary collision detection routine
// for moving/moving and moving/immovable collision events
void 
collisionDetection()
{
    // start anew
    contactPoints_.clear();

    // test for moving/moving collisions
    pairListSize_ = 0;

    // broadphase (not very sophisticated, I know...)
    for (int i = 0; i < NUM_BODIES_IN_SIM-1; i++)
    {
        for (int j = i+1; j < NUM_BODIES_IN_SIM; j++)
        {
            if (!farApart(i,j))
            {
		CUDASolver::CollisionPair pair(i,j);
		pairList_[pairListSize_++] = pair;
            }
        }
    }

    // narrowphase
    reducePairList();

    // reposition
    resolveCollisions();


    // test for moving/immovable collisions
    // penetrating objects will be forcibly repositioned
    // induced collisions with other objects will have to be
    // handled in next time step
    GbVec3<float> vertices[MAX_NUM_VERTICES];
    float vertexDistance[MAX_NUM_VERTICES];

    for (int i = 0; i < NUM_BODIES_IN_SIM; i++)
    {
	RigidShape& shape = *rigidBodyShape_[i];

        shape.moved = false;
        if (farFromBoundary(shape))
        {
            continue;
        }

        // These checks are done in pairs under the assumption that the objects
        // have smaller diameters than the separation of opposite boundaries, 
        // hence only one of each opposite pair of boundaries may be touched 
        // at any one time.
        shape.getWorldSpaceHullVertices(vertices);
        const float radius = shape.getRadius();
        const GbVec3<float>& position = shape.getPosition();

        // rear [0] and front[5] boundaries
        if (position[0] - radius < boundaryLocation_[0][0])
        {
            for (unsigned int j = 0; j < shape.getNumHullVertices(); j++)
            {
                vertexDistance[j] = vertices[j][0] - boundaryLocation_[0][0];
            }
            boundaryIntersection(i,0,vertexDistance);
        }
        else if (position[0] + radius > boundaryLocation_[5][0])
        {
            for (unsigned int j = 0; j < shape.getNumHullVertices(); j++)
            {
                vertexDistance[j] = boundaryLocation_[5][0] - vertices[j][0];
            }
            boundaryIntersection(i,5,vertexDistance);
        }

        // left [1] and right [3] boundaries
        if (position[1] - radius < boundaryLocation_[1][1])
        {
            for (unsigned int j = 0; j < shape.getNumHullVertices(); j++)
            {
                vertexDistance[j] = vertices[j][1] - boundaryLocation_[1][1];
            }
            boundaryIntersection(i,1,vertexDistance);
        }
        else if (position[1] + radius > boundaryLocation_[3][1])
        {
            for (unsigned int j = 0; j < shape.getNumHullVertices(); j++)
            {
                vertexDistance[j] = boundaryLocation_[3][1] - vertices[j][1];
            }
            boundaryIntersection(i,3,vertexDistance);
        }

        // bottom [2] and top [4] boundaries
        if (position[2] - radius < boundaryLocation_[2][2])
        {
            for (unsigned int j = 0; j < shape.getNumHullVertices(); j++)
            {
                vertexDistance[j] = vertices[j][2] - boundaryLocation_[2][2];
            }
            boundaryIntersection(i,2,vertexDistance);
        }
        else if (position[2] + radius > boundaryLocation_[4][2])
        {
            for (unsigned int j = 0; j < shape.getNumHullVertices(); j++)
            {
                vertexDistance[j] = boundaryLocation_[4][2] - vertices[j][2];
            }
            boundaryIntersection(i,4,vertexDistance);
        }
    }    

}


// calculate the impule generated by the contact points and
// apply forces to the objects
void 
impulse()
{
    // coefficient of restitution
    float restitution = 0.8f;
    float temp = 20.0f * float(NUM_BODIES_IN_SIM);

    if (totalKineticEnergy_ < temp)
    {
        restitution *= 0.5f * totalKineticEnergy_ / temp;
    }
    const float restitutionCoefficient = -(1.0f + restitution);

    for (unsigned int i = 0; i < contactPoints_.size(); i++)
    {
	const ContactPoint& contact = contactPoints_[i];
	GoRigidBody<float>& bodyA = *contact.A;
	GoRigidBody<float>& bodyB = *contact.B;
	const GbVec3<float> relPosA( contact.PA - bodyA.getPosition() );
	const GbVec3<float> relPosB( contact.PB - bodyB.getPosition() );

        const GbVec3<float> velA( bodyA.getLinearVelocity() + bodyA.getAngularVelocity().cross(relPosA) );
        const GbVec3<float> velB( bodyB.getLinearVelocity() + bodyB.getAngularVelocity().cross(relPosB) );
        const float preimpulseRelativeVelocity = contact.N.dot(velB-velA);

	float impulseMagnitude = 0.0f;
        if (preimpulseRelativeVelocity < 0.0f)
        {
            const GbVec3<float> relVel( bodyA.getLinearVelocity() - bodyB.getLinearVelocity() );
            const GbVec3<float> relPosACrossN( relPosA.cross(contact.N) );
            const GbVec3<float> relPosBCrossN( relPosB.cross(contact.N) );
            const GbVec3<float> inertiaA( bodyA.getWorldInverseInertia() * relPosACrossN );
            const GbVec3<float> inertiaB( bodyB.getWorldInverseInertia() * relPosBCrossN );

            float numer = restitutionCoefficient*( contact.N.dot(relVel)
						   + bodyA.getAngularVelocity().dot(relPosACrossN)
						   - bodyB.getAngularVelocity().dot(relPosBCrossN) );

            float denom = 
		bodyA.getInverseMass() +
		bodyB.getInverseMass() +
		relPosACrossN.dot(inertiaA) + 
		relPosBCrossN.dot(inertiaB);

            impulseMagnitude = numer / denom;
        }

        GbVec3<float> linearA(bodyA.getLinearMomentum());
        GbVec3<float> linearB(bodyB.getLinearMomentum());
        GbVec3<float> angularA(bodyA.getAngularMomentum());
        GbVec3<float> angularB(bodyB.getAngularMomentum());

        // update rigid body
        const GbVec3<float> impulse(impulseMagnitude * contact.N);
        linearA += impulse;
        linearB -= impulse;
        angularA += relPosA.cross(impulse);
        angularB -= relPosB.cross(impulse);

        bodyA.setLinearMomentum(linearA);
        bodyB.setLinearMomentum(linearB);
        bodyA.setAngularMomentum(angularA);
        bodyB.setAngularMomentum(angularB);
    }
}


// integrate bodies in time and update LCP solver data
void 
integrate()
{
    GbVec3<float> vertices[MAX_NUM_VERTICES];

    for (int i = 0; i < NUM_BODIES_IN_SIM; i++)
    {
	RigidShape& shape = *rigidBodyShape_[i];
	
        shape.integrate(simulationTime_,simulationTimeDelta_);

        shape.getWorldSpaceHullVertices(vertices);
	LCPDist_.storeVertices(i,shape.getNumHullVertices(),vertices);
    }

    LCPDist_.commitVertices();
}


// this is the primary collision response routine
void 
collisionResponse()
{
    if (!contactPoints_.empty())
    {
	staticdebugmsg("collisionResponse","processing "<<contactPoints_.size()<<" contact points");
        impulse();
    }

    integrate();
}


// main physics routine
void 
stepPhysics(GbBool triggerRedraw)
{
    staticdebugmsg("stepPhysics","stepping physics by "<<simulationTimeDelta_);

    worldUp = camera_.getUpDirection();

    collisionDetection();
    collisionResponse();

    // update global system state
    totalKineticEnergy_ = 0.0f;
    for (int i = 0; i < NUM_BODIES_IN_SIM; i++)
    {
        const GoRigidBody<float>& body = *rigidBodyShape_[i];
        const float fInvMass = body.getInverseMass();
        const GbMatrix3<float>& rkInertia = body.getWorldInertia();
        const GbVec3<float>& rkLinMom = body.getLinearMomentum();
        const GbVec3<float>& rkAngVel = body.getAngularVelocity();

        totalKineticEnergy_ += fInvMass*rkLinMom.dot(rkLinMom) + rkAngVel.dot(rkInertia*rkAngVel);
    }
    totalKineticEnergy_ *= 0.5f;

    // update the display
    if (triggerRedraw) glutPostRedisplay();

    // advance simulated clock
    simulationTime_ += simulationTimeDelta_;
}


// GLUT auto trigger
void 
myIdle()
{
    stepPhysics(true);
}


// GLUT display routine
void 
paintGL()
{
    staticdebugmsg("paintGL","painting");

    camera_.update(GLCamera::NONE);

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glRotatef(rotX_,1,0,0);
    glRotatef(rotY_,0,1,0);
    glRotatef(rotZ_,0,0,1);


    // draw the box
    glDepthMask(GL_FALSE);
    glColor4f(movableFace_.colorRGB[0],movableFace_.colorRGB[1],movableFace_.colorRGB[2],0.5f);
    GbVec3<float>* v = movableFace_.vertices;
    GbVec3<float> faceNormal((v[1] - v[0]).unitCross(v[2] - v[0]));
    glNormal3fv(&faceNormal[0]);
    glBegin(GL_TRIANGLES);
    for (int j=0; j<6; ++j)
    {
	int idx = movableFace_.indices[j];
	glVertex3fv((const GLfloat*)&v[idx]);
    }
    glEnd();
    if (drawNormals_)
    {
	GbVec3<float> faceCenter(v[0]+v[1]+v[2]+v[3]);
	faceCenter *= 0.25f;
	glColor3f(0,1,0);
	glBegin(GL_LINES);
	glVertex3fv(&faceCenter[0]);
	faceCenter += faceNormal;
	glVertex3fv(&faceCenter[0]);
	glEnd();
    }
    for (unsigned int i=0; i<boxFaces_.size(); ++i)
    {
	glColor4f(boxFaces_[i].colorRGB[0],boxFaces_[i].colorRGB[1],boxFaces_[i].colorRGB[2],0.5f);
	v = boxFaces_[i].vertices;
	faceNormal = ((v[1] - v[0]).unitCross(v[2] - v[0]));
	glNormal3fv(&faceNormal[0]);
	glBegin(GL_TRIANGLES);
	for (int j=0; j<6; ++j)
	    glVertex3fv((const GLfloat*)&v[boxFaces_[i].indices[j]]);
	glEnd();
	if (drawNormals_)
	{
	    GbVec3<float> faceCenter(v[0]+v[1]+v[2]+v[3]);
	    faceCenter *= 0.25f;
	    glColor3f(0,1,0);
	    glBegin(GL_LINES);
	    glVertex3fv(&faceCenter[0]);
	    faceCenter += faceNormal;
	    glVertex3fv(&faceCenter[0]);
	    glEnd();
	}
    }
    glDepthMask(GL_TRUE);


    // draw the rigid bodies
    std::vector<GbVec3<float> > vertices;
    for (int i = 0; i < NUM_BODIES_IN_SIM; i++)
    {

	if (drawBroadphase_)
	{
	    glColor3f(1,1,0);
	    GbVec3<float> pos(rigidBodyShape_[i]->getPosition());
	    glPushMatrix();
	    glTranslatef(pos[0],pos[1],pos[2]);
	    glutWireSphere(rigidBodyShape_[i]->getRadius(),8,8);
	    glPopMatrix();
	}

	vertices.resize(rigidBodyShape_[i]->getNumVertices());
	rigidBodyShape_[i]->getWorldSpaceVertices(&vertices[0]);
	const GbVec3i<int>* indices = rigidBodyShape_[i]->getFaces();

	for (unsigned int f=0; f<rigidBodyShape_[i]->getNumFaces(); ++f) 
	{
	    GbVec3<float> faceNormal((vertices[indices[f][1]] - vertices[indices[f][0]]).unitCross(vertices[indices[f][2]] - vertices[indices[f][0]]));
	    glNormal3fv(&faceNormal[0]);
	    glColor4f(1,1,1,1);
	    glBegin(GL_TRIANGLES);
	    GbVec3<float> faceCenter(0.0f);
	    for (int j=0; j<3; ++j) 
	    {
		glVertex3fv((const GLfloat*)&vertices[indices[f][j]]);
		faceCenter += vertices[indices[f][j]];
	    }
	    glEnd();

	    if (drawNormals_)
	    {
		faceCenter *= 0.33333333333f;
		glColor3f(0,1,0);
		glBegin(GL_LINES);
		glVertex3fv(&faceCenter[0]);
		faceCenter += faceNormal;
		glVertex3fv(&faceCenter[0]);
		glEnd();
	    }
	}

    }


    // draw contact points connection line
    glDisable(GL_LIGHTING);
    glLineWidth(3);
    glBegin(GL_LINES);
    glColor4f(1,0,0,1);
    for (unsigned int i=0; i<contactPoints_.size(); ++i)
    {
	if (!contactPoints_[i].isBoundaryContact) 
	{
	    glColor4f(1,0,0,1);
	    glVertex3fv((const GLfloat*)&contactPoints_[i].PA);
	    glColor4f(0,0,1,1);
	    glVertex3fv((const GLfloat*)&contactPoints_[i].PB);
	}
    }
    glEnd();
    glLineWidth(1);
    glEnable(GL_LIGHTING);


    // this seems to help on loaded systems to avoid CUDA hiccups
    glFinish();

    glutSwapBuffers();
}


// setup everything
void 
initializeApp()
{
    static const GLfloat mcold[4] = { 0.8f,0.8f,0.8f,1.0f };
    static const GLfloat mcola[4] = { 0.2f,0.2f,0.2f,1.0f };

    static const GLfloat light0_ambient[4]  = {0.1f, 0.1f, 0.1f, 1.0f};
    static const GLfloat light0_diffuse[4]  = {0.8f, 0.8f, 0.8f, 1.0f};
    static const GLfloat light0_specular[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    static const GLfloat light0_position[4] = {0.0f, 10.0f, 10.0f, 0.0f};

    int err = glewInit();
    if (GLEW_OK != err) {
	// problem: glewInit failed, something is seriously wrong
        staticerrormsg("initializeGL","cannot initialize OpenGL");
	staticerrormsg("initializeGL","GLEW Error: "<<glewGetErrorString(err));
        exit(1);
    }  
    
    glClearColor(0,0,0,1);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_FASTEST);
    glHint(GL_FOG_HINT, GL_FASTEST);

    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glDisable( GL_CULL_FACE ); 
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
    glEnable(GL_LIGHTING);

    glColorMaterial(GL_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,mcold);
    glMaterialfv(GL_FRONT,GL_AMBIENT,mcola);
    glEnable(GL_COLOR_MATERIAL);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    // set up camera
    float fAngle = 0.02f*GbMath<float>::PI;
    camera_.setFrustumParam(45.0f,1.0f,1.0f,1000.0f);
    camera_.setEye(GbVec3<float>(27.5f,8.0f,8.9f));
    camera_.setLookDirection(GbVec3<float>(-GbMath<float>::Cos(fAngle),0.0f,-GbMath<float>::Sin(fAngle)));

    LCPDist_.configure(MAX_NUM_VERTICES,MAX_NUM_FACES,NUM_BODIES_IN_SIM);

    // create scene
    createRigidBodies();
    createBox();

    // the objects are constrained to bounce inside the box
    // construct immovable rigid bodies for the box walls
    boundaryLocation_[0] = GbVec3<float>(1.0,-100.0,-100.0);
    boundaryNormal_[0]   = GbVec3<float>(1.0,0,0);
    boundaryLocation_[1] = GbVec3<float>(-100.0,2.0,-100.0);
    boundaryNormal_[1]   = GbVec3<float>(0,1.0,0);
    boundaryLocation_[2] = GbVec3<float>( -100.0,-100.0,1.0);
    boundaryNormal_[2]   = GbVec3<float>(0,0,1.0);
    boundaryLocation_[3] = GbVec3<float>( 100.0,15.0,100.0);
    boundaryNormal_[3]   = GbVec3<float>(0,-1.0,0);
    boundaryLocation_[4] = GbVec3<float>( 100.0,100.0,17.0);
    boundaryNormal_[4]   = GbVec3<float>(0,0,-1.0);
    boundaryLocation_[5] = GbVec3<float>( movableWallPosition_,100.0,100.0);
    boundaryNormal_[5]   = GbVec3<float>(-1.0,0,0);
    for (int i = 0; i < 6; i++)
    {
        boundary_[i].setMass(0.0f);
        boundary_[i].setPosition(boundaryLocation_[i]);
    }

    // allocate memory for static structures
    pairList_ = (CUDASolver::CollisionPair*)LCPDist_.allocSystemMem(sizeof(CUDASolver::CollisionPair) *
								    (NUM_BODIES_IN_SIM-1) *
								    (NUM_BODIES_IN_SIM-1)); // no collision with self
    reduceList_ = (CUDASolver::CollisionPair*)LCPDist_.allocSystemMem(sizeof(CUDASolver::CollisionPair) *
								      (NUM_BODIES_IN_SIM-1) *
								      (NUM_BODIES_IN_SIM-1)); // no collision with self
    statusList_ = (CUDASolver::StatusCode*)LCPDist_.allocSystemMem(sizeof(CUDASolver::StatusCode) *
								   (NUM_BODIES_IN_SIM-1) *
								   (NUM_BODIES_IN_SIM-1)); // no collision with self

    solutionPoints_ = (GbVec3<float>*)LCPDist_.allocSystemMem(sizeof(GbVec3<float>) * 2 *
							      (NUM_BODIES_IN_SIM-1) *
							      (NUM_BODIES_IN_SIM-1)); // no collision with self

    contactPoints_.reserve(NUM_BODIES_IN_SIM*NUM_BODIES_IN_SIM + 6);

    // initialize the simulation
    stepPhysics(false);
}


void 
resizeGL( int w, int h )
{
    glViewport(0,0,w,h);
    camera_.setAspect(float(w)/float(h));
    staticdebugmsg("resizeGL","resize "<<w<<"x"<<h);
    glutPostRedisplay();
}

// some glut libs support the mouse wheel
#ifndef GLUT_WHEEL_UP
#define GLUT_WHEEL_UP   0x3
#endif
#ifndef GLUT_WHEEL_DOWN
#define GLUT_WHEEL_DOWN 0x4
#endif

void 
mousePressEvent(int button, int state, int x, int y)
{
    lasty_ = y;
    lastx_ = x;

    mouseButtons_ = 0;
    if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) mouseButtons_ |= 1;
    if ((button == GLUT_MIDDLE_BUTTON) && (state == GLUT_DOWN)) mouseButtons_ |= 2;
    if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN)) mouseButtons_ |= 4;
    if (button == GLUT_WHEEL_DOWN)
    {
  	camera_.setEye(camera_.getEye() + camera_.getAspect() * camera_.getLookDirection());
	glutPostRedisplay();
    }
    if (button == GLUT_WHEEL_UP)
    {
  	camera_.setEye(camera_.getEye() - camera_.getAspect() * camera_.getLookDirection());
	glutPostRedisplay();
    }
}

void 
mouseMoveEvent(int x, int y)
{
    if (mouseButtons_ & 1) // left
    {
 	camera_.yaw(float(x-lastx_)*0.001f);
 	camera_.roll(float(y-lasty_)*0.001f);
    }
    if (mouseButtons_ & 2) // middle
    {
  	camera_.setEye(camera_.getEye() + (float(y-lasty_)/100.0f/camera_.getAspect()) * camera_.getLookDirection());
    }
    if (mouseButtons_ & 4) // right
    {
  	camera_.setEye(camera_.getEye() + (float(x-lastx_)/100.0f*camera_.getAspect()) * camera_.getUpDirection().cross(camera_.getLookDirection()));
  	camera_.setEye(camera_.getEye() + (float(y-lasty_)/100.0f/camera_.getAspect()) * camera_.getUpDirection());
    }
    lastx_=x;
    lasty_=y;
    glutPostRedisplay();
}

void 
keyPressEvent(unsigned char key, int x, int y) 
{
    switch (key) {
	case 27: 
	case 'q':
	    LCPDist_.freeSystemMem(pairList_);
	    LCPDist_.freeSystemMem(reduceList_);
	    LCPDist_.freeSystemMem(statusList_);
	    LCPDist_.freeSystemMem(solutionPoints_);
	    shutdownAndDeallocate();
	    exit(0);
	    break;

	case 13:
	    runSimulation_ = !runSimulation_;
	    if (runSimulation_) 
		glutIdleFunc(myIdle);
	    else
		glutIdleFunc(NULL);
	    break;

	case 'w':
	    drawWireframe_ = !drawWireframe_;
	    if (drawWireframe_)
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	    else
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	    glutPostRedisplay();
	    break;
	    
	case 'n':
	    drawNormals_ = !drawNormals_;
	    glutPostRedisplay();
	    break;
	    
	case 'b':
	    drawBroadphase_ = !drawBroadphase_;
	    glutPostRedisplay();
	    break;
	    
	case '+':
	    if (runSimulation_)  
	    {
		movableWallPosition_ += 0.1f;
		if (movableWallPosition_ > 17.0f) movableWallPosition_ = 17.0f;
		boundaryLocation_[5] = GbVec3<float>( movableWallPosition_,100.0,100.0);
		movableFace_.vertices[0] = GbVec3<float>(movableWallPosition_,1.0f,1.0f);
		movableFace_.vertices[1] = GbVec3<float>(movableWallPosition_,20.0f,1.0f);
		movableFace_.vertices[2] = GbVec3<float>(movableWallPosition_,20.0f,17.0f);
		movableFace_.vertices[3] = GbVec3<float>(movableWallPosition_,1.0f,17.0f);
	    }
	    break;

	case '-':
	    if (runSimulation_)  
	    {
		movableWallPosition_ -= 0.1f;
		if (movableWallPosition_ < 4.0f) movableWallPosition_ = 4.0f;
		boundaryLocation_[5] = GbVec3<float>( movableWallPosition_,100.0,100.0);
		movableFace_.vertices[0] = GbVec3<float>(movableWallPosition_,1.0f,1.0f);
		movableFace_.vertices[1] = GbVec3<float>(movableWallPosition_,20.0f,1.0f);
		movableFace_.vertices[2] = GbVec3<float>(movableWallPosition_,20.0f,17.0f);
		movableFace_.vertices[3] = GbVec3<float>(movableWallPosition_,1.0f,17.0f);
	    }
	    break;

	case '0':
	    stepPhysics(true);
	    break;

	default: 
	    break;
    }
}

int 
main(int argc, char *argv[])
{
    staticdebugmsg("main","Starting ...");

    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowPosition(50, 50);
    glutInitWindowSize(600, 600);
    glutInit(&argc,argv);
    glutCreateWindow("GpuGems3: LCP Algorithms for Collision Detection using CUDA");  

    initializeApp();
    
    glutDisplayFunc(paintGL);
    if (runSimulation_)
	glutIdleFunc(myIdle);
    glutReshapeFunc(resizeGL);
    glutKeyboardFunc(keyPressEvent);
    glutMouseFunc(mousePressEvent);
    glutMotionFunc(mouseMoveEvent);
    
    staticinfomsg("main","Usage:");
    staticinfomsg("main","<enter>\ttoggle simulation");
    staticinfomsg("main","<0>\tsingle step simulation");
    staticinfomsg("main","<+><->\tmove front plane");
    staticinfomsg("main","<w>\ttoggle wireframe");
    staticinfomsg("main","<n>\ttoggle normals");
    staticinfomsg("main","<b>\ttoggle draw broadphase");
    staticinfomsg("main","<q>\tquit");
    
    glutMainLoop();

    // never reached
    return 0;
}
