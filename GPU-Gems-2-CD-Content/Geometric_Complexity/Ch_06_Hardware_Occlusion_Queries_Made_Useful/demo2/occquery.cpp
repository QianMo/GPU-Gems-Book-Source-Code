// occquery.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

extern "C"
{
	#include "MathStuff.h"
	#include "DataTypes.h"
}

#include "glInterface.h"
#include "RenderTraverser.h"
#include <math.h>
#include <time.h>

double nearDist = 0.1; // eye near plane distance
int winWidth, winHeight;
int objectType = Geometry::TEAPOT;
int nextObjectType = objectType;

float visZoomFactor = 1.5f;

bool showHelp = false;
bool showStatistics = true;
bool showBoundingVolumes = false;
bool visMode = false;
bool showCreateParams = false;

// traverses and renders the hierarchy
RenderTraverser traverser;

Vector3 eyePos = {0.0, 0.0, 3.0};  // eye position 
Vector3 viewDir = {0.0, 0.0, -1.0};  // eye view dir 
Vector3 runDir = {0.0, 0.0, -1.0}; // direction where we are going  

Matrix4x4 eyeView; // eye view matrix
Matrix4x4 eyeProjection; // eye projection matrix
Matrix4x4 eyeProjView; //= eyeProjection*eyeView
Matrix4x4 invEyeProjView; //= eyeProjView^(-1)
Matrix4x4 visView; // visualisation view matrix

//mouse navigation state
int xEyeBegin, yEyeBegin, yMotionBegin, verticalMotionBegin, horizontalMotionBegin = 0;
int renderMode = RenderTraverser::RENDER_COHERENT;

// relative size of an object
float objectSize = 3.5f;

int numObjects = 2000;
int numNextObjects = numObjects;
float zLength = 30.0f;

// this defines the volume where objects can be drawn
Vector3 minTranslation = {-2.5f, -2.5f, -3.0f};
Vector3 maxTranslation = {2.5f, 2.5f, -3.0 - zLength};

const float minAngle = 0;
const float maxAngle = 360;

const int renderTimeSize = 100;
long renderTimes[renderTimeSize];
int renderTimesIdx = 0;
int renderTimesValid = 0;

typedef vector<Geometry *> GeometryList;
GeometryList geometry;

// optimization to directly render geometry for occlusion query
bool useOptimization = false;

Vector3 amb[2];
Vector3 dif[2];
Vector3 spec[2];

void begin2D(void);
void end2D(void);
void output(const int x, const int y, const char *string);
void initGLstate(void);
void keyboard(const unsigned char c, const int x, const int y);
void drawHelpMessage(void);
void drawStatistics(void);
void display(void);
void special(const int c, const int x, const int y);
void reshape(const int w, const int h);
void mouse(int button, int state, int x, int y);
void initExtensions(void);
void leftMotion(int x, int y);
void rightMotion(int x, int y);
void middleMotion(int x, int y);
void drawEyeView(void);
void setupEyeView(void);
void setupVisView(void);
void updateEyeMtx(void);
long calcRenderTime(void);
void resetTimer(void);
void cleanUp(void);
void calcDecimalPoint(string &str, int d);

HierarchyNode* generateHierarchy(int numObjects);
Geometry *generateGeometry(Vector3 translateRatio, float xRotRatio, 
						   float yRotRatio, float zRotRatio, int materialIdx);
void displayVisualization();

void deleteGeometry();


int _tmain(int argc, _TCHAR* argv[])
{
	glutInitWindowSize(800,600);
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	
	glutCreateWindow("Coherent Hierarchical Culling");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutIdleFunc(display);
	initExtensions();
	initGLstate();

	leftMotion(0,0);
	middleMotion(0,0);
	
	HierarchyNode *hierarchy = generateHierarchy(numObjects);
	traverser.SetHierarchy(hierarchy);

	/// initialise rendertime array
	for(int i=0; i<renderTimeSize; i++)
		renderTimes[i] = 0;

	glutMainLoop();

	// clean up
	cleanUp();
	
	return 0;
}


void initGLstate(void) 
{
	glClearColor(0.6, 0.6, 0.8, 1.0);
	
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT,1); 
	
	glDepthRange(0.0, 1.0);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glShadeModel(GL_SMOOTH);
	
	GLfloat ambient[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat position[] = { 0.0, 3.0, 3.0, 0.0 };
    
	GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
	GLfloat local_view[] = { 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, position);
	
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, local_view);

	glMaterialf(GL_FRONT, GL_SHININESS, 64);
	glEnable(GL_NORMALIZE);
			
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);
	
	glClearColor(0.2, 0.2, 0.8, 0.0);

	setupVisView();
}


void drawHelpMessage(void) 
{
	const char *message[] = 
	{
		"Help information",
		"",
		"'F1'           - shows/dismisses this message",
		"'F2'           - decreases number of objects (valid after scene recreation)",
		"'F3'           - increases number of objects (valid after scene recreation)",
		"'F4'           - decreases box length in z direction (valid after scene recreation)",
		"'F5'           - increases box length in z direction (valid after scene recreation)",
		"'F6'           - decreases object size (valid after scene recreation)",
		"'F7'           - increases object size (valid after scene recreation)",
		"'F8'           - cycles through object types (teapot, ...) (valid after scene recreation)",
		"",
		"'MOUSE-LEFT'   - turn left/right, move forward/backward",
		"'MOUSE-RIGHT'  - turn left/right, tilt up/down",
		"'MOUSE-MIDDLE' - move up/down, left/right",
		"'CURSOR UP'    - move forward",
		"'CURSOR BACK'  - move backward",
		"'CURSOR RIGHT' - turn right",
		"'CURSOR LEFT'  - turn left",
		"",
		"'SPACE'        - cycles through occlusion culling algorithms",
		"'-'            - decreases visibility threshold",
		"'+'            - increases visibility threshold",
		"'C'            - recreates the scene hierarchy",
		"'G'            - enables/disables optimization to take geometry as occluder",
		"",
		"'R'            - shows/hides recreation parameters",
		"'S'            - shows/hides statistics",
		"'V'            - shows/hides bounding volumes",
		"",
		"'1'            - shows/hides visualization",
		"'2'            - zooms out visualization",
		"'3'            - zooms in visualization",
		0,
	};
	
	int i;
	int x = 40, y = 42;
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glColor4f(0.0,1.0,0.0,0.2);  // 20% green. 

	// Drawn clockwise because the flipped Y axis flips CCW and CW. 
	glRecti(winWidth - 30, 30, 30, winHeight - 30);
	
	glDisable(GL_BLEND);
	

	glColor3f(1.0,1.0,1.0);
	for(i = 0; message[i] != 0; i++) {
		if(message[i][0] == '\0') {
			y += 7;
		} else {
			output(x,y,message[i]);
			y += 14;
		}
	}

}

// generates a teapot, all parameters between zero and one
Geometry *generateGeometry(Vector3 translationRatio, float xRotRatio, 
						   float yRotRatio, float zRotRatio, int materialIdx)
{
	float xRot = minAngle + xRotRatio * (maxAngle - minAngle);
	float yRot = minAngle + yRotRatio * (maxAngle - minAngle);
	float zRot = minAngle + zRotRatio * (maxAngle - minAngle);

	Vector3 translation;
	translation[0] = minTranslation[0] + translationRatio[0] * (maxTranslation[0] - minTranslation[0]);
	translation[1] = minTranslation[1] + translationRatio[1] * (maxTranslation[1] - minTranslation[1]);
	translation[2] = minTranslation[2] + translationRatio[2] * (maxTranslation[2] - minTranslation[2]);

	Geometry *result = new Geometry(translation, xRot, yRot, zRot, objectSize, objectType);

	result->SetAmbientColor(amb[materialIdx][0], amb[materialIdx][1], amb[materialIdx][2]);
	result->SetDiffuseColor(dif[materialIdx][0], dif[materialIdx][1], dif[materialIdx][2]);
	result->SetSpecularColor(spec[materialIdx][0], spec[materialIdx][1], spec[materialIdx][2]);

	return result;
}

// generates a the scene hierarchy with random values
HierarchyNode* generateHierarchy(int numObjects)
{
    HierarchyNode *hierarchy = new HierarchyNode();

	// initialise materials
	copyVector3Values(amb[0], 0.0215, 0.1745, 0.0215);
	copyVector3Values(dif[0], 0.727811, 0.633, 0.6);
	copyVector3Values(spec[0], 0.633, 0.727811, 0.633);

	copyVector3Values(amb[1], 0.1745, 0.01175, 0.01175);
	copyVector3Values(dif[1], 0.61424, 0.04136, 0.04136);
	copyVector3Values(spec[1], 0.727811, 0.626959, 0.626959);

	srand (time (0));
	
	printf("generating geometry with random position and orientation ... ");

	for(int i=0; i < numObjects; i++)
	{
		float xRotRatio = rand() / (float) RAND_MAX;
		float yRotRatio = rand() / (float) RAND_MAX;
		float zRotRatio = rand() / (float) RAND_MAX;

		Vector3 translationRatio = {rand() / (float) RAND_MAX, 
									rand() / (float) RAND_MAX, 
									rand() / (float) RAND_MAX};

		int materialIdx = int(2 * rand() / (float) RAND_MAX);
			
		Geometry *geo = generateGeometry(translationRatio, xRotRatio, 
									     yRotRatio, zRotRatio, materialIdx);
		hierarchy->AddGeometry(geo);

		// put into global geometry list for later deletion
		geometry.push_back(geo);
	}

	printf("finished\n");
	printf("generating new kd-tree hierarchy ... ");
	HierarchyNode::InitKdTree(hierarchy);
	hierarchy->GenerateKdTree();
	printf("finished\n");

	return hierarchy;
}


void updateEyeMtx(void) 
{
	const Vector3 up = {0.0, 1.0, 0.0};

	look(eyeView, eyePos, viewDir, up);
	mult(eyeProjView, eyeProjection, eyeView); //eyeProjView = eyeProjection*eyeView
	invert(invEyeProjView, eyeProjView); //invert matrix
}


void setupEyeView(void)
{
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(eyeProjection);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixd(eyeView);
}


void display(void) 
{
	char * msg[] = {"Frustum culling only", "Hierarchical stop and wait",
				    "Coherent hierarchical culling"};

	char msg2[200];
	char msg3[200];
	char msg4[100];
	char msg5[100];
	char msg6[100];
	char msg7[100];
	char msg8[100];

	sprintf(msg2, "Traversed: %4d, frustum culled: %4d, query culled: %4d (of %d nodes)",
			traverser.GetNumTraversedNodes(), traverser.GetNumFrustumCulledNodes(),
			traverser.GetNumQueryCulledNodes(), 
			traverser.GetHierarchy()->GetNumHierarchyNodes());
	
	char *optstr[2] = {"", ", using optimization"};
	
	float fps = 1000.0;
	long renderTime = calcRenderTime();
	if(renderTime) fps = 1000000.0f / (float)calcRenderTime();

	sprintf(msg3, "Threshold: %4d, algorithm rendering time: %ld ms (%3.3f fps)%s", 
			traverser.GetVisibilityThreshold(), renderTime / 1000, fps, optstr[useOptimization]);

	string str;
	string str2;

	calcDecimalPoint(str, Geometry::CountTriangles(objectType) * traverser.GetNumRenderedGeometry());
	calcDecimalPoint(str2, Geometry::CountTriangles(objectType) * numObjects);

	sprintf(msg4, "Rendered objects %d (of %d), rendered triangles: %s (of %s)", 
			traverser.GetNumRenderedGeometry(), numObjects, str.c_str(), str2.c_str()); 

	sprintf(msg5, "Next object num: %d", numNextObjects);
	sprintf(msg6, "Next length in z dir: %3.3f", zLength);
	
	char *objectTypeStr[3] = {"teapot", "torus", "sphere"};

	sprintf(msg7, "Next object type: %s", objectTypeStr[nextObjectType]);
	sprintf(msg8, "Next object size: %3.3f", objectSize);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	updateEyeMtx(); // bring eye modelview matrix up-to-date
	setupEyeView();

	traverser.SetViewpoint(eyePos);
	traverser.SetProjViewMatrix(eyeProjView);
	traverser.Render(renderMode);

	// cycle through rendertime array
	renderTimes[renderTimesIdx] = traverser.GetRenderTime();
	renderTimesIdx = (renderTimesIdx + 1) % renderTimeSize;

	if(renderTimesIdx  > renderTimesValid) 
		renderTimesValid = renderTimesIdx;

	if(visMode)
	{
		displayVisualization();
	}
	
	begin2D();
	if(showHelp)
		drawHelpMessage();
	else
	{
		glColor3f(1.0,1.0,1.0);
		output(10,winHeight-10, msg[renderMode]);

		if(showStatistics)
		{
			if(showCreateParams)
			{
				output(10, winHeight-150, msg8);
				output(10, winHeight-130, msg7);
				output(10, winHeight-110, msg6);
				output(10, winHeight-90,  msg5);
			}
			output(10, winHeight-70, msg3);
			output(10, winHeight-50, msg4);
			output(10, winHeight-30, msg2);
		}
	}
	end2D();

	glutSwapBuffers();
}


#pragma warning( disable : 4100 )
void keyboard(const unsigned char c, const int x, const int y) 
{
	int threshold;
	HierarchyNode *hierarchy;

	switch(c) 
	{
	case 27:
		exit(0);
		break;
	case 32: //space
		renderMode = (renderMode + 1) % RenderTraverser::NUM_RENDERMODES;
		
		resetTimer();
		traverser.Render(renderMode);		// render once so stats are updated
		break;
	case 'h':
	case 'H':
		showHelp = !showHelp;
		break;
	case 'v':
	case 'V':
		showBoundingVolumes = !showBoundingVolumes;
		HierarchyNode::SetRenderBoundingVolume(showBoundingVolumes);
		break;
	case 's':
	case 'S':
		showStatistics = !showStatistics;
		break;
	case '+':
		threshold = traverser.GetVisibilityThreshold() + 10;
		traverser.SetVisibilityThreshold(threshold);
		break;
	case '-':
		threshold = traverser.GetVisibilityThreshold() - 10;
		if(threshold < 0) threshold = 0;

		traverser.SetVisibilityThreshold(threshold);		
		break;
	case '1':
		visMode = !visMode;
		break;
	
	case '2':
		visZoomFactor += 0.1;	
		setupVisView();
		break;
	case '3':
		visZoomFactor -= 0.1;
		if(visZoomFactor < 0.1) visZoomFactor = 0.1;
	
		setupVisView();
		break;
	case 'r':
	case 'R':
		showCreateParams = !showCreateParams;
		break;
	case 'g':
	case 'G':
		useOptimization = !useOptimization;
		traverser.SetUseOptimization(useOptimization);
		break;
	case 'c':
	case 'C':	
		
		hierarchy = traverser.GetHierarchy();
		// delete old hierarchy
		if(hierarchy) delete hierarchy; 
		deleteGeometry();

		maxTranslation[2] = -3.0 - zLength;
		numObjects = numNextObjects;
		objectType = nextObjectType;
		
		hierarchy = generateHierarchy(numObjects);
		traverser.SetHierarchy(hierarchy);

		showCreateParams = false;

		traverser.Render(renderMode); // render once to update stats
		resetTimer();
		break;

	default:
		return;
	}

	glutPostRedisplay();
}


void special(const int c, const int x, const int y) 
{
	switch(c) 
	{
	case GLUT_KEY_F1:
		showHelp = !showHelp;
		break;
	case GLUT_KEY_F2:
		numNextObjects -= 100;
		if(numNextObjects < 100) numNextObjects = 100;
		break;
	case GLUT_KEY_F3:
        numNextObjects += 100;
		break;
	case GLUT_KEY_F4:
		zLength -= 1;
		if(zLength < 0) zLength = 0;
		break;
	case GLUT_KEY_F5:
		zLength += 1;
		break;		
	case GLUT_KEY_F6:
		objectSize -= 0.1;
		if(objectSize < 0.1) objectSize = 0.1;
		break;
	case GLUT_KEY_F7:
		objectSize += 0.1;
		break;
	case GLUT_KEY_F8:
		nextObjectType = (nextObjectType + 1) % Geometry::NUM_OBJECTS;
		break;
	case GLUT_KEY_LEFT:
		rotateVectorY(runDir, 0.2);
		rotateVectorY(viewDir, 0.2);
		break;
	case GLUT_KEY_RIGHT:
		rotateVectorY(runDir, -0.2);
		rotateVectorY(viewDir, -0.2);
		break;
	case GLUT_KEY_UP:
		linCombVector3(eyePos, eyePos, runDir, 0.6);
		break;
	case GLUT_KEY_DOWN:
		linCombVector3(eyePos, eyePos, runDir, -0.6);
		break;
	default:
		return;

	}

	glutPostRedisplay();
}
#pragma warning( default : 4100 )


void reshape(const int w, const int h) 
{
	double winAspectRatio = 1.0;

	glViewport(0, 0, w, h);
	
	winWidth = w;
	winHeight = h;

	if(w) winAspectRatio = (double) h / (double) w;

	perspectiveDeg(eyeProjection, 60.0, 1.0/winAspectRatio, nearDist, 150.0);

	glutPostRedisplay();
}


void mouse(int button, int state, int x, int y) 
{
	if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) 
	{
		xEyeBegin = x;
		yMotionBegin = y;

		glutMotionFunc(leftMotion);
	}
	else if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		xEyeBegin = x;
		yEyeBegin = y;

		glutMotionFunc(rightMotion);
	}
	else if(button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
	{
		horizontalMotionBegin = x;
		verticalMotionBegin = y;

		glutMotionFunc(middleMotion);
	}

	glutPostRedisplay();
}

/**
	rotation for left/right mouse drag
	motion for up/down mouse drag
*/
void leftMotion(int x, int y) 
{
	static double eyeXAngle = 0.0;
	
	eyeXAngle = 0.6 *  PI * (xEyeBegin - x) / 180.0;
	linCombVector3(eyePos, eyePos, runDir, (yMotionBegin - y) * 0.1);

	rotateVectorY(runDir, eyeXAngle);
	rotateVectorY(viewDir, eyeXAngle);

	xEyeBegin = x;
	yMotionBegin = y;

	glutPostRedisplay();
}

// rotate and tilt
void rightMotion(int x, int y) 
{
	static double eyeYAngle = 0.0;
	static double eyeXAngle = 0.0;
	
	eyeXAngle = 0.6 * PI * (xEyeBegin - x) / 180.0;
	
	rotateVectorY(runDir, eyeXAngle);
	rotateVectorY(viewDir, eyeXAngle);

	xEyeBegin = x;

	// the 90 degree rotated view vector is the axis of rotation
	Vector3 rVec = {runDir[0], runDir[1], runDir[2]};
	
	rotateVectorY(rVec, PI / 2.0);
	normalize(rVec);

	eyeYAngle = - 0.6 * PI * (yEyeBegin - y) / 180.0;

	Vector3 hview;

	rotateVector(hview, viewDir, eyeYAngle, rVec);

	if(fabs(hview[1]) < 0.95) // should not look directly down/up because of singularity
		copyVector3(viewDir, hview);
    
	yEyeBegin = y;

	glutPostRedisplay();
}


// strafe
void middleMotion(int x, int y) 
{
	// the 90 degree rotated view vector 
	
	Vector3 rVec = {runDir[0], runDir[1], runDir[2]};
	
	rotateVectorY(rVec, PI / 2.0);
	linCombVector3(eyePos, eyePos, rVec, (horizontalMotionBegin - x) * 0.1);
	
	eyePos[1] += (verticalMotionBegin - y) * 0.1;

	horizontalMotionBegin = x;
	verticalMotionBegin = y;

	glutPostRedisplay();
}


void initExtensions(void) 
{
	GLenum err = glewInit();
	if(GLEW_OK != err) {
		// problem: glewInit failed, something is seriously wrong
		fprintf(stderr,"Error: %s\n",glewGetErrorString(err));
		exit(1);
	}

	if(!GLEW_ARB_occlusion_query) {
		printf("I require the GL_ARB_occlusion_query OpenGL extension to work.\n");
		exit(1);
	}
}


void begin2D(void) 
{
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, winWidth, winHeight, 0);
}


void end2D(void) 
{
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
}


void output(const int x, const int y, const char *string) 
{
	if(string != 0) 
	{
		int len, i;
		glRasterPos2f(x,y);
		len = (int) strlen(string);
		
		for (i = 0; i < len; i++) 
		{
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13,string[i]);
		}
	}
}

// explicitly deletes geometry generated for hierarchy 
// (deleting the hierarchy does not delete the geometry)
void deleteGeometry()
{
	for (GeometryList::iterator it = geometry.begin(); it != geometry.end(); it++)
		if(*it)	delete (*it);
	
	geometry.clear();
}

// displays the visualisation of the kd tree node culling
void displayVisualization()
{
	begin2D();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glColor4f(0.0,0.0,0.0,0.5); 

	glRecti(winWidth, 0, winWidth / 2, winHeight / 2);
	glDisable(GL_BLEND);
	end2D();

	glViewport(winWidth / 2, winHeight / 2, winWidth, winHeight);
	glPushMatrix();
	glLoadMatrixd(visView);
	
	glClear(GL_DEPTH_BUFFER_BIT);

	// --- visualization of the occlusion culling
	HierarchyNode::SetRenderBoundingVolume(true);
	traverser.RenderVisualization();
	HierarchyNode::SetRenderBoundingVolume(showBoundingVolumes);

	// --- render current viewpoint
	glPushMatrix();
	glMultMatrixd(invEyeProjView);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_LIGHTING);
	//glDisable(GL_CULL_FACE);
		
	glColor3f(1, 1, 0);
	glutWireCone(5, 0.9, 15, 10); 
	
	glPopMatrix();
	
	//glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	glPopMatrix();
	glViewport(0, 0, winWidth, winHeight);
}

/**
	sets up view matrix in order to get a good position 
	for viewing the kd tree node culling
*/
void setupVisView()
{
	const Vector3 up = {0.0, 1.0, 0.0};
	
	Vector3 visPos = {24, 23, -6};
	
	visPos[0] *= visZoomFactor; 
	visPos[1] *= visZoomFactor; 
	visPos[2] *= visZoomFactor;

	Vector3 visDir = {-1.3,-1,-1};
	
	normalize(visDir);
	look(visView, visPos, visDir, up);
}

// we take a couple of measurements and compute the average
long calcRenderTime()
{
	long result = 0;

	for(int i=0; i<renderTimesValid; i++)
		result += renderTimes[i];
    
	if(renderTimesValid)
		result /= renderTimesValid;

	return result;
}


// reset the timer array for a new traversal method
void resetTimer()
{
	renderTimesValid = 0;
	renderTimesIdx = 0;
}

// cleanup routine after the main loop
void cleanUp()
{
	if(traverser.GetHierarchy()) 
		delete traverser.GetHierarchy();

	deleteGeometry();

	Geometry::CleanUp();
}

// this function inserts a dezimal point after each 1000
void calcDecimalPoint(string &str, int d)
{
	vector<int> numbers;
	char hstr[100];

	while(d != 0)
	{
		numbers.push_back(d % 1000);
		d /= 1000;
	}

	// first element without leading zeros
	if(numbers.size() > 0)
	{
		sprintf(hstr, "%d", numbers.back());
		str.append(hstr);
	}
	
	for(int i=numbers.size()-2; i >= 0; i--)
	{
		sprintf(hstr, ",%03d", numbers[i]);
		str.append(hstr);
	}
}