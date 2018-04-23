//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include <string>
#include <iostream>
#include "Main.h"
#include <Engine/GlInterface/PBuffer.h>
#include <Base/StringTools.h>
#include <Mathematic/Vector4.h>
#include <Mathematic/Matrix4Xform.h>
#include <Mathematic/MathTools.h>
#include <Mathematic/Average.h>
#include <Engine/WindowManager/KeyState.h>
#include "FrameState.h"

const std::string title = "Coherent Hierarchical Culling";
//class to deal with PBuffer allocation
PBuffer *buf = 0;

//handels drawing and contains the scene objects
Scene* scene = 0;

//handles keyboard state
Window::KeyState keyState;

//contains frame status information like frustum current pos statistics
FrameState frameState;

//used for averaged fps calculation
Math::Average<unsigned> frames(40);
Math::Average<unsigned> queries(40);

bool fly = false;
bool showHelp = false;
bool showKDTree = false;
bool showStats = true;

unsigned mode = 2;

Math::Matrix4d eyeView; // eye view matrix
Math::Matrix4d eyeProjection; // eye projection matrix
Math::Matrix4d eyeProjView; //= eyeProjection*eyeView
Math::Matrix4d invEyeProjView; //= eyeProjView^(-1)
Math::Geometry::Frustum<double> frustum;

Math::Matrix4d lightView;
Math::Matrix4d lightProjection;

unsigned frame = 0;
unsigned lastFrameTime = 50;
double nearDist = 0.1;
double farDist = 70.0;
const V3 up(0.0,1.0,0.0);
V3 eyePos(0.0, 0.0, 0.0);  // eye position 
V3 viewDir(0.0,0.0,-1.0);  // eye view dir 
V3 lightDir(0.0,-0.99,0.01);  // light dir 

int winWidth, winHeight;

unsigned depthMapSize = 2048;
GLuint depthTexture = 0;

//mouse navigation state
bool rotatingEye = false;
bool movingLight = false;
bool movingEye = false;
int xEyeBegin, yEyeBegin, eyeMove;
int xLightBegin, yLightBegin;

const GLfloat globalAmbient[] = {0.5, 0.5, 0.5, 1.0};

enum TView {
	EYE,
	EYE_SHADOWED,
	LIGHT
} view = EYE;

void prepareShadowMapping(void) {
	glActiveTextureARB(GL_TEXTURE1);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_GEN_S);
	glEnable(GL_TEXTURE_GEN_T);
	glEnable(GL_TEXTURE_GEN_R);
	glEnable(GL_TEXTURE_GEN_Q);

	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	
	glTexGendv(GL_S, GL_EYE_PLANE, Math::Vector4d::UNIT_X.addr());
	glTexGendv(GL_T, GL_EYE_PLANE, Math::Vector4d::UNIT_Y.addr());
	glTexGendv(GL_R, GL_EYE_PLANE, Math::Vector4d::UNIT_Z.addr());
	glTexGendv(GL_Q, GL_EYE_PLANE, Math::Vector4d::UNIT_W.addr());

	//in this function the different shadow algorithms are called
	updateLightMtx(scene->getAABox());

	glMatrixMode(GL_TEXTURE);
	glLoadMatrixd(Math::Matrix4d::T05_S05.addr());
	glMultMatrixd(lightProjection.addr());
	glMultMatrixd(lightView.addr());
	glMatrixMode(GL_MODELVIEW);

	glBindTexture(GL_TEXTURE_2D,depthTexture);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_COMPARE_MODE_ARB,GL_COMPARE_R_TO_TEXTURE_ARB);

	glActiveTextureARB(GL_TEXTURE0);
}

void unprepareShadowMapping(void) {
	glActiveTextureARB(GL_TEXTURE1);
	//Disable textures and texgen
	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_COMPARE_MODE_ARB,GL_NONE);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_GEN_S);
	glDisable(GL_TEXTURE_GEN_T);
	glDisable(GL_TEXTURE_GEN_R);
	glDisable(GL_TEXTURE_GEN_Q);
	glActiveTextureARB(GL_TEXTURE0);
}
void setupLightView(void) {
	updateLightMtx(scene->getAABox());

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(lightProjection.addr());

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixd(lightView.addr());
}

void updateEyeMtx(void) {
	Math::look(eyeView,eyePos,viewDir,up);
	eyeProjView = eyeProjection*eyeView;
	invEyeProjView.invert(eyeProjView);
}

void setupEyeView(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(eyeProjection.addr());

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixd(eyeView.addr());
}

void drawEyeView(void) {
	const Math::Vector4f lightPos(-lightDir[0], -lightDir[1], -lightDir[2], 0.0);
	glLightfv(GL_LIGHT0,GL_POSITION,lightPos.addr());
	Frustum f(eyeProjView.convert2<float>());
	//contains frustum and gl state manager
	frameState = FrameState(frame,eyePos.convert2<float>(),f,mode);
	scene->draw(frameState);
}

void drawLightView(void) {
	const Math::Vector4f lightPos(-lightDir[0], -lightDir[1], -lightDir[2], 0.0);
	glLightfv(GL_LIGHT0,GL_POSITION,lightPos.addr());
	glDisable(GL_FOG);
	Frustum f(eyeProjView.convert2<float>());
	//contains frustum and gl state manager
	frameState = FrameState(frame,eyePos.convert2<float>(),f,mode);
	scene->draw(frameState);

	glDisable(GL_LIGHTING);
	//draw view vector in red
	glColor3f(1,0,0);
	glBegin(GL_LINES);
		const V3 startPos(eyePos);
		const V3 dir(viewDir);
		glVertex3dv((startPos+dir*nearDist).addr());
		glVertex3dv((startPos+dir*farDist).addr());
	glEnd();

	//draw view frustum
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	
	glEnable(GL_BLEND);

	Vector3x8 v;
	Math::calcViewFrustumWorldCoord(v,invEyeProjView);
	glColor4f(1,1,1,0.5);
	drawBoxVolume(v);
	glDisable(GL_BLEND);

	glColor4f(0,0,0,1);
	glEnable(GL_POLYGON_OFFSET_LINE);
	drawLineBoxVolume(v);
	glDisable(GL_POLYGON_OFFSET_LINE);

	glEnable(GL_LIGHTING);
	glEnable(GL_FOG);
}

void setupKDTreeView(void) {
	Math::Matrix4d treeView;
	Math::look<double>(treeView,eyePos,V3(0.0,-1.0,0.0),V3::UNIT_X);
	StaticArray<AABox::V3,8> p;
	scene->getAABox().computeVerticesLeftHanded(p);
	AABox box(treeView,p);
	Math::Matrix4d treeProj;
	Math::scaleTranslateToFit(treeProj,box.getMin(),box.getMax());

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(treeProj.addr());
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixd(treeView.addr());
}

void drawKDTreeView(void) {
	glDisable(GL_FOG);
	glDisable(GL_LIGHTING);
	Frustum f(eyeProjView.convert2<float>());
	//contains frustum and gl state manager
	frameState = FrameState(frame,eyePos.convert2<float>(),f,mode);
	scene->drawKDTree(frameState);
	glEnable(GL_LIGHTING);
	glEnable(GL_FOG);
}

void generateDepthTexture(void) {
	buf->Bind();
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);

	setupLightView();
	//contains frustum and gl state manager
	frameState = FrameState(
		frame,
		Math::Vector3f::ONE*Math::Const<float>::infinity(),
		Frustum((lightProjection*lightView).convert2<float>()),
		mode
	);
	scene->drawShadowCasters(frameState);

	glBindTexture(GL_TEXTURE_2D,depthTexture);
	glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,depthMapSize,depthMapSize);
	glDisable(GL_POLYGON_OFFSET_FILL);
	buf->Unbind();
}

void output(const int x, const int y, const std::string& sz) {
	if(sz.empty()) {
		return;
	}
	glRasterPos2f(x,y);
	for(std::string::const_iterator i = sz.begin(); i != sz.end(); i++) {
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13,*i);
	}
}


void multiLineOutput(const int x, const int y, const std::string& sz, const unsigned lineheight = 20) {
	if(sz.empty()) {
		return;
	}
	int newY = y;
	glRasterPos2f(x,newY);
	for(std::string::const_iterator i = sz.begin(); i != sz.end(); i++) {
		if('\n' == *i) {
			newY += lineheight;
			glRasterPos2f(x,newY);
		}
		else {
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13,*i);
		}
	}
}

void begin2D(void) {
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,winWidth,winHeight,0);
}


void end2D(void) {
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
}

void drawHelpMessage(void) {
	const char *message[] = {
		"Help information",
		"",
		"'MOUSE-LEFT'         - eye look left/right move forward/backward",
		"'MOUSE-RIGHT'        - eye look up/down left/right",
		"'MOUSE-RIGHT + CTRL' - light direction angle incr/decr rotate cw/ccw",
		"'UP'                 - move forward",
		"'DOWN'               - move backward",
		"'LEFT'               - strafe left",
		"'RIGHT'              - strafe right",
		"'+'                  - move up (works only in flying mode)",
		"'-'                  - move down (works only in flying mode)",
		"",
		"'7'                  - set depth(test) texture size to 256x256",
		"'8'                  - set depth(test) texture size to 512x512",
		"'9'                  - set depth(test) texture size to 1024x1024",
		"'0'                  - set depth(test) texture size to 2048x2048",
		"",
		"'F1'                 - shows and dismisses this message",
		"'F2'                 - show eye view with shadows",
		"'F3'                 - show light view",
		"'F4'                 - show eye view without shadows",
		"",
		"'F'                  - toggle flying mode",
		"'S'                  - toggle show statistics",
		"'1'                  - toggle show culling tree",
		"'SPACE'              - cycle through the culling modes",
		0
	};
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glColor4f(0.0,1.0,0.0,0.2);  /* 20% green. */

	/* Drawn clockwise because the flipped Y axis flips CCW and CW. */
	glRecti(winWidth-30,30,30,winHeight-30);

	glDisable(GL_BLEND);

	glColor3f(1.0,1.0,1.0);
	int x = 40, y = 42;
	for(unsigned i = 0; message[i] != 0; i++) {
		if(message[i][0] == '\0') {
			y += 7;
		} else {
			output(x,y,message[i]);
			y += 14;
		}
	}

}

void handleInput(void) {
	using Window::KeyState;
	const float fact = lastFrameTime/300.0f;
	if(keyState.isKeyDown('+')) {
		eyePos[1] += fact;
	}
	else
	if(keyState.isKeyDown('-')) {
		eyePos[1] -= fact;
	}
	else
	if(keyState.isKeyDown(KeyState::KEY_UP_ARROW)) {
		eyePos += viewDir*fact;
	}
	else
	if(keyState.isKeyDown(KeyState::KEY_DOWN_ARROW)) {
		eyePos -= viewDir*fact;
	}
	else
	if(keyState.isKeyDown(KeyState::KEY_LEFT_ARROW)) {
		eyePos += V3().unitCross(up,viewDir)*fact;
	}
	else
	if(keyState.isKeyDown(KeyState::KEY_RIGHT_ARROW)) {
		eyePos -= V3().unitCross(up,viewDir)*fact;
	}
}

const std::string addNumberDel(const std::string sz) {
	std::string o;
	for(int i = sz.length()-1, unsigned pos = 0; i >= 0; i--, pos++) {
		if(0 == pos % 3  && pos != 0) {
			o = ','+o;
		}
		o = sz[i]+o;
	}
	return o;
}

std::string getState(void) {
	const unsigned frameTime = frames.avg();
	std::string m;
	m += Tools::toString(frameTime);
	m += " ms";
	if(frameTime > 0) {
		const float fps = 1000.0/frameTime;
		const float mult = 0.5;
		frames.setSampleCount(mult*fps);
		queries.setSampleCount(mult*fps);
		m += " (FPS:";
		m += Tools::toString(fps,4);
		m += ')';
	}

	queries.add(frameState.query_cnt);
	m += '\n';
	m += "Rendered objects: ";
	m += addNumberDel(Tools::toString(frameState.object_cnt));
	m += " (of ";
	m += addNumberDel(Tools::toString(scene->getObjectCount()));
	m += "), rendered triangles: ";
	m += addNumberDel(Tools::toString(frameState.triangle_cnt));
	m += " (of ";
	m += addNumberDel(Tools::toString(scene->getSceneTriangleCount()));
	m += ")\n";
	m += "Traversed: ";
	m += addNumberDel(Tools::toString(frameState.traversed_nodes_cnt));
	m += ", frustum culled: ";
	m += addNumberDel(Tools::toString(frameState.frustum_culled_nodes_cnt));
	m += ", queries issued: ";
	m += addNumberDel(Tools::toString(queries.avg()));
	m += " (of ";
	m += addNumberDel(Tools::toString(scene->getNodeCount()));
	m += " nodes)";
	return m;
}

unsigned getLineCount(const std::string sz) {
	unsigned lines = 1;
	for(unsigned i = 0; i < sz.length(); i++) {
		if('\n' == sz[i]) {
			lines++;
		}
	}
	return lines;
}

void displayMsg(const std::string& msgIn) {
	begin2D();
	if(showHelp) {
		drawHelpMessage();
	}
	else {
		std::string m = msgIn;
		glColor3f(1.0,1.0,1.0);
		if(showStats) {
			m += ', ';
			m += getState();
		}
		m += '\n';
		switch(mode) {
			case 0: m += "View Frustum Culling"; break;
			case 1: m += "Stop & Wait"; break;
			case 2: m += "Coherent hierarchical culling"; break;
		};
		multiLineOutput(10,winHeight-20*getLineCount(m)+10,m);
	}
	end2D();
}


void display(void) {
	handleInput();
	std::string msg;
	int start = glutGet(GLUT_ELAPSED_TIME);
	if(!fly) {
		eyePos[1] = scene->getHeight(eyePos.convert2<float>())+0.2;
	}
	updateEyeMtx(); //bring eye modelview matrix up-to-date

	switch(view) {
	case LIGHT:
		setupLightView();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawLightView();
		msg += "Light view";
		break;
	case EYE:
		setupEyeView();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawEyeView();
		msg += "Eye view";
		break;
	case EYE_SHADOWED:
		generateDepthTexture();
		setupEyeView();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		prepareShadowMapping();
		drawEyeView();
		unprepareShadowMapping();
		msg += "Shadowed eye view";
		break;
	};


	lastFrameTime = glutGet(GLUT_ELAPSED_TIME)-start;
	frames.add(lastFrameTime);
	displayMsg(msg);

	if(showKDTree) {
		glViewport(winWidth-1/4.0*winWidth,winHeight-1/4.0*winHeight,winWidth/4,winHeight/4);
		setupKDTreeView();
		glClear(GL_DEPTH_BUFFER_BIT);
		drawKDTreeView();
		glViewport(0,0,winWidth,winHeight);
		msg += " with KDTree";
	}

	glutSwapBuffers();
	frame++;
}

void idle() {
	glutPostRedisplay();
}

void remakeTextures(const unsigned size);
#pragma warning( disable : 4100 )
void keyboardUp(const unsigned char c, const int x, const int y) {
	keyState.glutKeyUp(c);
}

void keyboard(const unsigned char c, const int x, const int y) {
	keyState.glutKeyPress(c);
	switch(c) {
	case 27:
		delete scene;
		delete buf;
		exit(0);
		break;
	case '7':
		remakeTextures(256);
		break;
	case '8':
		remakeTextures(512);
		break;
	case '9':
		remakeTextures(1024);
		break;
	case '0':
		remakeTextures(2048);
		break;
	case 'h':
	case 'H':
		showHelp = !showHelp;
		break;
	case 's':
	case 'S':
		showStats = !showStats;
		break;
	case ' ':
		mode++; 
		if(mode > 2) mode = 0;
		break;
	case '1':
		showKDTree = !showKDTree;
		break;
	case 'f':
	case 'F':
		fly = !fly;
		break;
	default:
		return;
	}
	glutPostRedisplay();
}

void specialUp(const int c, const int x, const int y) {
	keyState.glutSpecialKeyUp(c);
}

void special(const int c, const int x, const int y) {
	keyState.glutSpecialKeyPress(c);
	switch(c) {
	case GLUT_KEY_F1:
		showHelp = !showHelp;
		break;
	case GLUT_KEY_F2:
		view = EYE_SHADOWED;
		break;
	case GLUT_KEY_F3:
		view = LIGHT;
		break;
	case GLUT_KEY_F4:
		view = EYE;
		break;
	default:
		return;
	}
	glutPostRedisplay();
}
#pragma warning( default : 4100 )

void reshape(const int w, const int h) {
	winWidth = w;
	winHeight = h;
	glViewport(0,0,w,h);
	glScissor(0,0,w,h);
	const double winAspectRatio = (double)winWidth / (double)winHeight;
	perspectiveDeg(eyeProjection,60.0,winAspectRatio,nearDist,farDist);
	glutPostRedisplay();
	std::string title = ::title;
	title += ' ';
	title += Tools::toString(glutGet(GLUT_WINDOW_WIDTH));
	title += 'x';
	title += Tools::toString(glutGet(GLUT_WINDOW_HEIGHT));
	glutSetWindowTitle(title.c_str());
}

void mouse(int button, int state, int x, int y) {
	keyState.glutMouseButton(button,state);
	const bool buttonA = GLUT_LEFT_BUTTON == button;
	const bool buttonB = GLUT_RIGHT_BUTTON == button;
	const bool down = GLUT_DOWN == state;

	if(buttonA && !keyState.isKeyDown(Window::KeyState::KEY_CTRL) && down) {
		movingEye = true;
		xEyeBegin = x;
		eyeMove = y;
	}
	if(buttonA && !down) {
		movingEye = false;
	}

	if(buttonB && !keyState.isKeyDown(Window::KeyState::KEY_CTRL) && down) {
		rotatingEye = true;
		xEyeBegin = x;
		yEyeBegin = y;
	}
	if(buttonB && keyState.isKeyDown(Window::KeyState::KEY_CTRL) && down) {
		movingLight = true;
		xLightBegin = x;
		yLightBegin = y;
	}
	if(buttonB && !down) {
		rotatingEye = false;
		movingLight = false;
	}

}

V3 setDir(const V3::ElementType& xAngle, const V3::ElementType& yAngle) {
	const double cosM = cos(yAngle);
	const double sinM = sin(yAngle);
	const double cosN = cos(xAngle);
	const double sinN = sin(xAngle);
	V3 dir;
	dir[0] = -cosM*sinN;
	dir[1] = sinM;
	dir[2] = -cosM*cosN;
	dir.normalize();
	return dir;
}

void motion(int x, int y) {
	const double PI_2 = Math::Const<double>::pi_2();
	static double eyeXAngle = 0.0;
	static double eyeYAngle = 0.0;
	if(rotatingEye) {
		eyeXAngle -= 0.015*(x - xEyeBegin);
		eyeYAngle -= 0.015*(y - yEyeBegin);
		Math::clamp(eyeYAngle,-PI_2+0.1,PI_2-0.1);
		xEyeBegin = x;
		yEyeBegin = y;
		viewDir = setDir(eyeXAngle,eyeYAngle);
	}
	if(movingEye) {
		const float fact = lastFrameTime/500.0f;
		eyeXAngle -= 0.015*(x - xEyeBegin);
		viewDir = setDir(eyeXAngle,eyeYAngle);
		eyePos += viewDir*(fact*(eyeMove-y));
		xEyeBegin = x;
		eyeMove = y;
	}
	if(movingLight) {
		static double lightXAngle = 0.0;
		static double lightYAngle = -0.99;
		lightXAngle -= 0.005*(x - xLightBegin);
		lightYAngle += 0.005*(y - yLightBegin);
		Math::clamp(lightYAngle,-PI_2+0.01,0.0);
		xLightBegin = x;
		yLightBegin = y;
		lightDir = setDir(lightXAngle,lightYAngle);
	}
	glutPostRedisplay();
}

void initExtensions(void) {
	GLenum err = glewInit();
	if(GLEW_OK != err) {
		/* problem: glewInit failed, something is seriously wrong */
		fprintf(stderr,"Error: %s\n",glewGetErrorString(err));
		exit(1);
	}

	if(!GLEW_ARB_multitexture) {
		printf("I require the GL_ARB_multitexture OpenGL extension to work.\n");
		exit(1);
	}
	if(!GLEW_ARB_occlusion_query) {
		printf("I require the GLEW_ARB_occlusion_query OpenGL extension to work.\n");
		exit(1);
	}
}
void initGLstate(void) {
	const GLfloat lightColor[] = {1.0, 1.0, 1.0, 1.0};
	const GLfloat fogColor[] = {0.6, 0.6, 0.8, 1.0};
	glClearColor(0.6,0.6,0.8,1.0);
	
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glPixelStorei(GL_PACK_ALIGNMENT,1);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbient);

	glClearDepth(1.0);
	glDepthRange(0.0,1.0);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT, Math::Vector4f::UNIT_W.addr());
	glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);

	glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glMaterialf(GL_FRONT, GL_SHININESS, 0.0);

	glFogi(GL_FOG_MODE,GL_LINEAR);
	glFogf(GL_FOG_START,20.0);
	glFogf(GL_FOG_END,farDist);
	glFogfv(GL_FOG_COLOR,fogColor);
	glEnable(GL_FOG);

	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);

	glDisable(GL_TEXTURE_1D);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_3D);
	glEnable(GL_SCISSOR_TEST);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,GL_FALSE);
	glEnable(GL_NORMALIZE);

	//for each scene other values are suitable
	glPolygonOffset(2.0,4.0);

	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);
}

void makeDepthTexture() {
	//generate depth texture and set its properties
	glDeleteTextures(1,&depthTexture);
	glGenTextures(1,&depthTexture);
	glBindTexture(GL_TEXTURE_2D,depthTexture);
	glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT,depthMapSize,depthMapSize,
		0,GL_DEPTH_COMPONENT,GL_UNSIGNED_BYTE,0);

	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
}

void remakeTextures(const unsigned size) {
	depthMapSize = size;
	delete buf;
	buf = new PBuffer(depthMapSize,depthMapSize,0,0,0,0,false,false,true,false,false,true);
	if(!buf->IsValid()) {
		std::cout << "Failed to alloc PBuffer\n";
		exit(1);
	}
	buf->Bind();
		initGLstate();
		glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
		glShadeModel(GL_FLAT);
		glDisable(GL_LIGHTING);
		glDisable(GL_FOG);
	buf->Unbind();
	makeDepthTexture();
}

int main(int argc, char **argv) {
	glutInitWindowSize(800,600);
	glutInitWindowPosition(30,30);
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow(title.c_str());

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardUpFunc(keyboardUp);
	glutSpecialUpFunc(specialUp);
	glutIdleFunc(idle);

	initExtensions();
	initGLstate();

	scene = new Scene();
	remakeTextures(depthMapSize);

	movingLight = true;
	motion(20,20);
	movingLight = false;
	rotatingEye = true;
	motion(170,0);
	rotatingEye = false;

	glutMainLoop();
	delete buf;
	delete scene;
	return 0;
}
