//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef OcclusionQueryH
#define OcclusionQueryH

#include "../Smp.h"
#include <GL/glHeader.h>
//---------------------------------------------------------------------------
class OcclusionQuery : public Base {
protected:
	GLuint id;

public:
	OcclusionQuery() {
		glGenQueries(1,&id);
	}

	~OcclusionQuery() {
		glDeleteQueries(1,&id);
	}
	void begin() const {
		glBeginQueryARB(GL_SAMPLES_PASSED_ARB,id);
	}

	void end() const {
		glEndQueryARB(GL_SAMPLES_PASSED_ARB);
	}

	bool finished() const {
		GLint available;
		glGetQueryObjectivARB(id, GL_QUERY_RESULT_AVAILABLE_ARB, &available);
		return TRUE == available;
	}

	unsigned getResult() const {	
		GLuint sampleCount;
		glGetQueryObjectuivARB(id, GL_QUERY_RESULT_ARB, &sampleCount);
		return sampleCount;
	}
};

#endif
