#ifndef _LOAD_CUBEMAP_H_
#define _LOAD_CUBEMAP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glut.h>
#include <GL/glext.h>

#ifdef __cplusplus
extern "C" {
#endif

	void load_bmp_cubemap ( const char * string, int mipmap );

#ifdef __cplusplus
}
#endif

#endif /* _LOAD_CUBEMAP_H_ */