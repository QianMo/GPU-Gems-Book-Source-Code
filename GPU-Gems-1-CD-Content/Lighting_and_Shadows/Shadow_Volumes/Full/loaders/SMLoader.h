/**
  @file SMLoader.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)

*/

#ifndef _SM_LOADER_
#define _SM_LOADER_

#include <string>


/**
 * SM files are about as simple as you get.
 * It is a basic ascii format consisting of the following:
 *
 * line with num_vertices
 * num_vertices lines with vertices
 * line with num_faces
 * num_faces lines with vertex indices
 *
 * here is an example file that has three vertices and one face:
 *
 * 3
 * -1 0 0
 *  1 0 0
 *  0 1 0
 *
 * 1
 * 0 1 2
 */
BasicModel* importSMFile(
        const std::string&      filePath);

bool exportSMFile(
        const BasicModel&       model,
        const std::string&      filePath);


#endif

