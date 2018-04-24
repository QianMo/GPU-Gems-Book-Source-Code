/*! \file main.cpp
 *  \author Jared Hoberock
 *  \brief Main file for gpuocclusion application.
 */

#include "OcclusionViewer.h"

int main(int argc, char **argv)
{
  OcclusionViewer v;
  OcclusionViewer::main(argc, argv, "High-Quality Ambient Occlusion", &v);
} // end main()
