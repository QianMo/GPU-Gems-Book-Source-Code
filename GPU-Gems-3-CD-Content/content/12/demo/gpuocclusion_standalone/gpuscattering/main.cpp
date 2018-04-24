/*! \file main.cpp
 *  \author Yuntao Jia
 *  \brief Main file for gpuoscattering application.
 */

#include "ScatteringViewer.h"

int main(int argc, char **argv)
{
  ScatteringViewer v;
  ScatteringViewer::main(argc, argv, "GPU-Accelerated Multiple Scattering", &v);
} // end main()
