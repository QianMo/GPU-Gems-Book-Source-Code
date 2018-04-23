/*
 * fxc.cpp
 *
 *  Handles compiling a shader to ps2.0
 */

#ifdef _WIN32
#pragma warning(disable:4786)
#endif
#include <sstream>
#include <iomanip>
#include <fstream>
extern "C" {
#include <stdio.h>
#include <string.h>
#include <assert.h>
}

#include "main.h"
#include "subprocess.h"
#include "project.h"
#include "fxc.h"

#include "ps2arb.h"

/*
 * compile_fxc --
 *
 *      Takes shader and runs it through the FXC compiler (and parses the
 *      results) to produce the corresponding fragment program.
 */

char *
compile_fxc (const char *name, 
             const char *shader, 
             CodeGenTarget target, 
             ShaderResourceUsage* outUsage,
             bool inValidate) {

  static const int kInputFileArgument = 5;
  static const int kOutputFileArgument = 4;
  char validate[]="/Vd";
  //char software[]="/Tps_2_sw";
  char hardware[]="/Tps_2_0";
  char ps2b[]="/Tps_2_b";
  char ps2a[]="/Tps_2_a";
  char ps30_targetstring[]="/Tps_3_0";
  char nothin[]=""; //gcc does not like ?: with ""

  char* targetstring = "";
  bool targetUsesRect = false;
  switch (target) {
  case CODEGEN_PS2B:
    targetstring = ps2b;
    break;
  case CODEGEN_PS2A:
    targetstring = ps2a;
    break;
  case CODEGEN_ARB:
    targetUsesRect = true;
    switch (globals.arch) {
    case GPU_ARCH_X800:
      targetstring = ps2b;
      break;
    case GPU_ARCH_6800:
      targetstring = ps2a;
      break;
    default:
      targetstring = hardware;
    }
    break;
  case CODEGEN_PS20:
     targetstring = hardware;
     break;
  case CODEGEN_PS30:
     targetstring = ps30_targetstring;
     break;
  default:
     fprintf(stderr, "Unsupported fxc target.\n");
     return NULL;
  }
  char DUSERECT [] ="/DUSERECT=1";
  char *argv[] = { "fxc", targetstring,
                   inValidate ? nothin : validate, "/nologo", 0, 0, 
                   "/DDXPIXELSHADER=1", targetUsesRect ? DUSERECT : 0, NULL };
  char *fpcode,  *errcode;

  std::string inputfname  = std::string(name) + ".cg";
  std::string outputfname = std::string(name) + ".ps";

  FILE *fp = fopen (inputfname.c_str(), "wb+");
  if (fp == NULL) {
     fprintf (stderr, "Unable to open tmp file %s\n", outputfname.c_str());
     return NULL;
  }
  fwrite(shader, sizeof(char), strlen(shader), fp);
  fclose(fp);
  
  argv[kOutputFileArgument] = strdup ((std::string("/Fc") + outputfname).c_str());
  argv[kInputFileArgument]  = strdup (inputfname.c_str());

  /* Run FXC */
  errcode = Subprocess_Run(argv, NULL);

  if (!globals.keepFiles)
     remove(inputfname.c_str());

  if (errcode == NULL) {
    fprintf(stderr, "%s resulted in an error,"
            "skipping ", argv[0]);

    switch (target) {
    case CODEGEN_PS2B:
       fprintf(stderr, "PS2B target.");
       break;
    case CODEGEN_PS2A:
       fprintf(stderr, "PS2A target.");
       break;
    case CODEGEN_PS20:
       fprintf(stderr, "PS20 target.");
       break;
    case CODEGEN_ARB:
       fprintf(stderr, "ARB target.");
       break;
    case CODEGEN_PS30:
       fprintf(stderr, "PS30 target.");
       break;
    default:
       break;
    }   
    fprintf(stderr, "\n");

    remove(argv[kOutputFileArgument]+3);
    free(argv[kOutputFileArgument]);
    free(argv[kInputFileArgument]);
    return NULL;
  }

  if (globals.verbose)
    fprintf(stderr, "FXC returned: [35;1m%s[0m\n",
            errcode);

  fp = fopen(argv[kOutputFileArgument]+3, "rt");
  if (fp == NULL) {
    fprintf (stderr, "Unable to open compiler output file %s\n", 
             argv[kOutputFileArgument]+3);
    fprintf(stderr, "FXC returned: [35;1m%s[0m\n",
            errcode);
    free(argv[kOutputFileArgument]);
    free(argv[kInputFileArgument]);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  long flen = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  fpcode = (char *) malloc (flen+1);
  char* comments = (char *) malloc(flen+1);
  
  // Have to do it this way to fix the \r\n's
  int pos = 0;
  int cpos = 0;
  int i;
  bool incomment = false;

  while ((i = fgetc(fp)) != EOF) {
     
    // Remove comment lines
    if (incomment) {
      if (i == (int) '\n') {
        comments[cpos++] = '\n';
        incomment = false;
        while ((i = fgetc(fp)) != EOF &&
               i == (int) '\n');
        if (i == EOF)
           break;
      } else {
        comments[cpos++] = (char)i;
        continue;
      }
    }  else if (pos > 0 && 
                fpcode[pos-1] == '/' &&
                i == (int) '/') {
      incomment = true;
      comments[cpos++] = '/';
      comments[cpos++] = '/';
      fpcode[--pos] = '\0';
      continue;
    }
    
    fpcode[pos++] = (char) i;
  }  
  fpcode[pos] = '\0';
  comments[cpos] = '\0';

  // TIM: get instruction count information before we axe the damn thing... :)
  if( outUsage )
  {
    const char* instructionLine = strstr( comments, "// approximately " );
    assert( instructionLine );

    const char* totalCountText = instructionLine + strlen("// approximately ");
    int totalInstructionCount = atoi( totalCountText );

	const char* nextLine = strstr( instructionLine, "\n" );
    const char* textureCount = strstr( instructionLine, "(" );
    if( (textureCount && !nextLine) || (textureCount && textureCount < nextLine) )
    {
      textureCount += strlen("(");

      const char* arithmeticCount = strstr( textureCount, ", " );
      //assert( arithmeticCount );
      arithmeticCount += strlen(", ");

      outUsage->arithmeticInstructionCount = atoi( arithmeticCount );
      outUsage->textureInstructionCount = atoi( textureCount );
    }
    else
    {
      outUsage->arithmeticInstructionCount = totalInstructionCount;
      outUsage->textureInstructionCount = 0;
    }

    // now look for register usage..
    // we know the pattern for what temps/constants/etc will look like, so:
    int samplerCount = 0;
    int interpolantCount = 0;
    int constantCount = 0;
    int registerCount = 0;
    int outputCount = 0;

    char registerName[128];

    for( int i = 0; i < 16; i++ )
    {
      sprintf( registerName, " s%d", i );
      if( strstr( fpcode, registerName ) )
        samplerCount = i+1;
      sprintf( registerName, " t%d", i );
      if( strstr( fpcode, registerName ) )
        interpolantCount = i+1;
      sprintf( registerName, " c%d", i );
      if( strstr( fpcode, registerName ) )
        constantCount = i+1;
      sprintf( registerName, " r%d", i );
      if( strstr( fpcode, registerName ) )
        registerCount = i+1;
      sprintf( registerName, " oC%d", i );
      if( strstr( fpcode, registerName ) )
        outputCount = i+1;
    }
    outUsage->samplerRegisterCount = samplerCount;
    outUsage->interpolantRegisterCount = interpolantCount;
    outUsage->constantRegisterCount = constantCount;
    outUsage->temporaryRegisterCount = registerCount;
    outUsage->outputRegisterCount = outputCount;
  }
  free(comments);
  
  fclose(fp);

  if (!globals.keepFiles)
     remove(argv[kOutputFileArgument]+3);

  free(argv[kOutputFileArgument]);
  free(argv[kInputFileArgument]);

  if (target == CODEGEN_ARB) {
     std::istringstream ifpcode(fpcode);
     std::ostringstream ofpcode;
     
     convert_ps2arb (ifpcode, ofpcode);
     free(fpcode);
     fpcode = strdup(ofpcode.str().c_str());
  }

  return fpcode;
}
