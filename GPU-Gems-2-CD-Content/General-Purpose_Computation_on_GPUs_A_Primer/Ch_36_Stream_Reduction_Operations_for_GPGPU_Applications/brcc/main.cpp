/*
 * main.c
 *
 *      Minor bit of code to drive the whole program.  Nothing interesting
 *      should happen here.
 */
#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif

#include <fstream>

extern "C" {
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "getopt.h"
}

#include "main.h"
#include "ctool.h"
#include "brtscatter.h"
#include "brtvout.h"
#include "codegen.h"

struct globals_struct globals;


/*
 * usage --
 *
 *      Dumps the legitimate commandline options and exits.
 */

static void
usage (void) {
  fprintf (stderr, "Brook CG Compiler\n");
  fprintf (stderr, "Version: 0.2  Built: %s, %s\n", __DATE__, __TIME__);
  fprintf (stderr,
        "brcc [-hvndktyAN] [-o prefix] [-w workspace] [-p shader ]\n"
        "     [-f compiler] [-a arch] foo.br\n\n"
        "   -h            help (print this message)\n"
        "   -v            verbose (print intermediate generated code)\n"
        "   -n            no codegen (just parse and reemit the input)\n"
        "   -d            debug (print cTool internal state)\n"
        "   -k            keep generated fragment program (in foo.cg)\n"
        "   -t            disable kernel call type checking\n"
        "   -y            emit code for ATI 4-output hardware\n"
        "   -A            enable address virtualization (experimental)\n"
        "   -N            deny support for kernels calling other kernels\n"
        "   -o prefix     prefix prepended to all output files\n"
        "   -w workspace  workspace size (16 - 2048, default 1024)\n"
        "   -p shader     cpu/ps20/ps2a/ps2b/arb/fp30/fp40 (can specify multiple)\n"
        "   -f compiler   favor a particular compiler (cgc / fxc / default)\n"
        "   -a arch       assume a particular GPU (default / x800 / 6800)\n"
        "\n");

  exit(1);
}


/*
 * parse_args --
 *
 *      Parses argv and sets the global options as a result.
 */

static void
parse_args (int argc, char *argv[]) {
  int opt, n;
  char *outputprefix = NULL;

  /*
   * zero initialization from the bss will take care of the rest of the
   * defaults.
   */
  globals.workspace    = 1024;
  globals.compilername = argv[0];
  while ((opt = getopt(argc, argv, "d:hkntyANSla:f:o:p:vw")) != EOF) {
     switch(opt) {
     case 'a':
       if (strcasecmp (optarg, "default") == 0)
         globals.arch = GPU_ARCH_DEFAULT;
       else if (strcasecmp (optarg, "x800") == 0)
         globals.arch = GPU_ARCH_X800;
       else if (strcasecmp (optarg, "6800") == 0)
         globals.arch = GPU_ARCH_6800;
       else
         usage();
       break;
     case 'h':
        usage();
        break;
     case 'k':
        globals.keepFiles = true;
        break;
     case 'n':
        globals.parseOnly = true;
        break;
     case 't':
        globals.noTypeChecks = true;
        break;
    // TIM: totally hacked
     case 'y':
        globals.allowDX9MultiOut = true;
        break;
     case 'A':
        globals.enableGPUAddressTranslation = true;
        break;
     case 'N':
        globals.allowKernelToKernel = false;
        break;
     case 'S':
        globals.enableKernelSplitting = true;
        break;
     case 'o':
	if (outputprefix) usage();
	outputprefix = strdup(optarg);
	break;
     case 'f':
       if (strcasecmp (optarg, "cgc") == 0)
         globals.favorcompiler = COMPILER_CGC;
       else if (strcasecmp (optarg, "fxc") == 0)
         globals.favorcompiler = COMPILER_FXC;
       else if (strcasecmp (optarg, "default") == 0)
         globals.favorcompiler = COMPILER_DEFAULT;
       else
         usage();
       break;
     case 'p':
	if (strcasecmp (optarg, "cpu") == 0)
	  globals.target |= TARGET_CPU;
	else if (strcasecmp (optarg, "ps20") == 0)
	  globals.target |= TARGET_PS20;
	else if (strcasecmp (optarg, "ps2b") == 0)
	  globals.target |= TARGET_PS2B;
	else if (strcasecmp (optarg, "ps2a") == 0)
	  globals.target |= TARGET_PS2A;
	else if (strcasecmp (optarg, "ps30") == 0)
	  globals.target |= TARGET_PS30;
	else if (strcasecmp (optarg, "fp30") == 0)
	  globals.target |= TARGET_FP30;
	else if (strcasecmp (optarg, "fp40") == 0)
	  globals.target |= TARGET_FP40;
	else if (strcasecmp (optarg, "arb") == 0)
	  globals.target |= TARGET_ARB;
	else if (strcasecmp (optarg, "cpumt") == 0)
	  globals.target |= TARGET_MULTITHREADED_CPU;
	else
	  usage();
	break;
     case 'v':
	globals.verbose = 1;
	break;
     case 'l':
        globals.printLineDirectives = true;
        break;
     case 'd':
        Project::gDebug = true;
        break;
     case 'w':
	globals.workspace = strtol(optarg, NULL, 0);
	if (globals.workspace < 16 ||
	    globals.workspace > 2048)
	  usage();
	break;
     default:
	usage();
     }
  }

  // The default build targets
  if (globals.target == 0)
     globals.target = TARGET_PS20 | TARGET_CPU | TARGET_MULTITHREADED_CPU |
                      TARGET_FP30 | TARGET_ARB | TARGET_FP40 | TARGET_PS30 |
                      TARGET_PS2B | TARGET_PS2A;

  argv += optind;
  argc -= optind;
  if (argc < 1) usage();
  globals.sourcename = (char *) argv[0];

  n = strlen(globals.sourcename);
  int suffixLength;
  bool isHeader = false;
  if (n >= 3 && !strcmp (globals.sourcename + n - 3, ".br")) {
    suffixLength = 3;
  }
  else if (n >= 4 && !strcmp(globals.sourcename + n - 4, ".brh")) {
    suffixLength = 4;
    isHeader = true;
  }
  else {
    usage();
  }

  if (!outputprefix) {
    outputprefix = strdup(globals.sourcename);
    outputprefix[n-suffixLength] = (char)  '\0';
  }

  globals.shaderoutputname = strdup(outputprefix);

  globals.coutputname = (char *) malloc (strlen(outputprefix) +
					 suffixLength + 2);
  if( isHeader ) {
    sprintf (globals.coutputname, "%s.hpp",outputprefix);
  } else {
    sprintf (globals.coutputname, "%s.cpp",outputprefix);
  }

  // Initialize the codegen unit.
  CodeGen_Init();

  free(outputprefix);
}


/*
 * ConvertToBrtStreamParams --
 *
 *      Converts stream arguments to non-kernel functions into their actual
 *      BRT types (i.e. makes it possible to pass streams as arguments to
 *      functions).
 */

static void
ConvertToBrtStreamParams(FunctionType *fType)
{
   unsigned int i;

   for (i = 0; i < fType->nArgs; i++) {
      Type ** paramTyp=&fType->args[i]->form;
      while(paramTyp) {
         if ((*paramTyp)->isStream()) {
            Type *newForm;
            
            newForm = new BrtStreamParamType((ArrayType *) *paramTyp);

            /*
             * Types are all on the global type list, so we can't just nuke it.
             delete fType->args[i]->form;
            */
            (*paramTyp) = newForm;
         } else if (fType->isKernel() && (*paramTyp)->isArray()) {
            Type* newForm = new BrtStreamParamType((ArrayType *) *paramTyp);
            (*paramTyp) = newForm;
         }
         paramTyp = (*paramTyp)->getSubType();
      }
   }
   return;
}


/*
 * ConvertToBrtStreams --
 *
 *      Converts stream declaration statement objects into BrtStreams.
 */

static void
ConvertToBrtStreams(Statement *s)
{
   DeclStemnt *declStemnt;

   if (!s->isDeclaration()) { return; }
   declStemnt = (DeclStemnt *) s;

   for (unsigned int i=0; i<declStemnt->decls.size(); i++) {
      Decl *decl = declStemnt->decls[i];
      ArrayType *stream;
      Type *brtType;

      if (!decl->form) continue;

      if (decl->form->isFunction()) {
         assert(decl->form->type == TT_Function);
         ConvertToBrtStreamParams((FunctionType *)decl->form);
      }

      if (!decl->isStream()) continue;
      stream = (ArrayType *) decl->form;

      if (decl->initializer == NULL) {
         assert((stream->getQualifiers() & TQ_Iter) == 0);
         brtType = new BrtInitializedStreamType (stream);
         // TIM: we don't want any initializer for stream types...
         // they are initialized by their constructor...
         //
         // decl->initializer = new BrtStreamInitializer(brtStream,
         //                                              declStemnt->location);
      } else {
         assert(decl->initializer->etype == ET_FunctionCall);
         assert((stream->getQualifiers() & TQ_Iter) != 0);
         brtType = new BrtIterType(stream, (FunctionCall *) decl->initializer);
         delete decl->initializer;
         decl->initializer = NULL;
      }
      assert (decl->initializer == NULL);
      decl->form = brtType;
   }
}


/*
 * ConvertToBrtFunctions --
 *
 *      This is the callback for the portion of the transformation phase
 *      when we iterate over all the function definitions and convert Brook
 *      specific ones to our types.
 */

static FunctionDef *
ConvertToBrtFunctions(FunctionDef *fDef)
{
   /*
    * The 'isReduce()' check _must_ come before the 'isKernel()' check
    * because, for better or for worse, reduction kernels are also
    * considered kernels.
    */

   if (fDef->decl->isReduce()) {
      return new BRTReduceKernelDef(*fDef);
   } else if (fDef->decl->isKernel()) {
      return new BRTMapKernelDef(*fDef);
   } else {
      assert(fDef->decl->form->type == TT_Function);
      ConvertToBrtStreamParams((FunctionType *) fDef->decl->form);
      return NULL;
   }

}

static void
ConvertToBrtDecls(Statement *stmt)
{
   DeclStemnt* declStemnt;

   if (!stmt->isDeclaration()) { return; }
   declStemnt = (DeclStemnt *) stmt;

   for (unsigned int i=0; i<declStemnt->decls.size(); i++) {
      Decl *decl = declStemnt->decls[i];

      if (!decl->form) continue;

      if (decl->form->isKernel()) {

         Type *brtType = new BrtKernelType((FunctionType* ) decl->form);
         decl->form = brtType;
      }
   }
}

/*
 * TypeCheckFunctionCallsExprFinder --
 *
 *     This is the callback that actually checks if the type of an expr
 *     is a function call ... and if so then it calls checkKernelCall() member
 *     on that function call
 */

static Expression * TypeCheckFunctionCallsExprFinder (Expression * e) {
   bool ret=true;
   if (e->etype==ET_FunctionCall) {
      FunctionCall * fc = static_cast<FunctionCall*>(e);
      ret=fc->checkKernelCall()&&ret;//print out type errors
   }
   /* 
    * Only assert after all functions are checked
    */
   assert(ret);
   return e;
}

static void TypeCheckFunctionCallsStatementFinder(Statement * ste) {
   ste->findExpr(&TypeCheckFunctionCallsExprFinder);
}

static void TypeCheckFunctionCalls(TransUnit * tu) {
   tu->findStemnt (&TypeCheckFunctionCallsStatementFinder);
}

/*
 * main --
 *
 *      Drive everything.  Parse the arguments, then compile the requested
 *      file.
 */

int
main(int argc, char *argv[])
{
   Project *proj;
   TransUnit *tu;

   parse_args(argc, argv);
   if (globals.verbose) {
      std::cerr << "***Compiling " << globals.sourcename << "\n";
   }

   proj = new Project();
   tu = proj->parse(globals.sourcename, false, NULL, false, NULL, NULL, NULL);
   if (tu) {
      std::ofstream out;
      TypeCheckFunctionCalls(tu);
      if (!globals.parseOnly) {
         /*
          * If I didn't mind violating some abstractions, I'd roll my own loop
          * here instead of using the Translation Unit methods.
          */
         Brook2Cpp_IdentifyIndexOf(tu);
         transform_vout(tu);
         tu->findStemnt(ConvertToBrtStreams);
         tu->findFunctionDef(ConvertToBrtFunctions);
         tu->findStemnt(ConvertToBrtDecls);
      }

     out.open(globals.coutputname);
     if (out.fail()) {
        std::cerr << "***Unable to open " << globals.coutputname << "\n";
        exit(1);
     }

     out << *tu << std::endl;
     out.close();
   } else {
      std::cerr << "***Unable to parse " << globals.sourcename << std::endl;
      exit(1);
   }

   if (globals.verbose) {
      std::cerr << "***Successfully compiled " << globals.sourcename << "\n";
   }
   delete proj;
   exit(0);
   return 0;    /* Appease CL */
}
