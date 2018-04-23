/*
 * subprocess.cpp --
 *
 *      Helper functions to run a child process and interact with it.
 */

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>

#ifdef _WIN32
#include <process.h>
#include <io.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#endif
}

#include "main.h"

#define READ_HANDLE 0
#define WRITE_HANDLE 1


/*
 * Subprocess_Run --
 *
 *      Runs the requested program with the provided input and returns the
 *      output.
 *
 *      Note: The output is allocated on the heap and should be free()d by
 *      the caller.
 */

char *
Subprocess_Run(char *argv[], char *input)
{
   bool debug =globals.verbose;
  int hStdOutPipe[2], hStdInPipe[2];
  int hStdOut, hStdIn;
  int pid, ret;

  char *output;
  int output_alloc, output_pos;

  // Create the pipe
#ifdef _WIN32
  if(_pipe(hStdOutPipe, 512, O_TEXT | O_NOINHERIT) == -1) {
    fprintf (stderr, "Unable to create pipe\n");
    return NULL;
  }
  if(_pipe(hStdInPipe, 512, O_TEXT | O_NOINHERIT) == -1) {
    fprintf (stderr, "Unable to create pipe\n");
    return NULL;
  }

  // Save the current stdout/stdin handles
  hStdOut = _dup(_fileno(stdout));
  hStdIn  = _dup(_fileno(stdin));

  // Duplicate the pipes
  if(_dup2(hStdOutPipe[WRITE_HANDLE], _fileno(stdout)) != 0) {
    fprintf (stderr, "Unable to redirect stdout\n");
    return NULL;
  }
  if(_dup2(hStdInPipe[READ_HANDLE], _fileno(stdin)) != 0) {
    fprintf (stderr, "Unable to redirect stdin\n");
    return NULL;
  }

#else
  if(pipe(hStdOutPipe) == -1) {
       fprintf (stderr, "Unable to create pipe\n");
       return NULL;
  }
  if(pipe(hStdInPipe) == -1) {
    fprintf (stderr, "Unable to create pipe\n");
    return NULL;
  }
  if((hStdOut = dup(fileno(stdout)))  == -1 ){
       fprintf (stderr, "dup stdout\n");
       return NULL;
  }
  if((hStdIn  = dup(fileno(stdin))) == -1 ){
       fprintf (stderr, "dup stdin\n");
       return NULL;
  };
  if( dup2(hStdOutPipe[WRITE_HANDLE], fileno(stdout)) == -1 ){
       fprintf (stderr, "Unable to redirect stdout\n");
       return NULL;
  }
  if( dup2(hStdInPipe[READ_HANDLE], fileno(stdin)) == -1) {
       fprintf (stderr, "Unable to redirect stdout\n");
       return NULL;
  }
#endif

  // Close our side of the pipes.
  // Cgc will be writing/reading these pipes
  if(close(hStdOutPipe[WRITE_HANDLE]) !=0){
       fprintf(stderr, "Write close error\n");
  }
  if(close(hStdInPipe[READ_HANDLE]) !=0){
       fprintf(stderr, "Write close error\n");
  }

#if _WIN32
  if ((pid = _spawnvp(P_NOWAIT, argv[0], argv)) == -1) {
    if (debug) fprintf( stderr, "Unable to start %s\n", argv[0]);
    return NULL;
  }

#else

  if ((pid = fork()) == 0) {
       close (hStdInPipe[WRITE_HANDLE]);

       if (execvp(argv[0], argv) == -1) {
            if (debug) fprintf( stderr, "Unable to start %s\n",argv[0] );
	    exit(-1);
       }
       /* Unreached... */
  } else {
       if (debug) fprintf(stderr, "Child has pid %d\n", pid);
  }
#endif


#if _WIN32
  // Restore the pipes for us.
  if(_dup2(hStdOut, _fileno(stdout)) != 0) {
    fprintf (stderr, "Unable to restore stdout\n");
    return NULL;
  }
  if(_dup2(hStdIn, _fileno(stdin)) != 0) {
    fprintf (stderr, "Unable to restore stdin\n");
    return NULL;
  }
#else
  if(dup2(hStdOut, fileno(stdout)) == -1) {
       fprintf (stderr, "Unable to restore stdout\n");
       return NULL;
  }
  if( dup2(hStdIn, fileno(stdin)) == -1) {
       fprintf (stderr, "Unable to restore stdin\n");
       return NULL;
  }
#endif


  // Close the remaining pipes
  if(close(hStdOut) !=0){
       fprintf(stderr, "Write close error\n");
  }
  if(close(hStdIn) !=0){
       fprintf(stderr, "Write close error\n");
  }

  /* Feed the cg code to the compiler */
  if (debug) fprintf(stderr, "Sending the input to %s\n", argv[0]);

  if (input) {
#if _WIN32
     _write (hStdInPipe[WRITE_HANDLE], input, strlen(input));
#else
     {
        char eof_holder = EOF;
        int retval;
        void (*oldPipe)(int);

        oldPipe = signal(SIGPIPE, SIG_IGN);
        /* fprintf(stderr, "Writing\n[35;1m%s[0m\n", input); */

        retval = write (hStdInPipe[WRITE_HANDLE], input, strlen(input));
        if (retval == -1 && errno == EPIPE) {
           if (debug) {
              fprintf(stderr, "Pipe vanished writing input to %s!\n", argv[0]);
           }
           oldPipe = signal(SIGPIPE, oldPipe);
           return NULL;
        } else if (retval != (int) strlen(input)) {
           perror("Write problem");
        }

        /*
         * We're sloppy with EPIPE on the EOF because we don't really care.
         * We already got the input off and if the child died, we'll notice
         * either when we call waitpid() or read the result so we just want
         * to not die with SIGPIPE here.  --Jeremy.
         */
        write(hStdInPipe[WRITE_HANDLE], &eof_holder, 1);
        oldPipe = signal(SIGPIPE, oldPipe);
     }
#endif
  }
  
  if (debug) fprintf(stderr, "Closing pipe to %s\n", argv[0]);
  if(close(hStdInPipe[WRITE_HANDLE]) !=0){
       fprintf(stderr, "Write close error\n");
  }

  output = (char *) malloc (1024);
  output_alloc = 1024;
  output_pos   = 0;

  while (1) {
    char buf[1024];

    if (debug) fprintf(stderr, "Reading pipe from %s...\n", argv[0]);
#ifdef _WIN32
    ret = _read(hStdOutPipe[READ_HANDLE], buf, 1023);
#else
    ret = read(hStdOutPipe[READ_HANDLE], buf, 1023);
#endif

    if (ret == 0) {
      if (debug) fprintf(stderr, "Got everything from %s.\n", argv[0]);
      break;
    } else if (ret == -1) {
      fprintf (stderr, "Error reading output compiler pipe.\n");
      return NULL;
    }
    buf[ret] = '\0';

    if (debug) fprintf(stderr, "Read %d bytes\n", ret);
    /*fprintf(stderr, "Read %d bytes '%s'\n", ret, buf);*/
    while (output_alloc - output_pos  <= ret) {
      output = (char *) realloc (output, output_alloc*2);
      output_alloc *= 2;
    }
    memcpy (output + output_pos, buf, ret);
    output_pos += ret;
  }
  output[output_pos] = '\0';

#if _WIN32
  _cwait(&ret, pid, WAIT_CHILD);
#else
  waitpid(pid, &ret, 0);
  ret = WIFEXITED(ret) ? WEXITSTATUS(ret) : -1;
#endif

  close (hStdOutPipe[READ_HANDLE]);

  if (ret != 0) {
    if (debug) fprintf (stderr, "%s exited with an error (%#x):\n", argv[0], ret);
    fwrite (output, strlen(output), 1, stderr);

    return NULL;
  }

  return output;
}
