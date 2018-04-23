/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)
/////////////////////////////////////////////////////////////////////////////


#include <cstring>
#include <cctype>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "argoption.h"



// Constructor.  Does not do any parsing or error checking.
// Make sure to call initialize() right after construction.
ArgOption::ArgOption (const char *str) 
    : format(str)

{
    flag = NULL;
    code = NULL;
    type = None;
    count = 0;
    param = NULL;
    callback = NULL;
    repetitions = 0;
    argc = 0;
    argv = NULL;
}



// Parses the format string ("-option %s %d %f") to extract the
// flag ("-option") and create a code string ("sdf").  After the
// code string is created, the param list of void* pointers is
// allocated to hold the argument variable pointers.
int
ArgOption::initialize()
{
    size_t n;
    const char *s;
    char *c;

    if (format[0] == '\0' ||
        (format[0] == '%' && format[1] == '*' && format[2] == '\0')) {
        type = Sublist;
        count = 1;                      // sublist callback function pointer
        code = strdup("*");
        flag = strdup("");
    } else {
        // extract the flag name
        s = format;
        assert(*s == '-');
        assert(isalpha(s[1]));
    
        s++;

        while (isalnum(*s) || *s == '_' || *s == '-') s++;

        n = s - format + 1;
        flag = (char *)malloc (n * sizeof(char));
        assert(flag != NULL);
        strncpy(flag, format, n-1);
        flag[n-1] = '\0';

        // Check to see if this is a simple flag option
        if (n == strlen (format) + 1) {
            type = Flag;
            count = 1;
            code = strdup("!");
        } else {
            // Parse the scanf-like parameters

            type = Regular;
    
            n = (strlen (format) - n) / 2;       // conservative estimate
            code = (char *)malloc(n * sizeof (char));
            assert(code != NULL);
            c = code;
    
            while (*s != '\0') {
                if (*s == '%') {
                    s++;
                    assert(*s != '\0');
            
                    count++;                    // adding another parameter
            
                    switch (*s) {
                    case 'd':                   // 32bit int
                    case 'g':                   // float
                    case 'f':                   // float
                    case 'F':                   // double
                    case 's':                   // char * 
                    case 'S':                   // allocated char *
                    case 'L':                   // allocated char * list
                        assert (type == Regular);
                        *c++ = *s;
                        break;

                    case '*':
                        assert(count == 1);
                        type = Sublist;
                        break;
                        
                    default:
                        std::cerr << "Programmer error:  Unknown option ";
                        std::cerr << "type string \"" << *s << "\"" << "\n";
                        abort();
                    }
                }
        
                s++;
            }
        }
    }
    
    // Allocate space for the parameter pointers and initialize to NULL
    assert(count > 0);
    param = (void **)calloc (count, sizeof(void *));
    assert(param != NULL);

    return 0;
}



// Stores the pointer to an argument in the param list and
// initializes flag options to FALSE.
void
ArgOption::add_parameter (int i, void *p)
{
    assert (i >= 0 && i < count);
    param[i] = p;
}



// Given a string from argv, set the associated option parameter
// at index i using the format conversion code in the code string.
void
ArgOption::set_parameter (int i, char *argv)
{
    assert(i < count);
    
    switch (code[i]) {
    case 'd':
        *(int *)param[i] = atoi(argv);
        break;

    case 'f':
    case 'g':
        *(float *)param[i] = (float)atof(argv);
        break;

    case 'F':
        *(double *)param[i] = atof(argv);
        break;

    case 's':
        strcpy((char *)param[i], argv);
        break;

    case 'S':
        *(char **)param[i] = strdup(argv);
        break;

    case 'L':
        if (*(char **)param[i] == NULL) {
            *(char **)param[i] = strdup(argv);
        } else {
            char *tmp = *(char **)param[i];
            size_t len = strlen (tmp) + strlen (argv) + 2;  // + (space & \0)
            *(char **)param[i] = (char *)malloc (len);
            sprintf (*(char **)param[i], "%s %s", tmp, argv);
        }
        break;

    case '!':
        *(bool *)param[i] = true;
        break;
        
    case '*':
    default:
        abort();
    }
}



// Call the sublist callback if any arguments have been parsed
int
ArgOption::invoke_callback() const
{
    assert (count == 1);

    if (argc == 0) {
        return 0;
    }
    
    if (((int (*)(int, char **))param[0]) (argc, argv) < 0) {
        return -1;
    }

    return 0;
}



// Add an argument to this sublist option
void
ArgOption::add_argument (char *argv)
{
    argc++;
    this->argv = (char **)realloc(this->argv, argc * sizeof(char *));
    assert (this->argv != NULL);

    this->argv[argc - 1] = strdup(argv);
}



ArgOption::~ArgOption()
{
    assert (flag != NULL);
    free (flag);

    if (code != NULL) {
        free (code);
    }

    if (param != NULL) {
        assert (count > 0);
        free (param);
    }

    if (argv != NULL) {
        assert (argc > 0);
        for (int i = 0; i < argc; i++)
            free (argv[i]);
        free (argv);
    }

    // free globbed argument strings
}

