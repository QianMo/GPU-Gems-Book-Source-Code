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

#include <cstdarg>
#include <cctype>
#include <cassert>
#include <iterator>

#include "argparse.h"
#include "argoption.h"



ArgParse::ArgParse (int argc, char **argv)
    : argc (argc), argv (argv), global(NULL)
{
    message[0] = '\0';
}



// Called after all command line parsing is completed, this function
// will invoke all callbacks for sublist arguments.  
inline int
ArgParse::invoke_all_sublist_callbacks()
{
    for (std::vector<ArgOption *>::const_iterator i = option.begin();
         i != option.end(); i++) {
        
        if (!(*i)->is_sublist()) {
            continue;
        }
        
        if ((*i)->invoke_callback() < 0) {
            return -1;
        }
    }

    return 0;
}



// Top level command line parsing function called after all options
// have been parsed and created from the format strings.  This function
// parses the command line (argc,argv) stored internally in the constructor.
// Each command line argument is parsed and checked to see if it matches an
// existing option.  If there is no match, and error is reported and the
// function returns early.  If there is a match, all the arguments for
// that option are parsed and the associated variables are set.
inline int
ArgParse::parse_command_line()
{
    for (int i = 1; i < argc; i++) {

        if (argv[i][0] == '-' && 
              (isalpha (argv[i][1]) || argv[i][1] == '-')) {         // flag

            ArgOption *option = find_option (argv[i]);
            if (option == NULL) {
                report_error ("Invalid option \"%s\"", argv[i]);
                return -1;
            }

            option->found_on_command_line();
            
            if (option->is_flag()) {
                option->set_parameter(0, NULL);
                if (global != NULL && global->name()[0] != '\0')
                    global = NULL;                              // disable
            } else if (option->is_sublist()) {
                global = option;                                // reset global
            } else {
                assert (option->is_regular());
                if (global != NULL && global->name()[0] != '\0')
                    global = NULL;                              // disable
                
                for (int j = 0; j < option->parameter_count(); j++) {

                    if (j+i+1 >= argc) {
                        report_error ("Missing parameter %d from option "
                            "\"%s\"", j+1, option->name());
                        return -1;
                    }

                    option->set_parameter (j, argv[i+j+1]);
                }

                i += option->parameter_count();
            }
            
        } else {
            // not an option nor an option parameter, glob onto global list
            if (global == NULL) {
                report_error ("Argument \"%s\" does not have an associated "
                    "option", argv[i]);
                return -1;
            }

            global->add_argument (argv[i]);
        }
    }

    return 0;
}



// Primary entry point.  This function accepts a set of format strings
// and variable pointers.  Each string contains an option name and a
// scanf-like format string to enumerate the arguments of that option
// (eg. "-option %d %f %s").  The format string is followed by a list
// of pointers to the argument variables, just like scanf.  All format
// strings and arguments are parsed to create a list of ArgOption objects.
// After all ArgOptions are created, the command line is parsed and
// the sublist option callbacks are invoked.
int
ArgParse::parse (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);

    for (const char *cur = format; cur != NULL; cur = va_arg (ap, char *)) {

        if (find_option (cur)) {
            report_error ("Option \"%s\" is multiply defined");
            return -1;
        }
        
        // Build a new option and then parse the values
        ArgOption *option = new ArgOption (cur);
        if (option->initialize() < 0) {
            return -1;
        }

        if (cur[0] == '\0' ||
            (cur[0] == '%' && cur[1] == '*' && cur[2] == '\0')) {
            // set default global option
            global = option;
        }
        
        // Grab any parameters and store them with this option
        for (int i = 0; i < option->parameter_count(); i++) {

            void *p = va_arg (ap, void *);
            if (p == NULL) {
                report_error ("Missing argument parameter for \"%s\"",
                    option->name());
                return -1;
            }
            
            option->add_parameter (i, p);
        }

        this->option.push_back(option);
    }

    va_end (ap);

    if (parse_command_line() < 0) {
        return -1;
    }

    if (invoke_all_sublist_callbacks() < 0) {
        return -1;
    }
    
    return 0;
}



// Find an option by name in the option vector
ArgOption *
ArgParse::find_option(const char *name)
{
    for (std::vector<ArgOption *>::const_iterator i = option.begin();
         i != option.end(); i++) {

        if (strlen(name) == strlen((*i)->name()) &&
            strcmp(name, (*i)->name()) == 0) {
            return *i;
        }
    }

    return NULL;
}



int
ArgParse::found(char *option_name)
{
    ArgOption *option = find_option(option_name);
    if (option == NULL) return 0;
    return option->parsed_count();
}


void
ArgParse::report_error(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vsprintf(message, format, ap);
    va_end(ap);
}



ArgParse::~ArgParse()
{
    for (std::vector<ArgOption *>::const_iterator i = option.begin();
         i != option.end(); i++) {
        delete *i;
    }
}
