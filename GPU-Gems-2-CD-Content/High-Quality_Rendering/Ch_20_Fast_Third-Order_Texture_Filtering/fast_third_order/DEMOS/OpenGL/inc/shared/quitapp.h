#ifndef __QUITAPP__
#define __QUITAPP__

#if defined(WIN32)
#   define quitapp(code) {\
    char buf;\
    std::cerr << std::endl << "Press <enter> to quit." << std::endl;\
    std::cin.getline(&buf, 1);\
    exit(code); }
#else
#   define quitapp(code) {exit(code);}
#endif

#endif /* __QUITAPP__ */
