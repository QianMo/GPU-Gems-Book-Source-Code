#include <sstream>

#include "ps2arb.h"
#include "ps2arb_intermediate.h"

std::istream *ps2arb_ps20code;

ps2arb::IntermediateLanguage *ps2arb::iLanguage;

int convert_ps2arb (std::istream &ps20code, 
		    std::ostream &arbcode){

#if YYDEBUG
  ps2arb_yydebug = 1;
#endif

  ps2arb::iLanguage = new ps2arb::IntermediateLanguage();

  ps2arb_ps20code = &ps20code;

  ps2arb_parse();

  ps2arb::iLanguage->print_arbfp(arbcode);

  delete ps2arb::iLanguage;

  return 1;
}
