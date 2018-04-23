#ifndef SHEVALUATE_HPP
#define SHEVALUATE_HPP

#include "ShStatement.hpp"
#include "ShProgramNode.hpp"

namespace SH {

// Evaluate the given statement.
// This calls an appropriate function from ShInstructions.
// Be sure to keep this synchronized with the operations in ShStatement.
SH_DLLEXPORT
void evaluate(ShStatement& stmt);

SH_DLLEXPORT
void evaluate(const ShProgramNodePtr& p);

}

#endif
