//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#include "MathTools.h"
namespace Math {

void createSingularRandomNumbers(DynamicArray<unsigned>& output, const unsigned count, const unsigned max) {
	const double fact = (1.0/RAND_MAX)*(max);
	if(count*10 < max) {
		//for small part of the range
		output.clear();
		for(unsigned i = 0; i < count; i++) {
			const unsigned val = Math::clamp<unsigned>(rand()*fact,0,max);
			bool insert = true;
			//make unique
			for(unsigned j = 0; j < output.size(); j++) {
				if(val == output[j]) {
					insert = false;
					i--;
					break;
				}
			}
			if(insert) {
				output.append(val);
			}
		}
	}
	else {
		//big part of range
		output.resize(max+1);
		for(unsigned i = 0; i < output.size(); i++) {
			output[i] = i;
		}
		for(unsigned i = 0; i < output.size()/2; i++) {
			const unsigned a = Math::clamp<unsigned>(rand()*fact,0,max);
			const unsigned b = Math::clamp<unsigned>(rand()*fact,0,max);
			std::swap(output[a],output[b]);
		}
		output.resize(count);
	}
}

//namespace
}