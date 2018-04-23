//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include "DataTypes.h"

void convObject2VecPoint(VecPoint& points,const Object& obj) {
	points.clear();
	for(unsigned i = 0; i < obj.size(); i++) {
		const VecPoint& p = obj[i];
		for(unsigned j = 0; j < p.size(); j++) {
			points.push_back(p[j]);
		}
	}
}


