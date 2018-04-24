/**
    file curvereader.h
    Copyright (c) NVIDIA Corporation. All rights reserved.
 **/
#ifndef __CURVEREADER__
#define __CURVEREADER__

#include "curveEngine.h"
#pragma warning(disable:4201 ) //C4201: nonstandard extension used : nameless struct/union

using namespace std;

class CurvePool;
class CurveReader;

enum EtTangentType 
{
	kTangentFixed,
	kTangentLinear,
	kTangentFlat,
	kTangentStep,
	kTangentSlow,
	kTangentFast,
	kTangentSmooth,
	kTangentClamped
};
typedef enum EtTangentType EtTangentType;

struct ReadKey
{
	float			time;
	float			value;
	EtTangentType	inTangentType;
	EtTangentType	outTangentType;
	float			inAngle;
	float			inWeight;
	float			outAngle;
	float			outWeight;
};
/*----------------------------------------------------------------------------------*/ /**

We are grouping curves so that we may have a vector from 1 to 4 dimensions.
This is interesting for connections with another plug which would be a vector

**/ //----------------------------------------------------------------------------------
class CurveVector
{
protected:
	union 
	{
		CurveReader *m_cvs[4]; ///< the 1 to 4 curves
		struct
		{
			CurveReader *m_cvx;
			CurveReader *m_cvy;
			CurveReader *m_cvz;
			CurveReader *m_cvw;
		};
	};
	int m_dim; ///< dimension of the 'vector'
	char * m_name; ///< the name. Pointed by CurvePool::CurveMapType
	float m_time, m_cvvals[4];
public:
	CurveVector(const char * name, int dim);
	~CurveVector();
	/// \name almost the same methods of Curve class but for a 'vector'
	//@{
	bool find(float time, int *indexx, int *indexy, int *indexz, int *indexw);
	float evaluate(float time, float *v1, float *v2 = NULL, float *v3 = NULL, float *v4 = NULL);
	float evaluateInfinities(float time, bool evalPre, float *v1 = NULL, float *v2 = NULL, float *v3 = NULL, float *v4 = NULL);
    void  startKeySetup(bool inputIsTime, bool outputIsAngular, bool isWeighted,
			  EtInfinityType preinftype, EtInfinityType postinftype);
	void addKey(float frame, float x, float y, float z, float w, 
				EtTangentType inTangentType=kTangentSmooth, EtTangentType outTangentType=kTangentSmooth, 
				float inAngle=0, float inWeight=0, float outAngle=0, float outWeight=0);
	void addKeyHere(EtTangentType inTangentType=kTangentSmooth, EtTangentType outTangentType=kTangentSmooth, 
				float inAngle=0, float inWeight=0, float outAngle=0, float outWeight=0);
	//@}
	virtual CurveReader *getCurve(int n);
	virtual void clear(int n=-1);

	friend CurvePool;
};
/*----------------------------------------------------------------------------------*/ /**

- contains a pool of allocated curves
.

**/ //----------------------------------------------------------------------------------
class CurvePool
{
public:
	~CurvePool();
	virtual void clear();
	CurveVector *newCV(const char *  name, int dim);
	CurveVector *newCVFromFile(const char *  fname, char *  overloadname = NULL);
	int getNumCV() {return (int)m_curves.size(); };
	CurveVector *getCV(const char *  name);
    CurveVector *getCVByIndex(int i) {if( i < (int)m_curves.size()) return m_curvesVec[i]; };
private:
	struct ltstr
	{
	  bool operator()(const char * s1, const char * s2) const
	  {
		return strcmp(s1, s2) < 0;
	  }
	};
	typedef std::map<const char *, CurveVector*, ltstr> CurveMapType;
	CurveMapType m_curves;
    std::vector<CurveVector*> m_curvesVec;
};


class CurveReader : public Curve
{
public:
	CurveReader();
	~CurveReader() {};
protected:
	//EtTangentType AsTangentType (const char *str);
	bool assembleAnimCurve(vector<ReadKey> &keys, bool isWeighted, bool useOldSmooth);

	vector<ReadKey> m_keys; ///< keys which define the curve
	float			m_unitConversion;
	float			m_frameRate;
	
public:
	virtual void setName(const char *  name) {}
	virtual void startKeySetup(bool inputIsTime=true, bool outputIsAngular=false, bool isWeighted=false,
						EtInfinityType preinftype=kInfinityConstant, EtInfinityType postinftype=kInfinityConstant);
	virtual void getKeySetup(bool &inputIsTime, bool &outputIsAngular, bool &isWeighted,
						EtInfinityType &preinftype, EtInfinityType &postinftype);
	virtual void addKey(float frame, float val, 
				EtTangentType inTangentType=kTangentSmooth, EtTangentType outTangentType=kTangentSmooth, 
				float inAngle=0, float inWeight=0, float outAngle=0, float outWeight=0);
	virtual void endKeySetup();

	virtual int getNumKeys() { return (int)m_keys.size(); };
	bool getKey(int n, ReadKey &k);
	virtual void clear();
	virtual bool delkey(int nkey);

	friend CurveVector;
};
#endif
