//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef StateH
#define StateH
//---------------------------------------------------------------------------
//#include <Base/Singelton.h>
//---------------------------------------------------------------------------
struct State /*: public Singelton*/ {
	static State EMPTY;
	virtual void begin() { }
	virtual void end() { }
};

class StateManager {
protected:
	State* currentState;
public:
	StateManager(): currentState(&State::EMPTY) { }
	void setState(State& newState) {
		if(&newState != currentState) {
			if(&State::EMPTY != currentState)
				currentState->end();

			currentState = &newState;

			if(&State::EMPTY != currentState)
				currentState->begin();
		}
	}
	State* getState() const { return currentState; }
};

#endif
