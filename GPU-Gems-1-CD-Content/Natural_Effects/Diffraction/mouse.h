#ifndef _MOUSE_H_
#define _MOUSE_H_


#ifndef FALSE
#define FALSE 0
#define TRUE 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

	void GetScreenFrame ( float * S, float * T, float * N );

	void SetBackGnd ( float r, float g, float b );
	void StartDisplay ( void );
	void EndDisplay ( void );
	int InitWindow ( long, long, long, long, float );
	int SaveView ();

	void MouseStart ( int button, int state, int x, int y );
	void MouseHandle ( int x, int y );

#ifdef __cplusplus
}
#endif


#endif /* _MOUSE_H_ */