typedef int (* render_info) (int line_finished,int total_lines);
const char *get_file_name(const char *fullfile);
const char *get_path_name(const char *fullfile);

typedef struct
{
	int id,line;
	unsigned long threadid;
	pPicture *pic;
	pRender *render;
	render_info info;
} pThreadInfo;

typedef struct 
{
	pVector ro;
	pVector rd;
	pVector color;
	pVector ip;
	pVector n;
	pVector uv;
	float x,y;
	float dist;
	int depth;
	float shadow;
	int thread;
} pRayInfo;

class pRender
{
	public:
		HWND hwnd;
		HDC hdc;
		HGLRC hrc;

		CGcontext cgcontext;

		pArray<pRenderProfile *> profile;
		pRenderProfile *cur_profile;
		int sel_profile;

		int sizex,sizey;
		int rendersizex,rendersizey;
		float aspect;
		float nearplane;
		float farplane;
		int colorbits,depthbits,stencilbits;
		
		pArray<DEVMODE> videomode;
		int selvideomode;
		
		HMENU winmenu;
		int fullscreen;

		int fontspic;
		int fontswidth[256];
		int fontspicsize;
		int fontspiccharsize;
		int cursorx,cursory;

		pVector ambient;
		int wireframe;
		int texmap;
		int texfilter;
		int texmipmap;
		int fog;
		int debug;
		int texdrop;
		int loadtex;
		int loadtexresize;
		int loadtwoside;
		
		int bgclear;
		int bgmode;
		int bgpic;
		pVector bgcolor;

		int shadowmode;
		int shadowsamples;
		int shadowfull;
		float shadowfactor;

		int maxtex2dsize;
		int maxtex3dsize;
		int maxlights;
		int maxraydepth;
		int maxoctreedepth;
		int maxtextureunits;

		int rendermode;
		int lightsamples;
		float lightfactor;
		int viewshadowmode;

		int savecopytex;
		int cameramode;
		int curdraw;
		
		pArray<unsigned> picid;
		pArray<pPicture *> pic;
		int selpic;

		pMesh *mesh;
		pOcTree *octree;

		pCamera camera;
		pFrustum view;
		
		pCameraCtrl *camctrl[2];
		int selcamctrl;
		int camcollide;

		int cam_viewport[3][4];
		double cam_proj_mat[3][16];
		double cam_model_mat[3][16];
		
		pPicture render_pic;
		pString output_image;
		pString scene_file;

		pThreadInfo threadinfo[P_MAX_THREADS];
		int numthreads,threadpriority,curline,curaaline,nextline,nextaaline;

		int antialias,antialias_factor,antialias_depth;

		pVector hitnormal,hitip;
		float hitdist;

		float curtimef,curdtf;
		unsigned starttime,curtime,curdt;
		pString app_path;

	pRender(const char *path="");
	virtual ~pRender();

	void pre_create(HWND hWnd);
	int create(int menuflag=1);
	void resize(int sx,int sy);
	void destroy();
	
	void load_ini();
	int load_mesh(const char *meshfile);
	void free_mesh();

	void init();
	void begin_draw();
	void draw_scene();
	void end_draw();
	void draw_background();
	void draw_shadows(pLight *l,pBoundBox& bb);
	void update();

	int load_tex(const char *texfile,int flipy=0,int bump=0);
	int load_tex_3d(const char *texfile);
	int load_tex_cubemap(const char *texfile,const char *ext,int flipy=0);
	void build_mipmaps(pPicture *p,unsigned type2);
	void set_tex_params(pPicture *p,unsigned type1,unsigned type2);
	void sel_tex(int tex);
	void update_texflags();
	void reset_tex_state();
	void copy_textures(const char *dest);
	int create_texture(int sx,int sy,int bytespixel,unsigned color);
	void set_fullscreen(int fsflag,int menuflag=1);

	void set_camera(const pCamera& cam,int sx,int sy,int getmat);
	int key_pressed(int key);
	void check_input();

	void get_ray(float x,float y,pVector& ray,int mat);
	void get_background_pixel(pRayInfo& ri) const;
	
	int ray_trace(pRayInfo& ri) const;
	void ray_trace(pPicture& p,render_info info);
	void ray_trace(pThreadInfo *ti);
	void ray_trace_antialias(pThreadInfo *ti,int factor,int depth);
	
	int compute_shadow_factor(const pVector& p,const pLight *l,pVector& s,int thread) const;
	float compute_bgshadow(const pVector& ip,const pVector& n,int thread) const;
	
	void compute_color(float dist,const pVector& t,int lm,const pMaterial *m,const pVector& ip,const pVector& n,const pVector& uv,pVector& color,int thread) const;
	void light_direct(float dist,const pVector& t,const pVector& v,const pMaterial *m,const pVector& ip,const pVector& n,const pVector& uv,pVector& color,int thread) const;
	void light_global(const pVector& t,const pVector& v,const pMaterial *m,const pVector& ip,const pVector& n,const pVector& uv,pVector& color,int thread) const;

	void load_fonts();
	void set_draw2d();
	void draw_text(int x,int y,const char *text,int size,int n=-1);
	void draw_text_center(int x,int y,const char *text,int size);
	int get_text_size(const char *text, int size);

	int box_collision(const pBoundBox& bbox,const pVector& pos,pVector& destpos,pVector& destvel,float bump,float friction,int maxcol=3);
	
	int build_normalizationcubemap(int size);
	int build_illuminationmap(int size);
	int build_onepixelmap(unsigned char r,unsigned char g,unsigned char b);

	void profile_build();
	void profile_select(int p);
};
