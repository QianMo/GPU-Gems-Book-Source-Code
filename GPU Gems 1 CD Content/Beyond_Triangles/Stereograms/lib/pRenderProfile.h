class pRenderProfile
{
	public:
		pString name;

	pRenderProfile()
	{ }
	virtual ~pRenderProfile()
	{ }
		
	virtual int test()=0;
	virtual void reset()=0;
	virtual void load(pRender* r)=0;
	virtual void draw(pRender *r,const pArray<pFace *>& f)=0;
	virtual void draw(pRender *r,const pMaterial& mat,const pBoundBox& bbox,void draw_geometry())=0;
	virtual void update_mat(pRender *r,const pMaterial *mat,int mode)=0;
	virtual void print(pString& str)=0;
};

class pRenderProfileAmbient : public pRenderProfile
{
	public:

	pRenderProfileAmbient(const char *profile_name)
	{ 
		name=profile_name;
	}
	virtual ~pRenderProfileAmbient()
	{ reset(); }
		
	int test();
	void reset();
	void load(pRender* r);
	void draw(pRender *r,const pArray<pFace *>& f);
	void draw(pRender *r,const pMaterial& mat,const pBoundBox& bbox,void draw_geometry());
	void update_mat(pRender *r,const pMaterial *mat,int mode);
	void print(pString& str);
};

class pRenderProfileVertex : public pRenderProfile
{
	public:
		pString vert_extension;
		CGprofile vert_profile;

		CGprogram cgprog_vert_ambient;
		CGparameter cgparam_vert_ambient_modelviewproj;
		CGparameter cgparam_vert_ambient_camerapos;
		CGparameter cgparam_vert_ambient_envmapflag;

		CGprogram cgprog_vert_shadow;
		CGparameter cgparam_vert_shadow_modelviewproj;
		CGparameter cgparam_vert_shadow_lightpos;

		CGprogram cgprog_vert_light;
		CGparameter cgparam_vert_light_modelviewproj;
		CGparameter cgparam_vert_light_camerapos;
		CGparameter cgparam_vert_light_specular;
		CGparameter cgparam_vert_light_lightpos;
		CGparameter cgparam_vert_light_lightcolor;
		CGparameter cgparam_vert_light_envmapflag;

	pRenderProfileVertex(const char *profile_name,const char *vert_ext,CGprofile vert_prof) :
		vert_extension(vert_ext),
		vert_profile(vert_prof)
	{ 
		name=profile_name;
		cgprog_vert_shadow=0;
		cgparam_vert_shadow_modelviewproj=0;
		cgparam_vert_shadow_lightpos=0;
		cgprog_vert_ambient=0;
		cgparam_vert_ambient_modelviewproj=0;
		cgparam_vert_ambient_camerapos=0;
		cgparam_vert_ambient_envmapflag=0;
		cgprog_vert_light=0;
		cgparam_vert_light_camerapos=0;
		cgparam_vert_light_specular=0;
		cgparam_vert_light_lightpos=0;
		cgparam_vert_light_lightcolor=0;
		cgparam_vert_light_envmapflag=0;
	}
	virtual ~pRenderProfileVertex()
	{ reset(); }
		
	int test();
	void reset();
	void load(pRender* r);
	void draw(pRender *r,const pArray<pFace *>& f);
	void draw(pRender *r,const pMaterial& mat,const pBoundBox& bbox,void draw_geometry());
	void update_mat(pRender *r,const pMaterial *mat,int mode);
	void print(pString& str);
};

class pRenderProfileFrag1 : public pRenderProfile
{
	public:
		pString vert_extension;
		pString frag_extension;
		CGprofile vert_profile;
		CGprofile frag_profile;

		CGprogram cgprog_vert_ambient;
		CGparameter cgparam_vert_ambient_modelviewproj;
		CGparameter cgparam_vert_ambient_camerapos;
		CGparameter cgparam_vert_ambient_envmapflag;

		CGprogram cgprog_vert_shadow;
		CGparameter cgparam_vert_shadow_modelviewproj;
		CGparameter cgparam_vert_shadow_lightpos;

		CGprogram cgprog_vert;
		CGparameter cgparam_vert_modelviewproj;
		CGparameter cgparam_vert_camerapos;
		CGparameter cgparam_vert_lightpos;
		CGparameter cgparam_vert_lightcolor;
		CGparameter cgparam_vert_bumpfactor;
		CGparameter cgparam_vert_envmapflag;
		CGparameter cgparam_vert_tangent_u;
		CGparameter cgparam_vert_tangent_v;

		CGprogram cgprog_frag;
		CGparameter cgparam_frag_texturemap;
		CGparameter cgparam_frag_normalmap;
		CGparameter cgparam_frag_illuminationmap;
		CGparameter cgparam_frag_diffuse;
		CGparameter cgparam_frag_specular;

		int illuminationmap;
		int null_normalmap;
		int null_texturemap;

	pRenderProfileFrag1(const char *profile_name,const char *vert_ext,const char *frag_ext,CGprofile vert_prof,CGprofile frag_prof) :
		vert_extension(vert_ext),
		frag_extension(frag_ext),
		vert_profile(vert_prof),
		frag_profile(frag_prof)
	{ 
		name=profile_name;
		cgprog_vert_shadow=0;
		cgparam_vert_shadow_modelviewproj=0;
		cgparam_vert_shadow_lightpos=0;
		cgprog_vert_ambient=0;
		cgparam_vert_ambient_modelviewproj=0;
		cgparam_vert_ambient_camerapos=0;
		cgparam_vert_ambient_envmapflag=0;
		cgprog_vert=0;
		cgparam_vert_modelviewproj=0;
		cgparam_vert_camerapos=0;
		cgparam_vert_lightpos=0;
		cgparam_vert_lightcolor=0;
		cgparam_vert_bumpfactor=0;
		cgparam_vert_envmapflag=0;
		cgparam_vert_tangent_u=0;
		cgparam_vert_tangent_v=0;
		cgprog_frag=0;
		cgparam_frag_texturemap=0;
		cgparam_frag_normalmap=0;
		cgparam_frag_illuminationmap=0;
		cgparam_frag_diffuse=0;
		cgparam_frag_specular=0;
		illuminationmap=-1;
		null_normalmap=-1;
		null_texturemap=-1;
	}
	virtual ~pRenderProfileFrag1()
	{ reset(); }
		
	int test();
	void reset();
	void load(pRender* r);
	void draw(pRender *r,const pArray<pFace *>& f);
	void draw(pRender *r,const pMaterial& mat,const pBoundBox& bbox,void draw_geometry());
	void update_mat(pRender *r,const pMaterial *mat,int mode);
	void print(pString& str);
};

class pRenderProfileFrag2 : public pRenderProfile
{
	public:
		pString vert_extension;
		pString frag_extension;
		CGprofile vert_profile;
		CGprofile frag_profile;

		CGprogram cgprog_vert_ambient;
		CGparameter cgparam_vert_ambient_modelviewproj;
		CGparameter cgparam_vert_ambient_camerapos;
		CGparameter cgparam_vert_ambient_envmapflag;

		CGprogram cgprog_vert_shadow;
		CGparameter cgparam_vert_shadow_modelviewproj;
		CGparameter cgparam_vert_shadow_lightpos;

		CGprogram cgprog_vert;
		CGparameter cgparam_vert_modelviewproj;
		CGparameter cgparam_vert_camerapos;
		CGparameter cgparam_vert_envmapflag;
		CGparameter cgparam_vert_tangent_u;
		CGparameter cgparam_vert_tangent_v;

		CGprogram cgprog_frag[4];
		CGparameter cgparam_camerapos[4];
		CGparameter cgparam_specular[4];
		CGparameter cgparam_texture[4];
		CGparameter cgparam_normalmap[4];
		CGparameter cgparam_lightpos[4];
		CGparameter cgparam_lightcolor[4];
		CGparameter cgparam_bumpfactor[4];

	pRenderProfileFrag2(const char *profile_name,const char *vert_ext,const char *frag_ext,CGprofile vert_prof,CGprofile frag_prof) :
		vert_extension(vert_ext),
		frag_extension(frag_ext),
		vert_profile(vert_prof),
		frag_profile(frag_prof)
	{ 
		int i;
		name=profile_name;
		cgprog_vert_shadow=0;
		cgparam_vert_shadow_modelviewproj=0;
		cgparam_vert_shadow_lightpos=0;
		cgprog_vert_ambient=0;
		cgparam_vert_ambient_modelviewproj=0;
		cgparam_vert_ambient_camerapos=0;
		cgparam_vert_ambient_envmapflag=0;
		cgprog_vert=0;
		cgparam_vert_modelviewproj=0;
		cgparam_vert_camerapos=0;
		cgparam_vert_envmapflag=0;
		cgparam_vert_tangent_u=0;
		cgparam_vert_tangent_v=0;
		for( i=0;i<4;i++ )
		{
			cgprog_frag[i]=0;
			cgparam_camerapos[i]=0;
			cgparam_specular[i]=0;
			cgparam_texture[i]=0;
			cgparam_normalmap[i]=0;
			cgparam_lightpos[i]=0;
			cgparam_lightcolor[i]=0;
			cgparam_bumpfactor[i]=0;
		}
	}
	virtual ~pRenderProfileFrag2()
	{ reset(); }
		
	int test();
	void reset();
	void load(pRender* r);
	void draw(pRender *r,const pArray<pFace *>& f);
	void draw(pRender *r,const pMaterial& mat,const pBoundBox& bbox,void draw_geometry());
	void update_mat(pRender *r,const pMaterial *mat,int mode);
	void print(pString& str);
};

