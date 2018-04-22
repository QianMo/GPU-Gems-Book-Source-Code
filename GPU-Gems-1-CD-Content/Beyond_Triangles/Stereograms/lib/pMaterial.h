class pMaterial
{
	public:
		pVector diffuse;
		pVector specular;
		pVector selfillum;
		float opacity;
		float reflection;
		float refraction;
		float bump;
		float envscale;
		int texid;
		int texbumpid;
		int flags;
		pString name;
		pString texname;
		pString texbumpname;
		
	pMaterial();

	pMaterial(pMaterial& in);

	void operator=(const pMaterial& in);

	void write(FILE *fp) const;
	void read(FILE *fp);
};
