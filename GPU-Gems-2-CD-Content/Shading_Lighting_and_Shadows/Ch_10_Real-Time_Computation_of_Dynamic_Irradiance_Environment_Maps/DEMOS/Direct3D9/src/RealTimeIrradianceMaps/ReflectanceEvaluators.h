#ifndef __REFLECTANCE_EVALUATORS_H_included_
#define __REFLECTANCE_EVALUATORS_H_included_

struct SH_Reflectance_Al_Evaluator
{
    virtual double operator()(int l) const = 0;
};

struct Lambert_Al_Evaluator: public SH_Reflectance_Al_Evaluator
{
    virtual double operator()(int l) const;
};

struct Phong_Al_Evaluator: public SH_Reflectance_Al_Evaluator
{
    Phong_Al_Evaluator( double spec ): m_specular(spec) { }
    virtual double operator()(int l) const;

protected:
    Phong_Al_Evaluator(): m_specular(1.f) { }
    double m_specular;
};

#endif