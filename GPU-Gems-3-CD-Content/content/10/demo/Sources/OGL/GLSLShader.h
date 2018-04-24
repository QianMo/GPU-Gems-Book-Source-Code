#pragma once

class GLSLShader
{
public:
  GLSLShader();
  ~GLSLShader();

  bool Load(const char *strVS, const char *strGS, const char *strFS, int iGSMaxVertices = 12);
  void Destroy(void);

  void SetMatrix(const char *strName, const Matrix &m);
  void SetMatrixArray(const char *strName, const Matrix *m, int iCount);
  void SetVector(const char *strName, const Vector4 &v);
  void SetVector(const char *strName, const Vector3 &v);
  void SetFloatArray(const char *strName, const float *p, int iCount);
  void SetInt(const char *strName, int i);
  void SetIntArray(const char *strName, const int *p, int iCount);

  void Activate(void);
  void Deactivate(void);

private:
  GLhandleARB m_FragmentShader;
  GLhandleARB m_VertexShader;
  GLhandleARB m_GeometryShader;
  GLhandleARB m_ProgramObject;
};