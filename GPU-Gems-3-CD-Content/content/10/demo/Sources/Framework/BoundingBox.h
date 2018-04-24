#pragma once

class BoundingBox
{
public:
  inline BoundingBox();

  // create from minimum and maximum vectors
  inline BoundingBox(const Vector3 &vMin, const Vector3 &vMax);
  // create from set of points
  inline BoundingBox(const void *pPoints, int iNumPoints, int iStride);

  // set from minimum and maximum vectors
  inline void Set(const Vector3 &vMin, const Vector3 &vMax);

  // set from set of points
  inline void Set(const void *pPoints, int iNumPoints, int iStride);

  // returns size of bounding box
  inline Vector3 GetSize(void) const { return m_vMax - m_vMin; }

  // compute union
  inline void Union(const BoundingBox &bb2);

  Vector3 m_pPoints[8];
  Vector3 m_vMin, m_vMax;
};

/////////////////////

BoundingBox::BoundingBox()
{
  m_vMin = Vector3( FLT_MAX, FLT_MAX, FLT_MAX);
  m_vMax = Vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
}

// compute union of two bounding boxes
inline BoundingBox Union(const BoundingBox &bb1, const BoundingBox &bb2)
{
  Vector3 vMin, vMax;
  vMin.x = Min(bb1.m_vMin.x, bb2.m_vMin.x);
  vMin.y = Min(bb1.m_vMin.y, bb2.m_vMin.y);
  vMin.z = Min(bb1.m_vMin.z, bb2.m_vMin.z);
  vMax.x = Max(bb1.m_vMax.x, bb2.m_vMax.x);
  vMax.y = Max(bb1.m_vMax.y, bb2.m_vMax.y);
  vMax.z = Max(bb1.m_vMax.z, bb2.m_vMax.z);
  return BoundingBox(vMin, vMax);
}

// compute union
void BoundingBox::Union(const BoundingBox &bb2)
{
  (*this) = ::Union(*this, bb2);
}

// create from minimum and maximum vectors
inline BoundingBox::BoundingBox(const Vector3 &vMin, const Vector3 &vMax)
{
  Set(vMin, vMax);
}

// create from set of points
inline BoundingBox::BoundingBox(const void *pPoints, int iNumPoints, int iStride)
{
  Set(pPoints, iNumPoints, iStride);
}


// create from minimum and maximum vectors
inline void BoundingBox::Set(const Vector3 &vMin, const Vector3 &vMax)
{
  // calculate points
  m_pPoints[0] = Vector3(vMin.x, vMin.y, vMin.z);
  m_pPoints[1] = Vector3(vMax.x, vMin.y, vMin.z);
  m_pPoints[2] = Vector3(vMin.x, vMin.y, vMax.z);
  m_pPoints[3] = Vector3(vMax.x, vMin.y, vMax.z);
  m_pPoints[4] = Vector3(vMin.x, vMax.y, vMin.z);
  m_pPoints[5] = Vector3(vMax.x, vMax.y, vMin.z);
  m_pPoints[6] = Vector3(vMin.x, vMax.y, vMax.z);
  m_pPoints[7] = Vector3(vMax.x, vMax.y, vMax.z);
  m_vMin = vMin;
  m_vMax = vMax;
}

// create from set of points
inline void BoundingBox::Set(const void *pPoints, int iNumPoints, int iStride)
{
  // calculate min and max vectors
  m_vMin = Vector3( FLT_MAX, FLT_MAX, FLT_MAX);
  m_vMax = Vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
  char *pData = (char *)pPoints;
  for(int i=0; i<iNumPoints; i++)
  {
    const Vector3 &vPoint = *(Vector3*)pData;
    if(vPoint.x < m_vMin.x) m_vMin.x = vPoint.x;
    if(vPoint.y < m_vMin.y) m_vMin.y = vPoint.y;
    if(vPoint.z < m_vMin.z) m_vMin.z = vPoint.z;

    if(vPoint.x > m_vMax.x) m_vMax.x = vPoint.x;
    if(vPoint.y > m_vMax.y) m_vMax.y = vPoint.y;
    if(vPoint.z > m_vMax.z) m_vMax.z = vPoint.z;

    // next position
    pData += iStride;
  }
  // create from vectors
  Set(m_vMin, m_vMax);
}

