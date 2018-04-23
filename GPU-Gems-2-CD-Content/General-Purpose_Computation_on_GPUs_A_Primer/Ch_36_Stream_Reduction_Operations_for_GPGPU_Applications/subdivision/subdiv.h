#include <brook/brook.hpp>

typedef struct STri_t{
   float4 A; // the last value of A indicates whether the edge AB is small
   float4 B; // enough to stop subdividing.  B.w also indicates if BC is small
   float4 C; // enough. C.w indicates if AC is small enough to stop subdividing
} STri;
#define tri_vertex_t float4
//Stores the neighbors of a given triangle
// the unused 'w' components act as holders
// for the recomputed neighbor list when a triangle is split
typedef struct Neighbor_t {
  float4 AB;// w = AB->B.x
  float4 BBC;// w = AB->B.y
  float4 ABB;// w = AB->B.z
  float4 BC;// w = BC->C.x
  float4 ACC;// w = BC->C.y
  float4 BCC;// w = BC->C.z
  float4 AC;// w = AC->A.x
  float4 AAB;// w = AC->A.y
  float4 AAC;// w = AC->A.z
}Neighbor;
