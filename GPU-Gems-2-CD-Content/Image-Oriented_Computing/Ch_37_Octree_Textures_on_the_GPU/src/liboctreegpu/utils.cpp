/* ----------------------------------------------------------

Octree Textures on the GPU - source code - GPU Gems 2 release
                                                   2004-11-21

Updates on http://www.aracknea.net/octreetex
--
(c) 2004 Sylvain Lefebvre - all rights reserved
--
The source code is provided 'as it is', without any warranties. 
Use at your own risk. The use of any part of the source code in a
commercial or non commercial product without explicit authorisation
from the author is forbidden. Use for research and educational
purposes is allowed and encouraged, provided that a short notice
acknowledges the author's work.
---------------------------------------------------------- */
bool is_puiss2(int n)
{
  for (int i=n;i>1;i=i>>1)
    if (i & 1)
      return (false);
  return (true);
}

int puiss2(int n)
{
  int p=0;
  for (int i=n;i>1;i=i>>1)
    p++;
  return (p);
}

int next_puiss2(int n)
{
  return (1 << (puiss2(n)+(is_puiss2(n)?0:1)));
}
