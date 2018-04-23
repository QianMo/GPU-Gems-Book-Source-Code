#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
using namespace std;

string findReplace (string s, string find, string replace) {
  unsigned int where;
  while ((where=s.find(find))!=string::npos) {
    s = s.substr(0,where)+replace+s.substr(where+find.length());
  }
  return s;
}
#define PB(X,Y,Z) ret.push_back(string3(X,Y,Z));
struct string3 {
  string s[3];
  string3(string a, string b, string c) {
    s[0]=a;s[1]=b;s[2]=c;
  }
};
static bool BrookFile =false;
vector<string3> VectorTypes() {
  vector<string3> ret;
  if (BrookFile) {
     PB("float", "float", "1");
     PB("float2", "float" , "2");
     PB("float3", "float" , "3");
     PB("float4", "float" , "4");
  }else {
     PB("vec<float,1> ","float","1");
     PB("vec<int,1> ","int","1");
     PB("vec<char,1> ","char","1");
     PB("vec<float,2> ","float","2");
     PB("vec<int,2> ","int","2");
     PB("vec<char,2> ","char","2");

     PB("vec<float,3> ","float","3");
     PB("vec<int,3> ","int","3");
     PB("vec<char,3> ","char","3");

     PB("vec<float,4> ","float","4");
     PB("vec<int,4> ","int","4");
     PB("vec<char,4> ","char","4");
  }
  return ret;
}
vector<string3> BasicTypes() {
  vector<string3> ret;
  PB("int","int","1");
  PB("char","char","1");
  PB("float","float","1");
  PB("double","double","1");
  PB("unsigned int","int","1");
  PB("bool","bool","1");
  return ret;
}
vector<string3> OperatorTypes() {
  vector<string3> ret;
  if (BrookFile) {
     PB("float", "float", "1");
     PB("float2", "float" , "2");
     PB("float3", "float" , "3");
     PB("float4", "float" , "4");
  }else {
     PB("vec<VALUE,1> ","VALUE","1");
     PB("vec<VALUE,2> ","VALUE","2");
     PB("vec<VALUE,3> ","VALUE","3");
     PB("vec<VALUE,4> ","VALUE","4");
  }
  return ret;
}

vector<string3> GeneralTypes() {
  vector <string3> ret = BasicTypes();
  vector<string3> vt =VectorTypes();
  ret.insert(ret.end(),vt.begin(),vt.end());
  return ret;
}

#undef PB
/*
vector<string3> operTypes = OperatorTypes();
vector<string3> basicTypes = BasicTypes();
vector <string3> generalTypes = GeneralTypes();
vector<string3> vectorTypes = VectorTypes();
*/
string preprocessTemplates (string s, vector<string3> replacementList) {
  string ret;
  s=findReplace (s,"template <class BRT_TYPE>","");
  //s=findReplace(s,"typename","");
  s=findReplace(s,"GCCTYPENAME","typename");
  s=findReplace(s,"INTERNALTYPENAME","");
  s=findReplace(s,"MSC_VER","ARRGH");
  for (unsigned int i=0;i<replacementList.size();++i) {
    
    string tmp = findReplace(s,"BRT_TYPE::TYPE",replacementList[i].s[1]);
    tmp = findReplace(tmp,"BRT_TYPE::size",replacementList[i].s[2]);
    ret+=tmp =findReplace(tmp,"BRT_TYPE",replacementList[i].s[0]);
    
  }
  return ret;
}
string removeTypenames (string in) {
  vector<string3> basicTypes = BasicTypes();
  for (unsigned int i=0;i<basicTypes.size();++i) {
    string findme ("typename "+basicTypes[i].s[0]);
    printf( "find %s\n",findme.c_str());
    in = findReplace(in,findme,basicTypes[i].s[0]);
  }
  in = findReplace (in, "typename vec","vec");
  in = findReplace (in, "typename VALUE","VALUE");
  return in;
}
string findBetween (string in, string name, string &pre, string &post) {
  pre="";
  post=in;
  unsigned int premarker = post.find ("#define "+name);
  string s;
  if (premarker!=string::npos) {
    s= post.substr(premarker);
    pre = post.substr(0,premarker);
	post="";
    string postmarkstr("#undef "+name);
    unsigned int postmarker = s.find(postmarkstr);
    if( postmarker!=string::npos) {
      postmarker+=postmarkstr.length();
      post = s.substr(postmarker);
      s = s.substr(0,postmarker);
      printf ("found %s\n",name.c_str());
    }
  } else {
    //printf("%s",post.c_str());
  }
  return s;
}
unsigned int countLines (string s) {
	unsigned int line=0;
	for (unsigned int i=0;i<s.length();++i) {
		if (s[i]=='\n')
			line++;
	}
	return line;
}
string lineString(unsigned int in) {
	char num [256];
	sprintf(num,"%d",in);
	return string("#line ")+num+string(" \"brtvector.hpp\"\n");
}
int main (int argc, char ** argv) {
  FILE * fp = fopen (argv[1],"rb");
  if (strstr(argv[1],".br")) {
     BrookFile=true;
     printf ("Brook File Identified");
  }
  struct stat st;
  stat (argv[1],&st);
  char * mem = (char *)malloc(st.st_size+1);
  fread(mem,st.st_size,1,fp);
  fclose(fp);
  string in(mem,st.st_size);
  in = findReplace(in,"\r\n","\n");
  in = findReplace(in,"\r","");
  free (mem);
  bool lin=true;
  
  if (argc>3)
     for (int i=3;i<argc;++i) {
	  if (strcmp(argv[i],"-noline")==0)
		  lin=false;             
     }
  string pre,general,firstpost,vectoronly,post,operonly,lastpost;
  
  general=findBetween(in,"GENERAL_TEMPLATIZED_FUNCTIONS",pre,post);
  unsigned int linestart=1;
  unsigned int line =linestart+countLines(pre);
  if (lin)
    pre= lineString(linestart)+pre;
  unsigned int generallines=countLines(general);
  if (lin)
	  general= lineString(line)+general;
  general+="\n";
  vectoronly=findBetween(post,"VECTOR_TEMPLATIZED_FUNCTIONS",firstpost,post);
  unsigned int firstpostlines=countLines(firstpost);
  line+=generallines;
  if (lin)
	  firstpost = lineString(line)+firstpost;
  line+=firstpostlines;
  unsigned int vectoronlylines=countLines(vectoronly);
  if (lin)
	  vectoronly= lineString(line)+vectoronly;
  line+=vectoronlylines;
  vectoronly+="\n";
  operonly=findBetween(post,"OPERATOR_TEMPLATIZED_FUNCTIONS",post,lastpost);
  unsigned int postlines = countLines(post);
  if (lin)
	  post+=lineString(line)+post;
  line+=postlines;
  unsigned int operonlylines=countLines(operonly);
  if (lin)
	  operonly= lineString(line)+operonly;    
  operonly+="\n";
  line+=operonlylines;
  if (lin)
	  lastpost= lineString(line)+lastpost;
  FILE * o = fopen (argv[2],"w");
  string writeme = findReplace(pre,"BRTVECTOR_HPP","VC6VECTOR_HPP");
#define WRITEME fwrite (writeme.c_str(),writeme.length(),1,o)
  WRITEME;
  writeme=removeTypenames (preprocessTemplates(general,GeneralTypes()));
  WRITEME;
  writeme=firstpost;
  WRITEME;
  writeme=removeTypenames(preprocessTemplates(vectoronly,VectorTypes()));
  WRITEME;
  writeme=post;
  WRITEME;
  writeme= removeTypenames(preprocessTemplates(operonly,OperatorTypes()));
  WRITEME;
  writeme=lastpost;
  WRITEME;
  fclose(o);
  return 0;
}


