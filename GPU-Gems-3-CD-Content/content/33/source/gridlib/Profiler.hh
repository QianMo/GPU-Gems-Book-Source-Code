#ifndef PROFILER_HH
#define PROFILER_HH

#ifndef WIN32

/*!
  \brief The low level hardware counter (running at CPU clock frequency)
*/
extern __inline__ unsigned long long rdtsc() {
    unsigned long long x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}

/*!	
  \brief This function starts recording the number of cycles elapsed.
  \internal "cpuid" is used before rdtsc to prevent out-of-sequence execution from producing wrong results.
  For more details, read Intel's application notes "Using the RDTSC instruction for performance monitoring".
*/
extern __inline__ unsigned long long startProfile() {
    unsigned long long v;
    // Intel recommends that a serializing instruction 
    // should be called before and after rdtsc. 
    // CPUID is a serializing instruction. 
    // ".align 128:" P 4 L2 cache line size 
    __asm__ __volatile__("xor %%eax,%%eax\n\t"
			 "cpuid\n\t"
			 "rdtsc\n\t"
			 "mov %%eax,(%0)\n\t"
			 "mov %%edx,4(%0)\n\t"
			 "xor %%eax,%%eax\n\t"
			 "cpuid\n\t"
			 : /* no output */ : "S"(&v) : "eax", "ebx", "ecx", "edx", "memory");
     return v;
}

/*!	
  \brief This function ends recording the number of cycles elapsed.
  \internal "cpuid" is used before rdtsc to prevent out-of-sequence execution from producing wrong results.
  For more details, read Intel's application notes "Using the RDTSC instruction for performance monitoring".
*/
extern __inline__ unsigned long long endProfile(unsigned long long& val) {
    unsigned long long v;
    __asm__ __volatile__("xor %%eax,%%eax\n\t"
			 "cpuid\n\t"
			 "rdtsc\n\t"
			 "mov %%eax,(%0)\n\t"
			 "mov %%edx,4(%0)\n\t"
			 "xor %%eax,%%eax\n\t"
			 "cpuid\n\t"
			 : /* no output */ : "S"(&v) : "eax", "ebx", "ecx", "edx", "memory");
    return v-val;
}

#else

// 32 bit cpuid/rdtsc

typedef struct cpuid_args_s {
	DWORD eax;
	DWORD ebx;
	DWORD ecx;
	DWORD edx;
} CPUID_ARGS;


inline void cpuid32(CPUID_ARGS* p) {
	__asm {
		mov	edi, p
		mov eax, [edi].eax
		mov ecx, [edi].ecx // for functions such as eax=4
		cpuid
		mov [edi].eax, eax
		mov [edi].ebx, ebx
		mov [edi].ecx, ecx
		mov [edi].edx, edx
	}
}

#define _RDTSC_STACK(ts) \
	__asm rdtsc \
	__asm mov DWORD PTR [ts], eax \
	__asm mov DWORD PTR [ts+4], edx

__inline unsigned long long _inl_rdtsc32() {
	unsigned long long t;
	_RDTSC_STACK(t);
	return t;
}
#define rdtsc _inl_rdtsc32

/*inline unsigned long long rdtsc()
{
   _asm _emit 0x0F
   _asm _emit 0x31
}*/

inline unsigned long long startProfile()
{
	CPUID_ARGS cpu;
	cpuid32(&cpu);
	unsigned long long t = rdtsc();
	cpuid32(&cpu);
	return t;
}

inline unsigned long long endProfile(unsigned long long start)
{
	CPUID_ARGS cpu;
	cpuid32(&cpu);
	unsigned long long t = rdtsc();
	cpuid32(&cpu);
	return t-start;
}

#endif // WIN32

#endif // PROFILER_HH
