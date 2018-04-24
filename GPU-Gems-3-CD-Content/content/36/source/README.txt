GPU Gems 3 "AES encryption and decryption on the GPU" sample program.

This sample program has two functions.
usage: ./xfeedback_aes <test | bench>

1. test --- verify example vectors on FIPS-197, Appendix C
	Each ciphertexts have to be followings:
	128 bit key	69c4e0d86a7b0430d8cdb78070b4c55a
	192 bit key	dda97ca4864cdfe06eaf70a0ec0d7191
	256 bit key	8ea2b7ca516745bfeafc49904b496089

2. bench --- measure throughput of encryption and transfer.
	You can get the results at each batch size.


Captions of source codes:
aes.vpt			A vertex program of GPU-based AES implementation.
xfeedback_aes.cc	An example of transform feedbak mode.
rijndael.h, rijndael.c	CPU-based AES implementation. We use it to expand a symmetric-key into key schedule.

--
Yamanouchi_Takeshi@sega.co.jp
