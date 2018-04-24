/**
 * @file	rijndael.h
 * @brief	AES-Rijndael cryptograph
 * @author	Yamanouchi_Takeshi@sega.co.jp
 *
 */

#ifndef RIJNDAEL_H_INCLUDED
#define RIJNDAEL_H_INCLUDED
#ifdef __cplusplus
extern "C" {
#if 0
}
#endif
#endif

// Rijndael params, key schedule
typedef struct {
    int		num_key;		// key length (4, 6, 8 word)
    int		num_round;		// number of round(10, 12, 14)
    int		num_block;		// block size (4 word, fixed)
    uint32_t	enc_key[60];		// encrypt key schedule
    uint32_t	dec_key[60];		// decrypt key schedule
} RijnKeyParam;

// set params, key schedule
void rijn_expand_key(RijnKeyParam *kp, int nb, int nk, const char *key);
// encrypt
void rijn_encrypt(const RijnKeyParam *kp, char *buff);
// decrypt
void rijn_decrypt(const RijnKeyParam *kp, char *buff);

#ifdef __cplusplus
}
#endif
#endif
