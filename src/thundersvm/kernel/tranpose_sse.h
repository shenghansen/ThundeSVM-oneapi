#ifndef _TRANSPOSE_SSE_
#define _TRANSPOSE_SSE_
extern "C"
{
void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size);
}
#endif
