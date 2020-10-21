#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMath.h"
#else

#include <ATen/core/Generator.h>

TH_API void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
TH_API int THTensor_(equal)(THTensor *ta, THTensor *tb);

#if !defined(TH_REAL_IS_HALF)

TH_API void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
TH_API void THTensor_(maskedSelectBool)(THTensor *tensor, THTensor* src, THBoolTensor *mask);
TH_API void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
TH_API void THTensor_(maskedCopyBool)(THTensor *tensor, THBoolTensor *mask, THTensor* src);

TH_API ptrdiff_t THTensor_(numel)(THTensor *t);

TH_API void THTensor_(addmm)(THTensor *r_, THTensor *t, THTensor *mat1, THTensor *mat2, scalar_t beta, scalar_t alpha);
TH_API void THTensor_(addr)(THTensor *r_, THTensor *t, THTensor *vec1, THTensor *vec2, scalar_t beta, scalar_t alpha);

#if !defined(TH_REAL_IS_BOOL)
TH_API void THTensor_(mul)(THTensor *r_, THTensor *t, scalar_t value);
#endif

#if !defined(TH_REAL_IS_BFLOAT16)

void THTensor_(preserveReduceDimSemantics)(THTensor *r_, int in_dims, int reduce_dimension, int keepdim);

TH_API void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
TH_API void THTensor_(take)(THTensor *tensor, THTensor *src, THLongTensor *index);
TH_API void THTensor_(put)(THTensor *tensor, THLongTensor *index, THTensor *src, int accumulate);
TH_API void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, scalar_t val);

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

TH_API accreal THTensor_(dot)(THTensor *t, THTensor *src);

TH_API void THTensor_(addbmm)(THTensor *r_, THTensor *t, THTensor *batch1, THTensor *batch2, scalar_t beta, scalar_t alpha);
TH_API void THTensor_(baddbmm)(THTensor *r_, THTensor *t, THTensor *batch1, THTensor *batch2, scalar_t beta, scalar_t alpha);

TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, int64_t k, int dimension, int keepdim);
TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
TH_API accreal THTensor_(trace)(THTensor *t);


TH_API void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(renorm)(THTensor *r_, THTensor *t, scalar_t value, int dimension, scalar_t maxnorm);
TH_API void THTensor_(histc)(THTensor *hist, THTensor *tensor, int64_t nbins, scalar_t minvalue, scalar_t maxvalue);

#endif
#endif
#endif
#else
TH_API accreal THTensor_(dot)(THTensor *t, THTensor *src);
TH_API void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
#endif /* !defined(TH_REAL_IS_HALF) */
#endif /* TH_GENERIC_FILE*/
