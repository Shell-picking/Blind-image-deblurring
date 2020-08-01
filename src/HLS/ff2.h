#include "fft_top.h"

struct wide_stream {
	ap_uint<32> data;
	ap_uint<1> user;
	ap_uint<1> last;
};

void fft2(wide_stream* in_stream, wide_stream* out_stream,int FDV);
