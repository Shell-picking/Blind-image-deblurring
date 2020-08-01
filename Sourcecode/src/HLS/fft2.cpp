#include "ff2.h"
#include <typeinfo>

void fft2(wide_stream* in_stream, wide_stream* out_stream,int FDV)
{
#pragma HLS INTERFACE axis port=in_stream bundle=INPUT
#pragma HLS INTERFACE axis port=out_stream bundle=OUTPUT

#pragma HLS INTERFACE s_axilite port=FDV bundle=CONTROL_BUS offset=0x14 clock=AXI_LITE_clk
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS clock=AXI_LITE_clk
#pragma HLS INTERFACE ap_stable port=FDV

	static cmpxDataIn inbuffer[128][128];
	static cmpxDataOut temp[128][128],copy[128],out[128][128];
//	static int buffer[128][256];

	double sc = ldexp(1.0, 9-1);
	bool ovfl;
#pragma HLS dataflow
//	for(int i = 0; i < 128;   i++)
//		{
//			copy[i] = cmpxDataOut(data_out_t(0,0));
//			for(int r = 0; r < 128; r++){
//				inbuffer[i][r] = cmpxDataIn(0,0);
//				temp[i*128+r] = cmpxDataOut(data_out_t(0,0));
//				out[i][r] = cmpxDataOut(data_out_t(0,0));
//			}
//		}
	for(int i = 0; i < 128;   i++)
	{
		for(int r = 0; r < 128/4; r++){
//			#pragma HLS pipeline II=4
				ap_uint<32> dat = in_stream->data;
				inbuffer[i][4*r] = cmpxDataIn((double(dat.range(7,0))/sc),0);
				inbuffer[i][4*r+1] = cmpxDataIn((double(dat.range(15,8))/sc),0);
				inbuffer[i][4*r+2] = cmpxDataIn((double(dat.range(23,16))/sc),0);
				inbuffer[i][4*r+3] = cmpxDataIn((double(dat.range(31,24))/sc),0);
				++in_stream;
			}
	}
	for(int i = 0; i < 128;   i++)
	{
		fft_top(1,inbuffer[i],temp[i],&ovfl);
	}
//	double k = ldexp(1.0, 6);
//
//	for(int i = 0; i < 128;   i++)
//		{
//		for(int j = 0; j < 128;   j++)
//			copy[j] = cmpxDataIn(data_in_t(double(temp[i + j*128].real())*k),data_in_t(double(temp[i + j*128].imag())*k));
//		fft_top(1,copy,out[i],&ovfl);
//		}
	int sn = ldexp(1.0, 6);
    for(int r = 0; r < 128; r++){
//    #pragma HLS pipeline II=4
		for(int c = 0; c < 128/2; c++){
			ap_uint<32> dat;
			dat.range(7,0) =  (double(temp[r][2*c].real())*sn);
			dat.range(15,8) =  (double(temp[r][2*c].imag())*sn);
			dat.range(23,16) =  (double(temp[r][2*c+1].real())*sn);
			dat.range(31,24) =  (double(temp[r][2*c+1].imag())*sn);
			out_stream->data = dat;
			out_stream->user = (r == 0 && c == 0)? 1: 0;
			out_stream->last = (r == 128-1 && c == 128/2-1)? 1: 0;
			++out_stream;
		}
	}
}
