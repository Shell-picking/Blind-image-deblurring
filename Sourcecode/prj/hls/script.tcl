############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project fft2
set_top fft2
add_files hls/fft_single/fft2.cpp
add_files hls/fft_single/fft_top.cpp
add_files -tb hls/fft_single/fft2_test.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb hls/fft_single/test2.jpg -cflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xc7z020clg400-1} -tool vivado
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog
#source "./fft2/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -rtl verilog -format ip_catalog
