(Readme file for miliankelib)

# 1. Summary
The `miliankelib` module contains utilities for 米联客 FPGA boards.

The _mz7030fa.py_ file contains everything you need to develop video capturing applications.
Currently (2019 12.20) we only have one MZ7030FA and one MZ7010 board. They baisically have the same functionalities. The _mz7030fa.py_ contains the API wrapper for operating on either of these two boards.

In the _**FogDemoData**_ subfolder, there is a board simulation script, which can act like the FPGA board - generating frames at specific frame rate and communicate with the upper machine via tcp/IP interface. It's easy to run this script on any machine (with python-opencv) and test application codes without having the FPGA board.。
# 2. Usage
> **NOTE**: The FPGA board has to be properly configured before using any of the Python APIs. The FPGA project archive (along with Zynq code image) can be found in the _FPGA_prj_ backup folder in the Z800 workstation.

There are three ways to use the `mz7030fa` module:
 1. Test FPGA board functionalities on a PC connectting to the board directly.
 2. Based on 1, further provide a "proxy layer" to forward the video stream to other PCs.
 3. On any PC, connect to a proxy and support app development just like using the OpenCV VideoCapture interface.

The following subsections illustrates usage examples.
## 2.1 Example 1

## 2.2 Example 2

## 2.3 Example 3