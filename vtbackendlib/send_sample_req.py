#!/usr/bin/env python3
"""
Author : ZHANG dongx (dongxu.c.zhang@nokia-sbell.com)
Date: 2018-05-10 - initial creation
      2019-01-16 - added mmap to work with the new remote control framework
--------------------------------------------------------------------------
Discription:
   This script is for fronthaul demo. Run on vt855 (amc726). Should be used 
   in coordination with the corresponding FPGA image which receives the 
   request packet and sends 20 sample packets in reply.

   (2019-01-16) Memory-map scheme is used to share date with other process.
   VadaTech VT855 chassis has ~30 second Ethernet switch fabric delay, it is
   neccessary to seperate this sample requesting process with other processes
   that handles GUI related functions via mmap scheme.
"""

import mmap
import socket
import selectors
import sys
from time import sleep


PKT_BYTES = 1200
N_REQ = 9999

_f = open('vt855_mmap_file.bin', 'rb+')
_m = mmap.mmap(_f.fileno(), 48000,  access=mmap.ACCESS_WRITE)

msg_to_fpga = 'abcdefghijklmnopqrstuvwxyz'
data = msg_to_fpga.encode('ascii')
sel = selectors.DefaultSelector()


def set_sock(rcv_port = 1060):
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Tx Socket Created")
    try:
        sock_rcv = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    except sock_rcv.error:
        print('Failed to create rcv_socket')
        sys.exit()
    print("Rx Socket Created")
    
    print(sock_rcv.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
    sock_rcv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    print('SOCKET SIZE = {}'.format(sock_rcv.getsockopt(socket.SOL_SOCKET,
                                                        socket.SO_RCVBUF)))
    sock_rcv.bind(('192.168.41.212', rcv_port))
    print('Listening at {}'.format(sock_rcv.getsockname()))
    return sock_send, sock_rcv


def sendudp(udp_sock, msg_str, ipaddr, udpport):
    """ for test use only """
    udp_sock.sendto(msg_str.encode('ascii'), (ipaddr, udpport))


def getdata(s_snd, s_rcv, snd_udpport, n_pkt):
    msg_to_fpga = 'abcdefghijklmnopqrstuvwxyz'
    data = msg_to_fpga.encode('ascii')
    s_snd.sendto(data, ('192.168.41.198', snd_udpport))  # ge1, to FPGA
    received_packets = read_packets(s_rcv, n_pkt)
    return received_packets


def read_packets(s_rcv, receive_num):
    i = 0
    received_packets = bytearray(b'')
    ready = sel.select(timeout=0)  # non-blocking select
    if ready:  # once ready, read enough bytes before return
        while (i < receive_num):
            data = s_rcv.recv(PKT_BYTES)
            print('pkt', i+1, 'len =', len(data))
            received_packets.extend(data)
            i = i + 1
    else:
        print('nothing arrived yet')
    return received_packets


if __name__ == '__main__':
    n_pkt = 20
    snd_udpport = 9220
    rcv_udpport = 1060
    socksnd, sockrcv = set_sock(rcv_port = rcv_udpport)
    sel.register(sockrcv, selectors.EVENT_READ)
    i = 0
    for i in range(N_REQ):
        sleep(1)
        alldata = getdata(s_snd = socksnd, s_rcv = sockrcv,
                          snd_udpport = snd_udpport, n_pkt = n_pkt)
        if alldata:
            if i % 2 == 0:
                _m[0:24000] = alldata
            else:
                _m[24000:48000] = alldata
            print('write into mmap file.')
    print('finish capturing')
    socksnd.close()
    sockrcv.close()
    _m.close()
    _f.close()
    print('close file')
