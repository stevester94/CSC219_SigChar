from timeit import timeit
import socket
import struct
import random
import deepsig_accessor as ds_accessor

BUFFER_SIZE = 20

def test_send():



    floatlist = ds_accessor.get_data_samples(["AM-DSB-SC"], [30])[0][0]

    print(floatlist)

    buf = struct.pack('%sf' % len(floatlist), *floatlist)

    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 1337))

        print("Sum of payload: %f" % sum(floatlist))

        sock.sendall(buf)

        response = sock.recv(BUFFER_SIZE)
        sock.close()

        classification, confidence = struct.unpack("If", response)
        print("Response %d,%f" % (classification, confidence))

# print("Time: %s" % (timeit(test_send, number=100)/100))
test_send()
