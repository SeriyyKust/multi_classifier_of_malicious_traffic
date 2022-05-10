import os
import socket
from re import *
import random
import config

input_folder = config.path_input
output_folder = config.path_output
files = os.listdir(input_folder)

for i in range(0,len(files)):
	print(str(i) + ') ' + files[i])
change = int(input())
file = files[change]
input_file = open(input_folder+'\\'+file,'r')
output_file = open(output_folder + '\\' + file,'w')


def file_write(value):
	output_file.write("\"" + str(value) + "\",")

#max_count_packages = 15000
max_count_packages = int(input("Enter count packages: "))
if max_count_packages == -1:
	max_count_packages = random.randint(25000,150000)
	print('Max count: ' + str(max_count_packages))
f = False
count = 0
tt = 0

for line in input_file:
	if not f:
		f = True
		continue
	line = line[1:len(line) - 2]
	elements = line.split("\",\"")
	
	print(elements)
	# time
	tmp = float(elements[1])
	tmp_2 = tmp - tt
	file_write(str(tmp_2))
	tt = tmp

	#protocol
	try:
		socket.getprotobyname(str(elements[4]))
		file_write(socket.getprotobyname(str(elements[4])))
	except:
		if str(elements[4]) == "RTSP":
			file_write(6)
		elif str(elements[4]) == "TLSv1.2" or str(elements[4]) == "TLSv1":
			file_write(143)
		elif str(elements[4]) == "HTTP" or str(elements[4]) == "0x0820" or str(elements[4]) == "0x0c00":
			file_write(144)
		elif str(elements[4]) == "MDNS" or str(elements[4]) == "DNS":
			file_write(145)
		elif str(elements[4]) == "IPv4":
			file_write(4)
		elif str(elements[4]) == "IGMPv2":
			file_write(2)
		elif str(elements[4]) == "ARP":
			file_write(91)
		else:
			file_write(250)


	#flags
	# SYN
	check = search("SYN", str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)
	# ACK
	check = search("ACK", str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)

	#Length all
	file_write(elements[5])

	#length add
	check = search(r'Len=\d*', str(elements[6]))
	if check is not None:
		file_write((check[0])[4:])
	else:
   		file_write(0)

	#Addition info
	#Sequence number 
	check = search(r'Seq=\d*', str(elements[6]))
	print(check)
	if check is not None:
		file_write((check[0])[4:])
	else:
		file_write(-1)


	# TCP CHECKSUM 
	check = search(r'\[TCP CHECKSUM INCORRECT\]', str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)


	# UDP CHECKSUM
	check = search(r'\[UDP CHECKSUM INCORRECT\]', str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)


	# Out-of-order
	check = search(r'\[TCP Out-Of-Order\]', str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)


	check = search(r'\[TCP Dup ACK', str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)

	check = search(r'\[TCP Previous segment not captured\]', str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)

	#[TCP Retransmission]
	check = search(r'\[TCP Retransmission\]', str(elements[6]))
	if check is not None:
		file_write(1)
	else:
		file_write(0)


	if file.find("ACK") != -1:
		output_file.write("\"" + str(2) + "\"\n")
	elif file.find("SYN") != -1:
		output_file.write("\"" + str(1) + "\"\n")
	elif file.find("UDP") != -1:
		output_file.write("\"" + str(3) + "\"\n")
	else:
		output_file.write("\"" + str(0) + "\"\n")



	count += 1

	if count >= max_count_packages:
		break
