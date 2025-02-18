#!/usr/bin/env python3
import pickle
import socket
import time, sys, os
import subprocess
import copy
import random
from hexbytes import HexBytes

from console import *
from aux import TCP_mp, TCP_server, TCP_server2, l2d, getFolderSize
from init_weights import *

precision = 10**9

class seblogger:
	def __init__(self, logfile, header) -> None:
		self.logfile = logfile
		self.header = header
		self.log(self.header, 'w')
	
	def log(self, data:list, method='a'):
		data = ','.join([str(x) for x in data])
		with open(self.logfile, method) as file:
			file.write(f'{data}\n')


def getEnodes():
    return [peer['enode'] for peer in w3.geth.admin.peers()]

def getIps(__enodes = None):
    if __enodes:
        return [enode.split('@',2)[1].split(':',2)[0] for enode in __enodes]
    else:
        return [enode.split('@',2)[1].split(':',2)[0] for enode in getEnodes()]

# peers_geth is the set [ips] we get from geth.admin 
# peers_buffer is our local buffer set [ips]
global peers_geth, peers_buffer, logs
peers_geth, peers_buffer = set(), set()
logs = {}
price = 5*10**18

def peering(peer_IPs):
    """ Control routine for robot-to-robot dynamic peering 
	argument: dict {id:ip} comes from ARGoS controller
	"""

    global peers_geth, peers_buffer

    peers_geth_enodes = getEnodes()
    peers_geth = set(getIps(peers_geth_enodes))

    for peer_ID, peer_IP in peer_IPs.items():
        if peer_IP not in peers_buffer:
            enode = tcp_enode.request(host=peer_IP, port=5000)
            if 'enode' in enode:
                w3.geth.admin.addPeer(enode)
                peers_buffer.add(peer_IP)
                print(f'Added peer: {peer_ID}')#|{enode}')

    temp = copy.copy(peers_buffer)

    for peer_IP in temp:
        if peer_IP not in peer_IPs.values():
            enode = tcp_enode.request(host=peer_IP, port=5000)
            if 'enode' in enode:
                w3.provider.make_request("admin_removePeer",[enode])

                peers_geth_enodes = getEnodes()
                if enode not in peers_geth_enodes:
                    peers_buffer.remove(peer_IP)
                    print(f'Removed peer: {peer_IP}')#|{enode}')

    # for enode in peers_geth_enodes:
    # 	peer_IP = getIp(enode)
    # 	if peer_IP not in peer_IPs.values():
    # 		w3.provider.make_request("admin_removePeer",[enode])
    # 		peers_buffer.remove(peer_IP)
    # 		print('Timed out peer: %s|%s' % (peer_ID, enode))

    tcp_peering.setData(len(peers_geth))


def blockHandle():
	""" Every time new blocks are synchronized
	# etherium = w3.fromWei(w3.eth.gtBalance(w3.eth.coinbase), 'ether')
	# my_address = sc.functions.getAddress().call 
	"""
	try:
		money = sc.functions.getMoney().call()
		previousAccepted = sc.functions.getPreviousParticipants().call()
		previousAccepted = ','.join([str(x) for x in previousAccepted])
		blockNumber = sc.functions.getBlockNumber().call()
		blockHash = sc.functions.getBlockHash().call().hex()
		previousBlockHash = sc.functions.getPreviousBlockHash().call().hex()
		version = sc.functions.version().call()

		logs['sc'].log([version, money, previousBlockHash, blockNumber, blockHash, previousAccepted])
		logs['bc_size'].log([getFolderSize('/root/.ethereum/devchain/geth/chaindata')])
	except Exception:
		print(Exception)
	

def get_server_message(server):
	full_msg = b''
	new_msg = True
	end_message = False
	while not end_message:
		msg = server.recv(1024)

		if new_msg:
			msglen = int(msg[:HEADERSIZE])
			new_msg = False

		full_msg += msg

		if len(full_msg)-HEADERSIZE == msglen:
			print("full message received")
			full_msg = pickle.loads(full_msg[HEADERSIZE:])
			end_message = True
	return full_msg

def server_training(ip, port):
	version = sc.functions.version().call()
	precision = 10**9
	print("Updated the version to:", version)
	if version:
		weights = sc.functions.getWeights().call()
		print(f"here are my before trained first 5 weights {weights[:5]}")
		weights = [float(w)/precision for w in weights]
	else :
		weights = INIT_WEIGHTS()
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.connect((ip,port))
		msg = {'ID': robotID, 'weights': weights}
		msg = pickle.dumps(msg)
		msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
		server.send(msg)
		print("Weights sent!")

		server_dict = get_server_message(server)
		nb_sample = server_dict['nb_samples']
		weights = server_dict['weights']
		weights = [int(w*precision) for w in weights]
		print(f"i've trained on {nb_sample} and here are my first 5 weights {weights[:5]}")
		if version == sc.functions.version().call() and nb_sample:
			sc.functions.submitWeights(nb_sample,weights).transact()
	return version

def server_training_v2(ip, port):
	precision = 10**9
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.connect((ip,port))
		print('waiting for server reply...')
		server_dict = get_server_message(server)
		print('server should be ready')
		if version := sc.functions.version().call():
			weights = sc.functions.getWeights().call()
			print(f"here are my before trained first 5 weights {weights[:5]}")
			weights = [float(w)/precision for w in weights]
		else :
			weights = INIT_WEIGHTS()
		msg = {'ID': robotID, 'weights': weights, 'version': version}
		msg = pickle.dumps(msg)
		msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
		server.send(msg)
		print("Weights sent!")
		server_dict = get_server_message(server)
		nb_sample = server_dict['nb_samples']
		weights = server_dict['weights']
		weights = [int(w*precision) for w in weights]
		print(f"i've trained on {nb_sample} and here are my first 5 weights {weights[:5]}")
		# if version == sc.functions.version().call() and nb_sample:
		
		sc.functions.submitWeights(nb_sample,weights).transact({"value":5*10**18})

	return version

def server_training_v3(ip, port, address):
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.connect((ip,port))

		server_dict = get_server_message(server)

		weights = sc.functions.getWeights().call()
		weights = [float(w)/precision for w in weights]
		version = sc.functions.version().call()
		msg = {'ID': robotID, 'weights': weights, 'version': version, 'address': address}
		msg = pickle.dumps(msg)
		msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
		server.send(msg)

		server_dict = get_server_message(server)
		nb_sample = server_dict['nb_samples']
		weights = server_dict['weights']
		weights = [int(w*precision) for w in weights]
		try :
			sc.functions.submitWeights(nb_sample,weights).transact({"value":price})
		except Exception:
			print(Exception)

	return version

def compute_mae(wa, wb):
	return sum(abs(wa[i]-wb[i]) for i in range(len(wa)))/len(wa)

def bad_behaviour(current, previous, rtype=2):
	if rtype == 0 :
		nb_sample = 200
		weights = [random.randint(-5*10**8, 5*10**8) for _ in range(2848)] # exp 2 & 3
	elif rtype == 1:
		weights = sc.functions.getWeights().call() # exp 4
		nb_sample = 200
	else :
		nb_sample = 200
		mae = compute_mae(current, previous)
		weights = [elem+int((random.random())*mae*2) for elem in current] # exp 5

	try:
		sc.functions.submitWeights(nb_sample,weights).transact({"value":price})
	except Exception:
		print(Exception)
		


if __name__ == '__main__':
    
	w3 = init_web3()
	sc = registerSC(w3)
	bf = w3.eth.filter('latest')

	robotID = sys.argv[1]
	print(f"robot id : {robotID}")

	#########################################################################################
	################################## SETTING UP BEHAVIOR ##################################
	#########################################################################################

	my_host = subprocess.getoutput("ip addr | grep 172.18.0. | tr -s ' ' | cut -d ' ' -f 3 | cut -d / -f 1")
	my_port = 9890
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind((my_host, my_port))
		s.listen()
		print("waiting for robot type behavior")
		clientsocket, address = s.accept()
		my_message = clientsocket.recv(1024)
		print("here is the mes:", my_message)


	if my_message.decode() == "good":
		colab = True
	elif  my_message.decode() == "bad":
		colab = False
	print(colab)
	# colab=True

	#########################################################################################
	######################## SETTING UP INITIAL WIGHTS ON BLOCKCHAIN ########################
	#########################################################################################

	# precision = 10**16
	if int(robotID) == 1:
		nb_sample = 200
		weights = INIT_WEIGHTS()
		weights = [int(w*precision) for w in weights]
		rankWeights = [1, 1, 1, 1, 1, -1, -1]
		print(rankWeights)
		version = sc.functions.version().call()
		print("hey before sending weight, version =", version)
		try:
			sc.functions.setInitWeights(weights, rankWeights).transact()
		except Exception:
			print(Exception)
		version = sc.functions.version().call()
		print("hey just sent the weight, version =", version)

	logfolder = f'/root/logs/{robotID}/'
	os.system(f"rm -rf {logfolder}")
	os.makedirs(os.path.dirname(logfolder), exist_ok=True) 

	# Experiment money for each robot
	name 		  = 'smartcontract.txt'
	myParticipants=','.join([f'participant{i}' for i in range(7)])
	header        = ['version','money','previousBlockHash','blockNumber','blockHash', myParticipants]
	logs['sc'] = seblogger(logfolder+name, header)

	name = 'bc_size.csv'
	header =['MB']
	logs['bc_size'] = seblogger(logfolder+name, header)

	version = sc.functions.version().call()
	current_weights = sc.functions.getWeights().call()
	previous_weights = current_weights
	address = sc.functions.getAddress().call()

	time_sleep = 0.5
	training_interval_time = 100 # in seconds
	HEADERSIZE = 10

	tf_ip = "172.18.0.1"
	tf_port = 9801

	start_time = time.time()
	training_time = time.time()

	while True:
		# clock = time.time()-start_time

		if colab and time.time() - training_time > training_interval_time: 
			version = server_training_v3(tf_ip, tf_port, address)
			training_time = time.time()
		elif time.time() - training_time > training_interval_time: 
			bad_behaviour(current_weights, previous_weights, 2)
			training_time = time.time()

		if not colab and version < sc.functions.version().call():
			version = sc.functions.version().call()
			previous_weights = current_weights
			current_weights = sc.functions.getWeights().call()

		newBlocks = bf.get_new_entries()
		if newBlocks:
			blockHandle()

		time.sleep(time_sleep)
