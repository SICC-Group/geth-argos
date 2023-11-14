#!/usr/bin/env python3
import pickle
import socket
import time, sys, os
import datetime
import subprocess
import copy
import random
from hexbytes import HexBytes
from copy import deepcopy

from console import *
from aux import TCP_mp, TCP_server, TCP_server2, l2d, getFolderSize
from init_weights import *

# work with all aggregation

PRECISION = 10**9
AGGREGATION_NAMES = [
    "multiKrum", "geoMed", "autoGM", "median",
    "trimmedMean", "centeredClipping", "clustering",
    "clippedClustering","DnC", "signGuard", "mean"
]
price = 5*10**16  # 5*10**18

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
	# try:
	version = sc.functions.version().call()
	blockNumber = sc.functions.getBlockNumber().call()
	mean0 = sc.functions.getmeanW0().call()

	logs['sc'].log([blockNumber, version, mean0/PRECISION])
	logs['bc_size'].log(
		[getFolderSize('/root/.ethereum/devchain/geth/chaindata')]
	)
	# except Exception:
	# 	print(Exception)
	

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


def server_training_grad(ip, port, address):
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.connect((ip,port))
		server_dict = get_server_message(server)  # 'info': "ready"

		all_weights = {}
		version = sc.functions.version().call()
		for aggregation in AGGREGATION_NAMES:
			weights = weights_to_get[aggregation]().call()
			weights = [float(w / PRECISION) for w in weights]
			all_weights[aggregation] = weights
			# time.sleep(0.5)
		msg = {
			'ID': robotID,
			'all_weights': all_weights,
			'version': version,
			'address': address,
			'colab': colab
		}
		msg = pickle.dumps(msg)
		msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
		server.send(msg)

		server_dict = get_server_message(server)
		updated_weights = server_dict['updated_weights']
		aggregated_gradients = server_dict['aggregated_gradients']

		if len(updated_weights) > 1 and len(aggregated_gradients) > 1:
			for aggregation in AGGREGATION_NAMES:
				weights = updated_weights[aggregation]
				weights = [int(w * PRECISION) for w in weights]
				weights_to_submit[aggregation](weights).transact()
				# gradients = aggregated_gradients[aggregation]
				# gradients = [int(g * PRECISION) for g in gradients]
				# gradients_to_submit[aggregation](gradients).transact()
				# time.sleep(0.5)
				# {"value":price}

	return version

def compute_mae(wa, wb):
	return sum(abs(wa[i]-wb[i]) for i in range(len(wa)))/len(wa)

def bad_behaviour(ip, port, adversary):
	assert adversary in ["noise", "last-round", "IPM", "ALIE"]
	
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.connect((ip,port))
		server_dict = get_server_message(server)  # 'info': "ready"

		msg = {
			'ID': robotID, 'adversary': adversary, 'colab': colab,
		}
		msg = pickle.dumps(msg)
		msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
		server.send(msg)

		server_dict = get_server_message(server)
		updated_weights = server_dict['updated_weights']
		aggregated_gradients = server_dict['aggregated_gradients']
		if len(updated_weights) > 1 and len(aggregated_gradients) > 1:
			for aggregation in AGGREGATION_NAMES:
				weights = updated_weights[aggregation]
				weights = [int(w * PRECISION) for w in weights]
				weights_to_submit[aggregation](weights).transact()


if __name__ == '__main__':
	current_weights = {}
    
	w3 = init_web3()
	sc = registerSC(w3)
	bf = w3.eth.filter('latest')
	weights_to_submit = {
		"multiKrum": sc.functions.submultiKrumW,
        "geoMed": sc.functions.subgeoMedW,
        "autoGM": sc.functions.subautoGMW,
        "median": sc.functions.submedianW,
        "trimmedMean": sc.functions.subtrimmedMeanW,
        "centeredClipping": sc.functions.subcenteredClippingW,
        "clustering": sc.functions.subclusteringW,
        "clippedClustering": sc.functions.subclippedClusteringW,
        "DnC": sc.functions.subDnCW,
        "signGuard": sc.functions.subsignGuardW,
        "mean": sc.functions.submeanW,
    }
	# gradients_to_submit = {
    #     "multiKrum": sc.functions.submultiKrumG,
    #     "geoMed": sc.functions.subgeoMedG,
    #     "autoGM": sc.functions.subautoGMG,
    #     "median": sc.functions.submedianG,
    #     "trimmedMean": sc.functions.subtrimmedMeanG,
    #     "centeredClipping": sc.functions.subcenteredClippingG,
    #     "clustering": sc.functions.subclusteringG,
    #     "clippedClustering": sc.functions.subclippedClusteringG,
    #     "DnC": sc.functions.subDnCG,
    #     "signGuard": sc.functions.subsignGuardG,
    #     "mean": sc.functions.submeanG,
    # }
	weights_to_get = {
        "multiKrum": sc.functions.getmultiKrumW,
        "geoMed": sc.functions.getgeoMedW,
        "autoGM": sc.functions.getautoGMW,
        "median": sc.functions.getmedianW,
        "trimmedMean": sc.functions.gettrimmedMeanW,
        "centeredClipping": sc.functions.getcenteredClippingW,
        "clustering": sc.functions.getclusteringW,
        "clippedClustering": sc.functions.getclippedClusteringW,
        "DnC": sc.functions.getDnCW,
        "signGuard": sc.functions.getsignGuardW,
        "mean": sc.functions.getmeanW,
    }
	# gradients_to_get = {
    #     "multiKrum": sc.functions.getmultiKrumG,
    #     "geoMed": sc.functions.getgeoMedG,
    #     "autoGM": sc.functions.getautoGMG,
    #     "median": sc.functions.getmedianG,
    #     "trimmedMean": sc.functions.gettrimmedMeanG,
    #     "centeredClipping": sc.functions.getcenteredClippingG,
    #     "clustering": sc.functions.getclusteringG,
    #     "clippedClustering": sc.functions.getclippedClusteringG,
    #     "DnC": sc.functions.getDnCG,
    #     "signGuard": sc.functions.getsignGuardG,
    #     "mean": sc.functions.getmeanG,
    # }

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

	# PRECISION = 10**16
	if int(robotID) == 1:
		weights = INIT_WEIGHTS()
		weights = [int(w * PRECISION) for w in weights]
		# rankWeights = [1, 1, 1, 1, 1, -1, -1]
		# print(rankWeights)
		version = sc.functions.version().call()
		print("hey before sending weight, version =", version)
		try:
			sc.functions.setInitWeights(weights).transact()
		except Exception:
			print(Exception)
		version = sc.functions.version().call()
		print("hey just sent the weight, version =", version)

	logfolder = f'/root/logs/{robotID}/'
	os.system(f"rm -rf {logfolder}")
	os.makedirs(os.path.dirname(logfolder), exist_ok=True) 

	# Experiment money for each robot
	name 		  = 'smartcontract.txt'
	header        = [
		'blockNumber','version',
		'first element of mean\'s weights'
	]
	logs['sc'] = seblogger(logfolder+name, header)
	name = 'bc_size.csv'
	header =['MB']
	logs['bc_size'] = seblogger(logfolder+name, header)

	address = sc.functions.getAddress().call()
	
	training_interval_time = 100 # in seconds
	HEADERSIZE = 10

	torch_ip = "172.18.0.1"
	torch_port = 9801

	training_time = time.time()
	time.sleep(15.0)
	# version = sc.functions.version().call()
	# for aggregation in AGGREGATION_NAMES:
	# 	weights = weights_to_get[aggregation]().call()
	# 	current_weights[aggregation] = weights
	# previous_weights = deepcopy(current_weights)

	while True:
		if colab and time.time() - training_time > training_interval_time:
			training_time = time.time()
			version = server_training_grad(torch_ip, torch_port, address)
		elif time.time() - training_time > training_interval_time:
			training_time = time.time()
			time.sleep(10)
			# previous_weights = deepcopy(current_weights)
			# for aggregation in AGGREGATION_NAMES:
			# 	weights = weights_to_get[aggregation]().call()
			# 	current_weights[aggregation] = weights
			
			bad_behaviour(torch_ip, torch_port, adversary="IPM")
			# ["noise", "last-round", "IPM", "ALIE"]

		newBlocks = bf.get_new_entries()
		if newBlocks:
			blockHandle()
		time.sleep(0.5)
