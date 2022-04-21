============ geth-argos ===============
==                                   ==
== ARGoS Blockchain Python Interface ==
==                                   ==
=======================================

AUTHOR: 

Alexandre Pacheco  <alexandre.melo.pacheco@gmail.com>

CREDITS:

Ken Hasselmann for argos-python wrapper <https://github.com/KenN7/argos-python>

Volker Strobel for docker-geth setup <https://github.com/Pold87/blockchain-swarm-robotics>

DATE: 22/06/2021


To-do-list:
- (DONE) fix Event 
- (DONE) Trigger experiment finish from criteria; restart automatically; iterate experimental parameters
- (DONE) Move peering to docker
- (medium) Make RPYC server hosted in docker containers (one web3.py instance per container)
- (medium) Create "reset-geth.sh" which does not reinialize docker but rather reset geth folder and process in every container (if faster/more efficient)
- (medium) Improve "stop_network.sh" since docker stop and docker rm is not working properly
- (DONE) open geth console with ./tmux-all for all docker containers
- (HALF-DONE) Create data recording files consistent with real robots (fix timestamp in simulation, add hash to sc.csv)
- (low) Link robot ID to docker IP (preferably 172.27.1.xxx where xxx is robotId)
- (low) Understand why the logging package messes up... They say its fixed in 3.8, so maybe exploit that in simulation although the robots have 3.6
- (low) Expose entire w3 instance in RPYC rather than individual functions

Improvements to argos-python:
- Implement the positioning sensor
- Implement E-puck ring LEDs
