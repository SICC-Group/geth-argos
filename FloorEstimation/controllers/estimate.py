import random, math
import sys
import os
experimentFolder = os.environ["EXPERIMENTFOLDER"]
sys.path.insert(1, experimentFolder+'/controllers')
sys.path.insert(1, experimentFolder)
import time

#####################################################

# Some parameters
isbyz = False
robot_speed = 10

# Some required transaction appendices
global gasLimit
global gasprice
global gas
gasLimit = 0x9000000
gasprice = 0x900000
gas = 0x90000

# Some experiment variables
global estimate, totalWhite, totalBlack
estimate = 0
totalWhite = 0
totalBlack = 0

global votetimer, updatetimer
updatetimer = time.time()
votetimer = time.time()

def init():
    global ticketPrice, rw, gs, myKey, w3, robotID

    #####################################################
    ## TEMPORARY SOLUTION: Read ID from File and delete
    IDfile = open("ids.txt", "r")
    IDs = IDfile.readlines()
    IDfile.close()
    IDfile = open("ids.txt", "w")
    robotID = int(IDs[0].strip())
    del IDs[0]
    if IDs:
        for ID in IDs:
            IDfile.write(ID)
    else:
        for i in range(1, robotID+1):
            IDfile.write('%s\n' % i)
    IDfile.close()

    ## Desired way to get ID (implement in py wrapper) 
    # robotId = robot.get_id()

    namePrefix = 'ethereum_eth.'+str(robotID)
    containersFile = open('identifiers.txt', 'r')
    for line in containersFile.readlines():
        if line.__contains__(namePrefix):
            ip = line.split()[-1]

    print(robotID, ip)

    # #####################################################
    # ## ERROR METHOD: import w3 multiple times; 
    # ## Ask ilpincy about argos interpreter/subinterpreters
    # from console import init_web3, registerSC
    # w3 = init_web3(ip)

    ## CURRENT SOLUTION: connect to a w3 wrapper hosted via rpyc
    import rpyc
    conn = rpyc.connect("localhost", 4000)
    w3 = conn.root

    # Do stuff over rpyc
    print(w3.getBalance(robotID-1))
    print(w3.getKey(robotID-1))
    print(w3.isMining(robotID-1))

    w3.minerStart(robotID-1)
    w3.transact(robotID-1, 'registerRobot')
    ticketPrice = w3.call(robotID-1, 'getTicketPrice')
    print(ticketPrice)
    myKey = w3.getKey(robotID-1)

    rw=RandomWalk(robot_speed)
    gs=GroundSensor()

def controlstep():
    global  votetimer, updatetimer
    global  rw, gs, w3, myKey, robotID
    
    rw.walking()
    gs.sensing()
    Estimate()


    if time.time()-votetimer > 30:
        votetimer = time.time()
        try:
            vote = int(estimate*1e7)
            ticketPriceWei = w3.toWei(robotID-1, ticketPrice)
            votehash = w3.transact2(robotID-1, 'sendVote', vote, {"from":myKey, "value":ticketPriceWei, "gas":gasLimit, "gasPrice":gasprice})
            print(votehash)
        except ValueError:
            print("Vote Failed. No Balance: ", w3.getBalance(robotID-1))
        except:
            print("Vote Failed. Unknown") 

        if robotID == 1:
        # print("Voted Successfully. Estimate: ", round(estimate,2))
            nrobs = w3.call(robotID-1, 'robotCount')
            mean = w3.call(robotID-1, 'getMean')*1e-7
            bn = w3.blockNumber(robotID-1)
            bal = w3.getBalance(robotID-1)
            votecount = w3.call(robotID-1,'getVoteCount') 
            voteOkcount = w3.call(robotID-1,'getVoteOkCount') 
            print('#Rob; Mean; #Block; Balance #Votes, #OkVotes')
            print(nrobs, mean, bn, bal, votecount,voteOkcount)

    if time.time()-updatetimer > 15:
        updatetimer = time.time()

        consensus = w3.call(robotID-1, 'isConverged')
        newRound = w3.call(robotID-1, 'isNewRound')
        ubi = w3.call(robotID-1, 'askForUBI')
        payout = w3.call(robotID-1, 'askForPayout')
        
        w3.transact1(robotID-1, 'updateMean', {"gas":gasLimit})

        if ubi != 0:
            w3.transact1(robotID-1, 'askForUBI', {"gas":gasLimit})
            # print("Asked for UBI") 

        if payout != 0:
            w3.transact1(robotID-1, 'askForPayout', {"gas":gasLimit})
            # print("Asked for Payout") 

        if consensus:
            print('CONSENSUS IS REACHED')
            robot.epuck_leds.set_all_colors("red")


def Estimate():
    """ Control routine to update the local estimate of the robot """
    global estimate, totalWhite, totalBlack
    # Set counters for grid colors
    newValues = gs.getNew()
    # print([newValue for newValue in newValues])

    for value in newValues:
        if value != 0:
            totalWhite += 1
        else:
            totalBlack += 1
    if isbyz:
        estimate = 0
    else:
        estimate = (0.5+totalWhite)/(totalWhite+totalBlack+1)

def reset():
    robot.logprint("reset")

def destroy():
	robot.logprint("destroy")


class GroundSensor(object):
    """ Set up a ground-sensor data acquisition loop on a background thread
    The __sensing() method will be started and it will run in the background
    until the application exits.
    """
    def __init__(self, freq = 20):
        """ Constructor
        :type freq: str
        :param freq: frequency of measurements in Hz (tip: 20Hz)
        """
        self.freq = freq
        self.groundValues = [0 for x in range(3)]
        self.groundCumsum = [0 for x in range(3)]
        self.count = 0


    def sensing(self):
        """ This method runs in the background until program is closed 
        """  

        # Initialize variables
        self.groundValues = robot.epuck_ground.get_readings()

        # Compute cumulative sum
        self.groundCumsum[0] += self.groundValues[0] 
        self.groundCumsum[1] += self.groundValues[1]
        self.groundCumsum[2] += self.groundValues[2]
        self.count += 1


    def getAvg(self):
        """ This method returns the average ground value since last call """

        # Compute average
        try:
            groundAverage =  [round(x/self.count) for x in self.groundCumsum]
        except:
            groundAverage = None

        self.count = 0
        self.groundCumsum = [0 for x in range(3)]
        return groundAverage

    def getNew(self):
        """ This method returns the instant ground value """

        return self.groundValues;


class RandomWalk(object):
    """ Set up a Random-Walk loop on a background thread
    The __walking() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, MAX_SPEED):
        """ Constructor
        :type range: int
        :param enode: Random-Walk speed (tip: 500)
        """
        self.MAX_SPEED = MAX_SPEED                          
        self.__stop = 1
        self.__walk = True
     
        # Random walk parameters
        self.remaining_walk_time = 3
        self.my_lambda = 10 # Parameter for straight movement
        self.turn = 4
        self.possible_directions = ["straight", "cw", "ccw"]
        self.actual_direction = "straight"

        # Obstacle Avoidance parameters
        self.weights_left  = 50*[-10, -10, -5, 0, 0, 5, 10, 10]
        self.weights_right = 50*[-1 * x for x in self.weights_left]

    def walking(self):
        """ This method runs in the background until program is closed """
        # robot.epuck_leds.set_all_colors("black")
        
        # Random Walk
        if (self.remaining_walk_time == 0):
            if self.actual_direction == "straight":
                self.actual_direction = random.choice(self.possible_directions)
                self.remaining_walk_time = math.floor(random.uniform(0, 1) * self.turn)
            else:
                self.remaining_walk_time = math.ceil(random.expovariate(1/(self.my_lambda * 4)))
                self.actual_direction = "straight"
        else:
            self.remaining_walk_time -= 1

        # Find Wheel Speed for Random-Walk
        if (self.actual_direction == "straight"):
            left = right = self.MAX_SPEED/2
        elif (self.actual_direction == "cw"):
            left  = self.MAX_SPEED/2
            right = -self.MAX_SPEED/2
        elif (self.actual_direction == "ccw"):
            left  = -self.MAX_SPEED/2
            right = self.MAX_SPEED/2

        # Obstacle avoidance
        readings = robot.epuck_proximity.get_readings()
        self.ir = [reading.value for reading in readings]
                
        # Find Wheel Speed for Obstacle Avoidance
        for i, reading in enumerate(self.ir):
            if(reading > 0.2 ):
                left  = self.MAX_SPEED/2 + self.weights_left[i] * reading
                right = self.MAX_SPEED/2 + self.weights_right[i] * reading
                # robot.epuck_leds.set_all_colors("red")                

        # Saturate Speeds greater than MAX_SPEED
        if left > self.MAX_SPEED:
            left = self.MAX_SPEED
        elif left < -self.MAX_SPEED:
            left = -self.MAX_SPEED

        if right > self.MAX_SPEED:
            right = self.MAX_SPEED
        elif right < -self.MAX_SPEED:
            right = -self.MAX_SPEED

        if self.__walk:
            # Set wheel speeds
            robot.epuck_wheels.set_speed(right, left)
        else:
            # Set wheel speeds
            robot.epuck_wheels.set_speed(0, 0)
        

    def setWalk(self, state):
        """ This method is called set the random-walk to on without disabling I2C"""
        self.__walk = state

    def setLEDs(self, state):
        """ This method is called set the outer LEDs to an 8-bit state """
        if self.__LEDState != state:
            self.__isLEDset = False
            self.__LEDState = state
        
    def getIr(self):
        """ This method returns the IR readings """
        return self.ir
        

    def setWalk(self, state):
        """ This method is called set the random-walk to on without disabling I2C"""
        self.__walk = state

    def setLEDs(self, state):
        """ This method is called set the outer LEDs to an 8-bit state """
        if self.__LEDState != state:
            self.__isLEDset = False
            self.__LEDState = state
        
    def getIr(self):
        """ This method returns the IR readings """
        return self.ir




# robot.epuck_range_and_bearing.set_data([1,0,0,0])
# robot.epuck_leds.set_all_colors("red")

# def process_rab():
#     global number_robot_sensed 
#     number_robot_sensed = 0
#     for reading_i in robot.epuck_range_and_bearing.get_readings():
#         if reading_i.data[1] == 1:
#             number_robot_sensed += 1
