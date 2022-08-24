#!/usr/bin/env python3
# Experimental parameters used in loop and qt_user functions
# Reqs: parameter dictionary is named "params"

import math
import os

# All environment variables
params = dict()
params['environ'] = os.environ

# Generic parameters; include adaptations of environment variables
params['generic'] = dict()
params['generic']['time_limit'] = float(os.environ["TIMELIMIT"]) * 60
params['generic']['arena_size'] = float(os.environ["ARENADIM"])
params['generic']['num_robots'] = int(os.environ["NUMROBOTS"])
params['generic']['seed']       = 358 # None for randomgen
params['generic']['tps'] = eval(os.environ["TPS"])
params['generic']['num_1'] = eval(os.environ["NUM1"])
params['generic']['num_2'] = eval(os.environ["NUM2"])
params['generic']['density'] = eval(os.environ["DENSITY"])
params['generic']['arena_dim'] = eval(os.environ["ARENADIM"])
params['generic']['rab_range'] = eval(os.environ["RABRANGE"])
params['generic']['block_period'] = eval(os.environ["BLOCKPERIOD"])
params['generic']['max_workers'] = eval(os.environ["MAXWORKERS"])


# Parameters for marketplace
params['market'] = dict()
params['market']['radius'] = params['generic']['arena_size'] * math.sqrt(0.05/math.pi)
params['market']['x'] = 0
params['market']['y'] = 0 #params['generic']['arena_size']/2-params['market']['radius']*1.5


# Parameters for cache
params['cache'] = dict()
params['cache']['radius']  = params['generic']['arena_size'] * math.sqrt(0.10/math.pi)
params['cache']['x'] = params['market']['x']
params['cache']['y'] = params['market']['y']

params['patches'] = dict()
params['patches']['distribution'] = 'uniform' 
# params['patches']['distribution'] = 'patchy'
# params['patches']['hotspots']      = [{'x_mu': 0.25 * params['generic']['arena_size'], 
# 									     'y_mu': 0.25 * params['generic']['arena_size'], 
# 									     'x_sg': 0.15 * params['generic']['arena_size'], 
# 									     'y_sg': 0.15 * params['generic']['arena_size']}]

params['patches']['respawn']  = False
params['patches']['known']    = True
params['patches']['radius']   = 0.12
params['patches']['qtty_min'] = 15
params['patches']['qtty_max'] = 15
# params['patches']['dist_min'] = 1 * params['cache']['radius'] 
# params['patches']['dist_max'] = 0.5 * params['generic']['arena_size'] 

params['patches']['dist_min'] = 2.00 * params['cache']['radius'] 
params['patches']['dist_max'] = 2.01 * params['cache']['radius']

params['patches']['qualities']   = {'red', 'green' , 'blue', 'yellow'}
params['patches']['forage_rate'] = {'red': 5, 'green':  5, 'blue': 5, 'yellow': 5}
params['patches']['regen_rate']  = {'red': 30, 'green':  30, 'blue': 30, 'yellow': 30}
params['patches']['utility']     = {'red': 2, 'green':  4, 'blue': 6, 'yellow': 8}

params['patches']['dec_returns_func']   = 'linear' # constant, linear or logarithmic decreasing returns
params['patches']['dec_returns_thresh'] = 15       # qqty of resource before dec returns starts
params['patches']['dec_returns_mult']   = 8        # multiplier of the original forage rate for the last resource being foraged

# params['patches']['area_percent'] = 0.005 * (10/generic_params['num_robots'])
# params['patches']['radius']    = params['generic']['arena_size']  * math.sqrt(resource_params['area_percent']/math.pi) 

# params['patches']['radius']    = params['generic']['arena_size']  * math.sqrt(resource_params['area_percent']/math.pi) 
# params['patches']['abundancy']    = 0.03
# params['patches']['frequency'] = {'red': 0.25, 'green': 0.25 , 'blue': 0.25, 'yellow': 0.25}


params['patches']['counts'] = {'red': 1, 'green': 1 , 'blue': 1, 'yellow': 1}
params['patches']['radii']  = {'red':    params['patches']['radius'], 
							   'green':  params['patches']['radius'] , 
							   'blue':   params['patches']['radius'], 
							   'yellow': params['patches']['radius']}

# Parameters for the economy
# economy_params = dict()
# economy_params['fuel_cost'] = 0.1 # eth per second of exploration

# Initialize the files which store QT_draw information 
params['files'] = dict()
params['files']['patches'] = 'loop_functions/patches.txt'
params['files']['robots']  = 'loop_functions/robots.txt'
params['files']['position']  = 'loop_functions/position.txt'
params['files']['rays']    = 'loop_functions/rays.txt'
