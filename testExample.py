import gymnasium
import gym_examples
from gym_examples.wrappers import RelativePosition
import random

env = gymnasium.make('gym_examples/logistics-v0', size=10, render_mode='human')
wrapped_env = RelativePosition(env)
initTuble = wrapped_env.reset()
print(initTuble)     # E.g.  [-3  3], {}

stepper = 0
priorStep = 0
distanceChange = initTuble[1]['distance']
targetX = 0
targetY = 0

## Test left and Right
tuple = wrapped_env.step(stepper)
print(tuple)     # E.g.  [-3  4], 0.0, False, {}    

if tuple[2] == True:
    print("We are done")
    exit()    

while True:
    ## Hone in    
    tuple = wrapped_env.step(targetX)
    print(tuple)
    if tuple[2] == True:
        print("We are done")
        exit()

    if tuple[4]['distance'] < distanceChange:
        distanceChange = tuple[4]['distance']
        tuple = wrapped_env.step(targetX)
        print(tuple)
        if tuple[2] == True:
            print("We are done")
            exit()
        distanceChange = tuple[4]['distance']               
    else:
        if targetX == 0:
            targetX = 2
        else:
            targetX = 0
        tuple = wrapped_env.step(targetX)
        print(tuple)
        if tuple[2] == True:
            print("We are done")
            exit()
        distanceChange = tuple[4]['distance']
        

    tuple = wrapped_env.step(targetY)
    print(tuple)

    if tuple[4]['distance'] < distanceChange:
        distanceChange = tuple[4]['distance']
        tuple = wrapped_env.step(targetY)        
        print(tuple)
        if tuple[2] == True:
            print("We are done")
            exit()
        distanceChange = tuple[4]['distance']
    else:
        if targetY == 1:
            targetY = 3
        else:
            targetY = 1
        tuple = wrapped_env.step(targetY)
        print(tuple)
        if tuple[2] == True:
            print("We are done")
            exit()
        distanceChange = tuple[4]['distance']

    if tuple[2] == True:
        print("We are done")
        exit()


    
    
    
