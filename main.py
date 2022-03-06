from box_dynamics import BoxEnv
from time import sleep

# create enviroment with agent width and height in meters
env = BoxEnv()

env.reset()

width, height = env.get_world_size()

for i in range(10):
    env.create_static_zone(pos=(width*0.1*i, height*0.1*i), size=(3,2))

for i in range(3):
    env.create_static_obstacle(pos=(width*0.1*i, height*0.3), size=(10,10))
    # env.create_static_obstacle(pos=(width*(1 - 0.1*i), height*0.7), size=(3,3))

env.create_static_zone(pos=(10,5), size=(10,10))
env.create_static_zone(pos=(10,10), size=(10,10))
env.create_static_zone(pos=(10,15), size=(10,10))
env.create_static_zone(pos=(10,20), size=(10,10))
env.create_static_zone(pos=(10,25), size=(10,10))
env.create_static_zone(pos=(10,30), size=(10,10))
env.create_static_zone(pos=(10,35), size=(10,10))

env.create_kinematic_obstacle(pos=(10,20), size=(10,10), velocity=(10,10))
env.create_kinematic_obstacle(pos=(20,20), size=(10,10), velocity=(10,0))
env.create_kinematic_obstacle(pos=(30,20), size=(10,10), velocity=(0,10))
env.create_kinematic_zone(pos=(40,20), size=(10,10), velocity=(10,10))
env.create_kinematic_zone(pos=(50,20), size=(10,10), velocity=(10,0))
env.create_kinematic_zone(pos=(60,20), size=(10,10), velocity=(0,10))


action = env.action_space.sample()
action[0] = 0
action[1] = 1

import math

counter = 0
while True:
    # action[0] = action[0] + (math.pi / 10)
    action = env.action_space.sample()
    state, step_reward, done, info = env.step(action=action)
    # print("state: {}, step_reward: {}, done: {}, info: {}".format(state, step_reward, done, info))
    # if counter > 10:
    #     action = [0,0]
        # input()
    if not env.render():
        break
    counter += 1