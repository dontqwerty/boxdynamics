from boxdynamics import BoxEnv

# create enviroment with agent width and height in meters
env = BoxEnv()

env.reset()

width, height = env.get_world_size()

env.create_moving_obstacle((30,10), (3,2), (5,20), angle=2)
env.create_moving_obstacle((10,30), (3,2), (6,3), angle=2.3)
env.create_moving_obstacle((10,40), (5,4), (1,10), angle=2.3)
env.create_moving_obstacle((10,50), (3,6), (5,7), angle=2.3)
env.create_moving_obstacle((30,60), (1,1), (4,-10), angle=2.3)
env.create_moving_obstacle((60,20), (5,5), (0,3), angle=2.3)

env.create_moving_zone((10,20), (3,2), (10,10))
env.create_moving_zone((50,70), (7,5), (10,0))
env.create_moving_zone((60,10), (1,7), (10,0))

env.create_static_obstacle((30,40), (10,3))

env.create_static_zone((60,60), (10,10))
env.create_static_zone((30,30), (5,50))

action = env.action_space.sample()
action[0] = 0
action[1] = 0.0001

counter = 0
total_reward = 0
while True:
    # action[0] = action[0] + (math.pi / 10)
    action = env.action_space.sample()
    state, step_reward, done, info = env.step(action=action)
    if step_reward:
        # print("state: {}, step_reward: {}, done: {}, info: {}".format(state, step_reward, done, info))
        total_reward += step_reward
    if done:
        print("done")
        break