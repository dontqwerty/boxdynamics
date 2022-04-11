from boxdynamics import BoxEnv

env = BoxEnv()

width, height = env.get_world_size()

env.reset()
action = env.action_space.sample()
action[0] = 0
action[1] = 0.0001

env.world_design()

counter = 0
total_reward = 0
while True:
    action = env.action_space.sample()
    state, step_reward, done, info = env.step(action=action)
    if step_reward:
        # print("state: {}, step_reward: {}, done: {}, info: {}".format(state, step_reward, done, info))
        total_reward += step_reward
        # print(total_reward)
    if done:
        break