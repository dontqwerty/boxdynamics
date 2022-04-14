from boxdynamics import BoxEnv

env = BoxEnv()

env.world_design()
env.save_conf()
# env.load_conf()

total_reward = 0
while True:
    action = env.action_space.sample()
    state, step_reward, done, info = env.step(action=action)
    if step_reward:
        # print("state: {}, step_reward: {}, done: {}, info: {}".format(state, step_reward, done, info))
        total_reward += step_reward
        # print(total_reward)
    if done:
        env.reset()