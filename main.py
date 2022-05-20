# if you are not running this from inside the
# boxenv folder, then conside adding the boxenv
# folder path to the python path by uncommenting
# and filling the lines below

# import sys
# sys.path.append(<location of the boxenv folder>)

from boxdynamics import BoxEnv

env = BoxEnv()

# lets you create an environment by using the GUI
env.world_design()

# when you finish your design, you can save the
# environment configuration with the name "myconf.json"
# by uncommenting the following line
# env.save_conf("myconf.json")

# if you already have a configuration named "myconf.json"
# file with or without environment definition, you can
# use it by uncommenting the followig line
# this would overwrite the previous configuration
# env.load_conf("myconf.json")

# simulation loop
while True:
    # take a random action from the possible ones
    action = env.action_space.sample()

    # compute one simulation step
    state, step_reward, done, info = env.step(action)

    # check for done
    if done:
        break