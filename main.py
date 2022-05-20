# if you are not running this from inside the
# boxenv folder, then conside adding the boxenv
# folder path to the python path by uncommenting
# and filling the lines below

# import sys
# sys.path.append(<location of the boxenv folder>)

from boxdynamics import BoxEnv

# create an environment instance
# if there is no "config.json" file
# in the current working directory,
# then the deafult configuration will
# be used
env = BoxEnv()

# lets you create an environment by using the GUI
env.world_design()

# SAVE CONFIGURATION
# save the environment configuration with the name
# "myconf.json" by uncommenting the following line
# this also includes the environment design
# env.save_conf("myconf.json")

# LOAD CONFIGURATION
# if you already have a configuration named "myconf.json"
# file you can use it by uncommenting the followig line
# this overwrites every previous configuration
# env.load_conf("myconf.json")

# SAVE DESIGN
# if you just want to save the design named "mydesign.json"
# to utilize it later on, uncomment the following line
# env.save_design("mydesign.json")

# LOAD DESIGN
# if you have a design named "mydesign.json"
# you can load it by uncommenting the following line
# env.load_design("mydesign.json")

# note that you could also load and save designs
# from from the GUI

# simulation loop
while True:
    # take a random action from the possible ones
    action = env.action_space.sample()

    # compute one simulation step
    state, step_reward, done, info = env.step(action)

    # check for done
    if done:
        break