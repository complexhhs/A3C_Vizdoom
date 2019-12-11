# main program part
import random
import time
import tensorflow as tf
import os
import shutil
import sys
import threading

sys.path.append('./vizdoom')

from vizdoom import *
from parameter import *
from A3C_vizdoom_defend_center import *
	
def main_train(tf_configs=None):
	if not os.path.exists(model_path):
		os.mkdir(model_path)
		
	if not os.path.exists(summary_path):
		os.mkdir(summary_path)

	if not os.path.exists(frame_path):
		os.mkdir(frame_path)

	start_time = time.time()
	tf.reset_default_graph()
	with tf.device("/cpu:0"):
		global_episodes = tf.Variable(0, dtype = tf.int32, name='global_episodes', trainable = False)
		global_network = A3Cagent('global')
		num_agents = 4
		agents = []
		# Create agents
		for cpu_num in range(num_agents):
			agents.append(Agent(DoomGame(),cpu_num,global_episodes))
		saver = tf.train.Saver(max_to_keep=save_term)
	
	with tf.Session(config=tf_configs) as sess:
		coord = tf.train.Coordinator()
		# sess.run(tf.global_variables_initializer())
		saver.restore(sess,tf.train.latest_checkpoint('./save_model_defend_center/'))
		agent_threads = []
		for agent in agents:
			agent_train = lambda: agent.work(max_episode,sess,coord,saver)
			t=threading.Thread(target = (agent_train))
			t.start()
			time.sleep(0.5)
			agent_threads.append(t)
		coord.join(agent_threads)
		print("training_ends. computation time is:{}".format(time.time()-start_time))

def main_play(tf_configs=None):
	tf.reset_default_graph()
	with tf.device("/cpu:0"):
		global_episodes = tf.Variable(0,dtype = tf.int32,name='global_episodes',trainable=False)
		global_network = A3Cagent('global')
		saver = tf.train.Saver()
	with tf.Session(config=tf_configs) as sess:
		coord = tf.train.Coordinator()
		print('Loading play data.......')
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.latest_checkpoint(model_path)
		saver.restore(sess,ckpt)
		print('Successfully loaded!')
		Agent.play_game(sess,10)
		
if __name__=='__main__':
	# train = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	if load_model == False:
		main_train(tf_configs=config)
	else:
		main_play(tf_configs=config)