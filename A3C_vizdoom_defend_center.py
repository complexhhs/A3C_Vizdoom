import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import time
import threading
import cv2

sys.path.append('./vizdoom')

from vizdoom import *
from parameter import *
from helper import *

class A3Cagent():
	def __init__(self,scope):
		# Loading parameters
		
		with tf.variable_scope(scope):
			self.inputs = tf.placeholder(shape=[None,height_screen,width_screen,1],dtype=tf.float32)
			self.imageIn = tf.reshape(self.inputs,shape=[-1,height_screen,width_screen,1])
			# Convolution layers
			self.h1_layer = slim.conv2d(activation_fn = tf.nn.relu, inputs = self.inputs, num_outputs = 32\
				, kernel_size=[8,8], stride=2, padding = 'VALID')
			self.h2_layer = slim.conv2d(activation_fn = tf.nn.relu, inputs = self.h1_layer, num_outputs = 64\
				, kernel_size=[4,4], stride=2, padding = 'VALID')
			self.h3_layer = slim.conv2d(activation_fn = tf.nn.relu, inputs = self.h2_layer, num_outputs = 128\
				, kernel_size=[4,4], stride=2, padding = 'VALID')
			self.ft_layer = slim.fully_connected(slim.flatten(self.h3_layer),512,activation_fn = tf.nn.elu)
			
			# Recurrence network
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(512,state_is_tuple = True)
			c_init = np.zeros((1,lstm_cell.state_size.c), np.float32)
			h_init = np.zeros((1,lstm_cell.state_size.h), np.float32)
			self.state_init = [c_init, h_init]
			c_in = tf.placeholder(dtype=tf.float32, shape=[1,lstm_cell.state_size.c])
			h_in = tf.placeholder(dtype=tf.float32, shape=[1,lstm_cell.state_size.h])
			self.state_in = (c_in, h_in)
			rnn_in = tf.expand_dims(self.ft_layer,[0])
			step_size = tf.shape(self.imageIn)[:1]
			state_in = tf.contrib.rnn.LSTMStateTuple(c_in,h_in)
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,rnn_in,initial_state=state_in,sequence_length\
				=step_size, time_major = False) 
			lstm_c, lstm_h = lstm_state
			self.state_out = (lstm_c[:1,:], lstm_h[:1,:])
			rnn_out = tf.reshape(lstm_outputs,[-1,512])
			
			# action and critic estimation
			self.mid = slim.fully_connected(rnn_out,128,activation_fn=tf.nn.relu)
			self.actor = slim.fully_connected(self.mid, action_size, activation_fn = tf.nn.softmax)
			self.critic = slim.fully_connected(self.mid, 1, activation_fn = None)
			
			# loss function setting > only workers update gradient and loss function
			if scope != 'global':				
				self.actions = tf.placeholder(shape=[None], dtype = tf.int32)
				self.action_onehot = tf.one_hot(self.actions,action_size,dtype = tf.float32)
				self.target = tf.placeholder(shape=[None], dtype = tf.float32)
				self.advantages = tf.placeholder(shape=[None], dtype = tf.float32)
				
				self.action_prob = tf.reduce_sum(self.action_onehot*self.actor,axis=1)
				
				self.entropy = -tf.reduce_sum(tf.log(self.actor)*self.actor,axis=-1)
				self.cross_entropy = -tf.reduce_sum(tf.log(self.action_prob)*self.advantages)
				
				self.critic_loss = tf.reduce_sum(tf.square(tf.reshape(self.critic,[-1])-self.target))
				self.loss = self.cross_entropy - 0.008*self.entropy + 0.1*self.critic_loss
				
				# gradient from local network 
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
				self.gradients = tf.gradients(self.loss,local_vars) 
				grads,_ = tf.clip_by_global_norm(self.gradients,gradient_clip)
				
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global')
				self.train=tf.train.AdamOptimizer(learning_rate = learn_speed).apply_gradients(zip(grads,global_vars))
				
class Agent():
	def __init__(self,game,name,global_episode):
		# threading.Thread.__init__(self)
		self.game = game
		self.name = "agent_"+str(name)
		self.global_episode = global_episode
		self.increment = self.global_episode.assign_add(1)
		self.summary_writer = tf.summary.FileWriter(summary_path+"/train_"+str(name))
		
		# self setting 
		self.actions = np.identity(action_size,dtype=bool).tolist()
		# act_list = [0,1,2]
		# self.actions = tf.one_hot(act_list,action_size)
		self.local_A3C = A3Cagent(self.name)
		self.update_local_ops = tf.group(update_target_graph('global',self.name))
		
		# game environment setting
		# self.game.set_doom_scenario_path(scenario_path)
		self.game.load_config(scenario_path)
		self.game.set_screen_resolution(ScreenResolution.RES_640X480)
		self.game.set_screen_format(ScreenFormat.RGB24)
		self.game.set_render_hud(True)
		self.game.set_render_crosshair(False)
		self.game.set_render_weapon(True)
		self.game.set_render_effects_sprites(True)
		self.game.set_render_decals(False)
		self.game.set_render_particles(False)
		self.game.set_labels_buffer_enabled(True)
		self.game.add_available_button(Button.TURN_LEFT)
		self.game.add_available_button(Button.TURN_RIGHT)
		self.game.add_available_button(Button.ATTACK)
		self.game.add_available_game_variable(GameVariable.USER1)
		self.game.set_episode_start_time(episode_start)
		
		# train setting
		self.game.set_window_visible(window_visible)
		self.game.set_sound_enabled(sound_enable)
		self.game.set_living_reward(living_reward)
		self.game.set_mode(Mode.PLAYER)		
		self.game.init()
		self.env = self.game	
		
	def work(self,max_episode,sess,coord,saver):	
		episode = 0
		episode_count = sess.run(self.global_episode)
		start_t = time.time()
		skiprate = 4
		print(str(self.name)+" begins training")
		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():
				sess.run(self.update_local_ops)
				episode_frames = []
				
				buffer = []
				self.episode_rewards=[]
				self.episode_episode_health=[]
				self.episode_health=[]
				self.episode_kills = []
				self.episode_length=[]
				self.avg_p_max = 0
			
				episode_reward = 0
				episode_kills = 0
				episode += 1
				step = 0
				episode_step_count = 0
				done=False
				
				last_total_health = 100
				last_total_ammo2 = 26
				
				self.env.new_episode()
				episode_st = time.time()	
				while not self.env.is_episode_finished():
					screen = self.env.get_state().screen_buffer
					screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
					episode_frames.append(screen)
					observation = pre_processing(screen)
					rnn_state = self.local_A3C.state_init
					self.batch_rnn_state = rnn_state
					
					action_prob, val_old, rnn_state = sess.run([self.local_A3C.actor,self.local_A3C.critic,self.local_A3C.state_out],\
						feed_dict={self.local_A3C.inputs:[observation],self.local_A3C.state_in[0]:rnn_state[0],\
						self.local_A3C.state_in[1]:rnn_state[1]})
					val_old = val_old[0,0]
					action_prob = action_prob[0]
					action_idx = self.get_action(action_prob)
					self.avg_p_max += np.amax(action_prob)
					
					# reward calculation
					kill_reward = self.env.make_action(self.actions[action_idx[0]],skiprate)/100.
					cur_ammo2=self.env.get_game_variable(GameVariable.AMMO2)
					cur_health=self.env.get_game_variable(GameVariable.HEALTH)
					if cur_ammo2 < last_total_ammo2:
						ammo_reward = -0.05
					else:
						ammo_reward = 0
					if cur_health < last_total_health:
						health_reward = -0.1
					else:
						health_reward = 0
					last_total_ammo2 = cur_ammo2
					last_total_health = cur_health
					reward = kill_reward + health_reward + ammo_reward
					
					episode_reward += reward
					episode_kills += kill_reward

					done = self.env.is_episode_finished()
					if not done:
						new_screen = self.env.get_state().screen_buffer
						new_observation = pre_processing(new_screen)
					else:
						new_observation = observation
					
					buffer.append([observation,action_idx,val_old,reward])
					observation = new_observation
					
					step += 1
					episode_step_count += 1
					
					if step >= step_period or done == False:
						val_new = sess.run(self.local_A3C.critic, feed_dict={self.local_A3C.inputs:[observation],self.local_A3C.state_in[0]:rnn_state[0],
							 self.local_A3C.state_in[1]:rnn_state[1]})[0,0]
						self.train_model(sess,buffer,val_new,done)
						sess.run(self.update_local_ops)
						buffer=[]
						step = 0
						
					if done or last_total_ammo2 <=0:
						self.episode_health.append(self.env.get_game_variable(GameVariable.HEALTH))
						self.episode_rewards.append(episode_reward)
						self.episode_episode_health.append(last_total_health)
						self.episode_length.append(episode_step_count)
						self.episode_kills.append(episode_kills-2)
						
						summary = tf.Summary()
						summary.value.add(tag='Total_reward', simple_value = episode_reward)
						summary.value.add(tag='Total_health', simple_value = last_total_health)
						summary.value.add(tag='Total_kills', simple_value = episode_kills+1)
						summary.value.add(tag = 'Max_probability', simple_value = self.avg_p_max/float(episode_step_count))
						self.summary_writer.add_summary(summary,episode)
						self.summary_writer.flush()
						print('%s, health: %d, episode: %d, reward: %3.2f, killed: %d, ammo2_left: %d, max_prob: %3.3f time_cost: %3.3f' \
							%(self.name, last_total_health, episode, episode_reward, episode_kills-2, last_total_ammo2, self.avg_p_max/float(episode_step_count),time.time()-episode_st))
						break							
					
				# At the end of episode: update, update the network
				if len(buffer) != 0:
					self.train_model(sess,buffer,0.,True)
						
				if episode % 100 == 0 and self.name =='agent_0':
					saver.save(sess,model_path+'/model-'+str(episode)+'.cptk')
					print("Episode count %d, saved Model, time costs %5.3f"%(episode_count, time.time()-start_t))
					start_t = time.time()
					print("##########################################################")
					print(".......................Model saved........................")
					time_per_step=0.12
					images = np.array(episode_frames)
					make_gif(images,frame_path+'/image-'+str(episode)+'.gif',duration=len(images)*time_per_step,\
						true_image=True,salience=False)
				
				if reward >= 22.:
					time_per_step=0.15
					images = np.array(episode_frames)
					make_gif(images,frame_path+'/image-'+str(episode)+'.gif',duration=len(images)*time_per_step,\
						true_image=True,salience=False)
					print('I am satisfying this shot reward!')
					coord.request_stop()

				if self.name == 'agent_0':
					sess.run(self.increment)
				episode_count += 1
				
				if episode_count == 120000:  # thread to stop
					print("Stop training name:{}".format(self.name))
					coord.request_stop()				
							
	def discounted_reward(self,rewards,new_critic,done):
		discounted_reward = np.zeros_like(rewards)
		temp = 0
		
		if done == False:
			temp = new_critic
		
		for t in reversed(range(0,len(rewards))):
			temp = discount_factor*temp+rewards[t]
			discounted_reward[t] = temp
		return np.array(discounted_reward)
	
	def train_model(self,sess,buffer,new_critic,done):
		buffer = np.array(buffer)
		observations = buffer[:,0]
		actions = buffer[:,1]
		critics = buffer[:,2]
		rewards = buffer[:,3]
		
		discounted_rewards = self.discounted_reward(rewards,new_critic,done)
		advantages = discounted_rewards - critics
			
		feed_dict = {self.local_A3C.target:discounted_rewards,
			self.local_A3C.inputs:np.stack(observations),
			self.local_A3C.actions:actions,
			self.local_A3C.advantages:advantages,
			self.local_A3C.state_in[0]:self.batch_rnn_state[0],
			self.local_A3C.state_in[1]:self.batch_rnn_state[1]}
		sess.run(self.local_A3C.train,feed_dict = feed_dict)
		
	def get_action(self,prob):
		action_list = np.random.choice(action_size,1,p=prob)
		return action_list
		
	def play_game(self,sess,episode_num):
		if not isinstance(sess,tf.Session):
			raise TypeError('saver should be tf.train.Saver')
			
		for i in range(episode_num):
			self.env.new_episode()
			screen = self.env.get_state()
			screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
			observation = pre_processing(screen.screen_buffer)
			episode_rewards = 0
			last_total_shaping_reward = 0
			step = 0
			start_time = time.time()
			episode_frames=[]
			while not self.env.is_episode_finished():
				step += 1
				screen = self.env.get_state()
				episode_frames.append(screen)
				time_per_step = 0.15
				images = np.array(episode_frames)
				make_gif(images,frame_path+'/image'+str(i)+'.gif',duration=len(images)*time_per_step,\
					true_image=True,salience=False)
				observation = pre_processing(screen.screen_buffer)
				a_dist, value, rnn_state = sess.run([self.local_A3C.actor,self.local_A3C.critic,self.local_A3C.state_out],\
						feed_dict={self.local_A3C.inputs:[self.observation],self.local_A3C.state_in[0]:rnn_state[0],\
						self.local_A3C.state_in[1]:rnn_state[1]})
				self.action = self.get_action(a_dist)
				reward = self.env.make_action(self.actions[self.action])
				episode_rewards += reward
				
				print('Current step :#{}'.format(step))
				print('Current action: ',self.act_name[self.action])
				print('Current health: ',self.env.get_game_variable(GameVariable.HEALTH))
				print('Current ammo: {0}'.format(self.env.get_game_variable(GameVariable.AMMO2)))
				print('Current reward: {0}'.format(reward))
				if self.env.get_game_variable(GameVariable.AMMO2) <= 0:
					break
			print('------------------------------')
			print('Runt out of AMMO')
			print('Env episode: {}, Total Reward: {}',format(i,episode_rewards))
			print('Time lapsed: {0}'.format(time.time()-start_time))
			time.sleep(3)
		
def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,to_scope)
	
	op_holder=[]
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder
	
# def normalized_columns_initializer(std=1.0):
    # def _initializer(shape, dtype=None, partition_info=None):
        # out = np.random.randn(*shape).astype(np.float32)
        # out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        # return tf.constant(out)
    # return _initializer
		
def pre_processing(image):
	image = image[75:-75,:]	
	image = cv2.resize(image, (height_screen, width_screen), interpolation=cv2.INTER_LINEAR)
	image = rgb2gray(image)
	return image
	
def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray.reshape((height_screen, width_screen, 1))