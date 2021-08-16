import numpy as np
import time
import os
#from chainer import cuda

#import cupy as cp

#backend
#be = "gpu"
#device = 0


be = "cpu"

class SimpleDQN(object):
	
	
	# constructor
	def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate,gamma,decay_rate,greedy_e_epsilon,random_seed):
		# store hyper-params
		self._A = num_actions
		self._D = input_size
		self._H = hidden_layer_size
		self._learning_rate = learning_rate
		self._decay_rate = decay_rate
		self._gamma = gamma
		
		# some temp variables
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[]

		# variables governing exploration
		self._exploration = True # should be set to false when evaluating
		self._explore_eps = greedy_e_epsilon
		
		#create model
		self.init_model(random_seed)
		
		self.log_dir = 'results'
		self.env_id = 'NovelGridworld-v0'
		os.makedirs(self.log_dir, exist_ok = True)

	def init_model(self,random_seed):
		# create model
		#with cp.cuda.Device(0):
		self._model = {}
		np.random.seed(random_seed)
	   
		# weights from input to hidden layer   
		self._model['W1'] = np.random.randn(self._D,self._H) / np.sqrt(self._D) # "Xavier" initialization
	   
		# weights from hidden to output (action) layer
		self._model['W2'] = np.random.randn(self._H,self._A) / np.sqrt(self._H)
			
		# print("model is: ", self._model)
		# time.sleep(5)		
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory

	
	# softmax function
	def softmax(self,x):
		probs = np.exp(x - np.max(x, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		return probs
		
	  
	def discount_rewards(self,r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * self._gamma + r[t]
			discounted_r[t] = float(running_add)
    
		return discounted_r
	
	# feed input to network and get result
	def policy_forward(self,x):
		if(len(x.shape)==1):
			x = x[np.newaxis,...]

		h = x.dot(self._model['W1'])
		
		if np.isnan(np.sum(self._model['W1'])):
			print("W1 sum is nan")
			time.sleep(5)
		if np.isnan(np.sum(self._model['W2'])):
			print("W2 sum is nan")
		
		if np.isnan(np.sum(h)):
			print("nan")
			
			h[np.isnan(h)] = np.random.random_sample()
			h[np.isinf(h)] = np.random.random_sample()
			

		if np.isnan(np.sum(h)):
			print("Still nan!")
		
		
		h[h<0] = 0 # ReLU nonlinearity
		logp = h.dot(self._model['W2'])

		p = self.softmax(logp)
  
		return p, h # return probability of taking actions, and hidden state
		
	
	def policy_backward(self,eph, epdlogp):
		""" backward pass. (eph is array of intermediate hidden states) """
		dW2 = eph.T.dot(epdlogp)  
		dh = epdlogp.dot(self._model['W2'].T)
		dh[eph <= 0] = 0 # backpro prelu
  
		t = time.time()
  
		if(be == "gpu"):
		  self._dh_gpu = cuda.to_gpu(dh, device=0)
		  self._epx_gpu = cuda.to_gpu(self._epx.T, device=0)
		  self._dW1 = cuda.to_cpu(self._epx_gpu.dot(self._dh_gpu) )
		else:
		  self._dW1 = self._epx.T.dot(dh) 
    

		#print((time.time()-t0)*1000, ' ms, @final bprop')

		return {'W1':self._dW1, 'W2':dW2}
	
	def set_explore_epsilon(self,e):
		self._explore_eps = e
	
	# input: current state/observation
	# output: action index
	def process_step(self, x, exploring):

		# feed input through network and get output action distribution and hidden layer
		aprob, h = self.policy_forward(x)
		
		#print(aprob)
		
		# if exploring
		if exploring == True:
			
			# greedy-e exploration
			rand_e = np.random.uniform()
			#print(rand_e)
			if rand_e < self._explore_eps:
				# set all actions to be equal probability
				aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]
				#print("!")
		
		
		if np.isnan(np.sum(aprob)):
			print(aprob)
			aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]
			print(aprob)
			#input()
		
		aprob_cum = np.cumsum(aprob)
		u = np.random.uniform()
		a = np.where(u <= aprob_cum)[0][0]	

		# record various intermediates (needed later for backprop)
		t = time.time()
		self._xs.append(x) # observation
		self._hs.append(h)

		#softmax loss gradient
		dlogsoftmax = aprob.copy()
		dlogsoftmax[0,a] -= 1 #-discounted reward 
		self._dlogps.append(dlogsoftmax)
		
		t  = time.time()

		return a
		
	# after process_step, this function needs to be called to set the reward
	def give_reward(self,reward):
		
		# store the reward in the list of rewards
		self._drs.append(reward)
		
	# reset to be used when evaluating
	def reset(self):
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[] # reset 
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory

		
	# this function should be called when an episode (i.e., a game) has finished
	def finish_episode(self):
		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		
		# this needs to be stored to be used by policy_backward
		# self._xs is a list of vectors of size input dim and the number of vectors is equal to the number of time steps in the episode
		self._epx = np.vstack(self._xs)
		
		
		#for i in range(0,len(self._hs)):
		#	print(self._hs[i])
		
		# len(self._hs) = # time steps
		# stores hidden state activations
		eph = np.vstack(self._hs)
		
		#for i in range(0,len(self._dlogps)):
		#	print(self._dlogps[i])
		
		# self._dlogps stores a history of the probabilities over actions selected by the agent
		epdlogp = np.vstack(self._dlogps)
		
		# self._drs is the history of rewards
		#for i in range(0,len(self._drs)):
		#	print(self._drs[i])
		epr = np.vstack(self._drs)
		
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[] # reset array memory

		# compute the discounted reward backwards through time
		discounted_epr = (self.discount_rewards(epr))
		#for i in range(0,len(discounted_epr)):
		#	print(str(discounted_epr[i]) + "\t"+str(epr[i]))
		
		
		# #print(discounted_epr)
		# discounted_epr_mean = np.mean(discounted_epr)
		# #print(discounted_epr_mean)
		
		# # standardize the rewards to be unit normal (helps control the gradient estimator variance)
		
		# #discounted_epr -= np.mean(discounted_epr)
		# discounted_epr = np.subtract(discounted_epr,discounted_epr_mean)
		
		
		# discounted_epr /= np.std(discounted_epr)+0.01
		
		epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
		
		start_time = time.time()
		grad = self.policy_backward(eph, epdlogp)
		#print("--- %s seconds for policy backward ---" % (time.time() - start_time))
		
		for k in self._model: self._grad_buffer[k] += grad[k] # accumulate grad over batch

	# called to update model parameters, generally every N episodes/games for some N
	def update_parameters(self):
		for k,v in self._model.items():
			g = self._grad_buffer[k] # gradient
			self._rmsprop_cache[k] = self._decay_rate * self._rmsprop_cache[k] + (1 - self._decay_rate) * g**2
			self._model[k] -= self._learning_rate * g / (np.sqrt(self._rmsprop_cache[k]) + 1e-5)
			self._grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

	def save_model(self, curriculum_no, beam_no, env_no):

		experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
		path_to_save = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
		np.savez(path_to_save, layer1 = self._model['W1'], layer2 = self._model['W2'])
		print("saved to: ", path_to_save)

	def load_model(self, curriculum_no, beam_no, env_no):

		experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
		path_to_load = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
		data = np.load(path_to_load)
		self._model['W1'] = data['layer1']
		self._model['W2'] = data['layer2']
