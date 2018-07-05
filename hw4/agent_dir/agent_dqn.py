from agent_dir.agent import Agent
import tensorflow.contrib.layers as ly 
import os 
import sys 
import tensorflow as tf 
import numpy as np 
import random
#import pandas as pd 
#import matplotlib.pyplot as plt 
from collections import deque



REPLAY_SIZE =  100000
NUM_EPISODES = 50000
MAX_NUM_STEPS = 10000
UPDATE_TIME = 3000
OBSERVE = 50000 # timesteps to observe before training
EXPLORE = 1000000 # frames over which epsilon decreases
BATCH_SIZE = 32

INITIAL_EPSILON = .9
FINAL_EPSILON = .1
GAMMA = .99

#np.random.seed()
# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess,state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output,feed_dict= { self.input_state : state })

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        self.arg = args
        self.env = env 
        #self.replay_size = 100000
        #self.update_time = 3000
        self.train_skip = 3
        #self.replayMemory = deque()
        #self.timestep = 0
        #self.epsion = 1 
        #self.actions = 4 
        self.replay_memory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = 1
        self.actions = 4
        #self.StateInput = tf.placeholder(tf.float32,shap=[None,84,84,4])

        self.reward_memory = deque(maxlen=30)
        self.reward_history = []

        print(self.arg.Dueling)
        if self.arg.Dueling : 
            a = self.createQNetwork()
            #print(a)
            for i in range(len(a)):
                print(a[i])
            self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2,self.W_fc3,self.b_fc3 = self.createQNetwork()


            # init Target Q Network
            self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T,self.W_fc3T,self.b_fc3T = self.createQNetwork()


            self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2),self.W_fc3T.assign(self.W_fc3),self.b_fc3T.assign(self.b_fc3)]

        else : 
            self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

        # init Target Q Network
            self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T= self.createQNetwork()

            self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.cost_history = []
        #checkpoint = tf.train.get_checkpoint_state("saved_networks")
        #if checkpoint and checkpoint.model_checkpoint_path:
        #        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        #        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        #else:
        #        print("Could not find old network weights")

        tensorboard_dir = 'DQN/'   
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        if args.test_dqn:
            self.saver.restore(self.sess,'network-dqn-3000000')
            #you can load your model here
            print('loading trained model')


        def init_game_setting(self):
            random.seed(23)





        '''    

        Inp_q , self.q_prediction , self.q_loss ,self.q_train_op ,self.merge ,self.q_para= self.build_model('q',summary_dir = tensorboard_dir )
        self.q_x_pl = Inp_q[0]
        self.q_y_pl = Inp_q[1]
        self.q_action_pl = Inp_q[2]

        self.writer = tf.summary.FileWriter(tensorboard_dir)

        if args.test_dqn:
            self.sess = tf.Session()

            self.sess.run(tf.global_variables_initializer())


            self.epsilons = np.linspace(1,0.025,10000)

            self.policy = self.make_epsilon_greedy_police(len(VALID_ACTIONS))

            self.saver = tf.train.Saver()
            self.saver.restore(self.sess,'deep_q/6')
            #you can load your model here
            print('loading trained model')


        Inp_target , self.target_prediction ,self.target_loss ,self.target_train_op  , self.target_para = self.build_model('target_q') 

        self.target_q_x_pl = Inp_target[0]
        self.target_q_y_pl = Inp_target[1]
        self.target_q_action_pl = Inp_target[2]
        '''


        ##################
        # YOUR CODE HERE #
        ##################
    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)

    def createQNetwork(self):
        # network weights
        if self.arg.Dueling :
            W_conv1 = self.weight_variable([8,8,4,32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4,4,32,64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3,3,64,64])
            b_conv3 = self.bias_variable([64])

            W_fc1 = self.weight_variable([3136,512])
            b_fc1 = self.bias_variable([512])

            W_fc2 = self.weight_variable([512,self.actions])
            b_fc2 = self.bias_variable([self.actions])

            W_fc3 = self.weight_variable([512,1])
            b_fc3 = self.bias_variable([1])           

            stateInput = tf.placeholder("float",[None,84,84,4])
            # hidden layers
            h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
            #h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

            h_conv3_shape = h_conv3.get_shape().as_list()

            print("dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])

            h_conv3_flat = tf.reshape(h_conv3,[-1,3136])

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1)+b_fc1)

            V = tf.matmul(h_fc1,W_fc2) + b_fc2

            A = tf.matmul(h_fc1,W_fc3) + b_fc3

            QValue  = A + (V - tf.reduce_mean(V,axis=1,keep_dims=True ))

            return  stateInput , QValue , W_conv1 ,\
                    b_conv1 , W_conv2 , b_conv2 , W_conv3 ,\
                    b_conv3 , W_fc1, b_fc1, W_fc2 , b_fc2,\
                    W_fc3, b_fc3

        else:    
            W_conv1 = self.weight_variable([8,8,4,32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4,4,32,64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3,3,64,64])
            b_conv3 = self.bias_variable([64])

            W_fc1 = self.weight_variable([3136,512])
            b_fc1 = self.bias_variable([512])

            W_fc2 = self.weight_variable([512,self.actions])
            b_fc2 = self.bias_variable([self.actions])

            # input layer

            stateInput = tf.placeholder("float",[None,84,84,4])

            # hidden layers
            h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
            #h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
            h_conv3_shape = h_conv3.get_shape().as_list()
            print("dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])
            h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

            # Q Value layer
            QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

            return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2


    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def predict(self,sess,observation):
        return sess.run(self.q_prediction,feed_dict={self.q_x_pl:observation})

    def target_predict(self,sess,observation):
        return sess.run(self.target_prediction,feed_dict={self.target_q_x_pl:observation})


    def train(self):
        for k in range(NUM_EPISODES):
            ########## initial value 
            state = self.env.reset()
            step_count = 0 
            total_reward = 0

            for _ in range(MAX_NUM_STEPS):
                action = self.make_action(state,test=False)
                #print(action)
                next_state,reward,done,_ = self.env.step(action)
                self.percieve(state,action,reward,next_state,done)

                state = next_state 
                step_count +=1 
                total_reward += reward

                if done : 
                    self.reward_memory.append(total_reward)
                    if len(self.reward_memory) == 30 : 
                        self.reward_history.append(np.mean(np.array(self.reward_memory)))
                        #self.reward_history.append(sum(self.reward_memory))
                    break
            '''
            if total_reward > 40 : 
                plt.plot(self.reward_history)
                plt.xlabel('Episode')
                plt.ylabel('Mean Reward ( Sum reward/30)')
                plt.title('Learning Curve')
                plt.savefig('DQN.png')
                plt.show()
            '''
            print(total_reward)

    def train_Q_network(self):

        
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory ,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        #print(self.arg.DDQN)
        # Step 2: calculate y 
        if self.arg.DDQN:
            y_batch = []
            QValueT_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
            QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})
            for i in range(0,BATCH_SIZE):
                terminal = minibatch[i][4]

                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    estimate = np.argmax(QValue_batch,1)[i]
                    y_batch.append(reward_batch[i]+GAMMA*QValueT_batch[i][estimate])
           
        else:
            y_batch = []
            QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
            for i in range(0,BATCH_SIZE):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        ### for double dqn :  np.argmax(qValue_batch,1) 
        ### qvalue_batch =qTValue_baycj[ff]
        ###  reward + GAMMA *(qvalue_batch)  use q to estimate target value ...

        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch
            })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.sess, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()


    def make_action(self,observation,test=True):
        QValue = self.QValue.eval(feed_dict= {self.stateInput:observation.reshape(1,84,84,4)})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon and not test :

            action_index = random.randrange(self.actions)

        else:

            action_index = np.argmax(QValue)

        if test and random.random() < 0.01 : 

            action_index = random.randrange(self.actions)

        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action_index

    def percieve(self,state,action,reward,next_state,done):
        action_one_hot = np.zeros(4)
        action_one_hot[action] = 1 

        self.replay_memory.append((state,action_one_hot,reward,next_state,done))

        if len(self.replay_memory) >REPLAY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > BATCH_SIZE and self.timeStep % self.train_skip==0 :
            self.train_Q_network()

        self.timeStep+=1

    def createTrainingMethod(self):

        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)







    def update(self,sess,observation , action , target_value ): #### regression problem 

        feed_dict = {self.q_x_pl:observation , self.q_action_pl:action , self.q_y_pl : target_value }
        summaries , _ , loss ,gl_step= sess.run([self.merge,self.q_train_op,self.q_loss,tf.contrib.framework.get_global_step()],feed_dict=feed_dict)

        if self.writer :
             self.writer.add_summary(summaries,gl_step)

        return loss 

    def copy_train_parameters(self, sess , q_para , target_para): ### target_model is fixed , training after ?? epochs .

        #e1_param = [t for t in tf.trainable_variables() if name.startswith(model.scope)]
        e1_params = sorted(q_para,key = lambda v:v.name)

        #e2_param = [t for t in tf.trainable_variables() if name.startswith(target_model.scope)]
        e2_params = sorted(target_para , key = lambda v: v.name)
        assign_op = []

        for v_old , v_new in zip(e1_params,e2_params):
            assign_op.append(v_new.assign(v_old))

        sess.run(assign_op)

    def make_epsilon_greedy_police(self, nA):

        def policy_fn(sess,observation,epsilon):
            A = np.ones(nA , dtype=float) * epsilon /nA
            q_values = self.predict(sess,np.expand_dims(observation,0))[0]
            best_action  = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A 

        return policy_fn 


    def deep_q_algorithm(self,sess,num_episodes,state_processor,replay_memory_size=10000 , replay_memory_init_size=1000,\
                         update_target_estimator_every = 1000 , discount_factor =0.99 , epsilon_start = 1 ,epsilon_end = 0.025,epsilon_decay_steps=10000,\
                         batch_size = 32 , record_video_every = 50 ):

        Transition = namedtuple('Transition' , ['state','action','reward','next_state','done'])
        ### like buffer , right ? 
        replay_memory = []

        stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes) , episode_rewards = np.zeros(num_episodes))

        checkpoint_path = 'deep_q/'
        monitor_path = 'monitor/'

        saver = tf.train.Saver()

        latest_check_point = tf.train.latest_checkpoint(checkpoint_path)

        if latest_check_point : 
            print("Loading checkpint of q model {} ....\n".format(latest_checkpoint))
            saver.restore(sess,latest_checkpoint)

        # get the current time step 
        total_t = sess.run(tf.contrib.framework.get_global_step())

        # epsilon decay sechedule : 

        epsilons = np.linspace(epsilon_start,epsilon_end,epsilon_decay_steps)

        ##policy we follow ?  

        policy = self.make_epsilon_greedy_police(len(VALID_ACTIONS))

        ### populate the replay memory with initial experience 

        print('populating replay memory ....')
        state = self.env.reset()
        #print(state.shape)
        #state = state_processor.process(sess,state)
        #State = np.stack([state]*4, axis=2)  #### why multiply for 4 ?? 
        for i in range(replay_memory_init_size):
            action_probs = policy(sess,state,epsilons[min(total_t,epsilon_decay_steps-1)])
            action = np.random.choice(np.arange(len(action_probs)),p=action_probs)
            next_state , reward , done , info = self.env.step(action)
            #next_state = state_processor.process(next_state)
            #print(next_state.shape)
            #next_state = np.append(state[:,:,1],np.expand_dims(next_state,2),axis=2)
            replay_memory.append(Transition(state,action,reward,next_state,done))  ## store state , reward ,action (current) , next_state about this action ..

            if done : 
                state = self.env.reset()
                #state = state_processor.process(sess,state)
                #state = np.stack([state]*4 , axis=2)
            else : 
                state = next_state


        # blow will save about the video this episode play ...  bY MONITOR .. 

        #mon = Monitor(self.env,directory=monitor_path,video_callable = lambda count : count % record_video_every ==0 ,resume=True)

        for i_episode in range(num_episodes):

            saver.save(tf.get_default_session(),checkpoint_path)

            state = self.env.reset()
            #state = state_processor.process(sess,state)
            #state = np.stack([state]*4,axis=2)
            loss = None 

            # one step in the enviroment 

            for t in itertools.count():
                ##epsilon for this time step 
                epsilon = epsilons[min(total_t,epsilon_decay_steps-1)]

                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=epsilon,tag='epsilon')
                self.writer.add_summary(episode_summary,total_t)

                if total_t % update_target_estimator_every == 0 : 
                    self.copy_train_parameters(sess,self.q_para,self.target_para)
                    print('\n Copied model parameters to target network')


                print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")

                sys.stdout.flush()

                #take a step 
                action_probs = policy(sess,state,epsilon)
                action = np.random.choice(np.arange(len(action_probs)),p=action_probs)
                next_state , reward , done ,info = self.env.step(VALID_ACTIONS[action])
                #next_state = state_processor.process(sess,next_state)
                #next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

                ## if full , drop one from replay memory .. 

                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                replay_memory.append(Transition(state, action, reward, next_state, done))
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                samples = random.sample(replay_memory,batch_size)
                states_batch , action_batch , reward_batch , next_state_batch , done_batch = map(np.array,zip(*samples))

                q_values_next = self.target_predict(sess,next_state_batch)

                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.argmax(q_values_next,1)

                states_batch = np.array(states_batch)

                for _ in range(4):
                    loss = self.update(sess,states_batch,action_batch,targets_batch)

                if done : 
                    break

                state = next_state
                total_t +=1 

            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
            episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
            self.writer.add_summary(episode_summary)
            self.writer.flush()
            yield total_t ,plotting.EpisodeStats(
                episode_lengths=stats.episode_lengths[:i_episode+1],
                episode_rewards=stats.episode_rewards[:i_episode+1])

        return stats



    def build_model(self,name,summary_dir = None): 
        with tf.variable_scope(name): 
            x_pl = tf.placeholder(shape=[None,84,84,4],dtype=tf.float32,name='x')
            y_pl = tf.placeholder(shape=[None] , dtype =tf.float32,name='target_value')
            action_pl = tf.placeholder(shape=[None],dtype=tf.int32,name='actions')
            x = tf.to_float(x_pl)/255.0
            batch_size = tf.shape(x_pl)[0]
            conv1 = ly.conv2d(x_pl,32,8,4,activation_fn = tf.nn.relu)
            conv2 = ly.conv2d(conv1,64,4,2,activation_fn = tf.nn.relu)
            conv3 = ly.conv2d(conv2,64,3,1,activation_fn = tf.nn.relu)
            flatten = ly.flatten(conv3)
            #fc1 = ly.fully_connected(flatten , 128 , activation_fn = tf.nn.relu)
            fc2 = ly.fully_connected(flatten , 512 , activation_fn = tf.nn.relu)
            predictions = ly.fully_connected(fc2 , 4 , activation_fn = None)

            gather_indices = tf.range(batch_size) * tf.shape(predictions)[1] + action_pl

            action_predictions = tf.gather(tf.reshape(predictions,[-1]),gather_indices)
            losses = tf.squared_difference(y_pl , action_predictions)
            loss = tf.reduce_mean(losses)

            optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            train_op = optimizer.minimize(loss)



        Inp = (x_pl , y_pl , action_pl)

        parameters  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name)


        if summary_dir is not None :
            summary = tf.summary.merge([
                tf.summary.scalar("loss", loss),
                tf.summary.histogram("loss_hist", losses),
                tf.summary.histogram("q_values_hist", predictions),
                tf.summary.scalar("max_q_value", tf.reduce_max(predictions))])



            #writer = tf.summary.FileWriter(tensorboard_dir) 

            return Inp ,predictions ,loss , train_op , summary ,parameters

        else : 
            return Inp , predictions , loss , train_op , parameters


    def _train(self):

        state_processor = StateProcessor()

        #tf.reset_default_graph()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        #print(self.q_x_pl)
        #ess = tf.Session()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #print(self.env.action_space)
            #print(state_processor.process(sess,np.zeros((210,160,3))).shape)
            #print(self.env.reset())
            #print(self.env.step(1)[0].dtype)
            #state = self.env.reset()
            #print(state[:,:,0])
            #print(self.StateProcessor.input_state)
            #print(self.prepro(sess,np.zeros((210,160,3))))
            for t , stats in self.deep_q_algorithm(sess,num_episodes=100000,state_processor= state_processor):

                #print("\nEpisode Reward: {}".format(stats.episode_rewards[:-1]))
                print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################

        #self.sess = tf.Session()

        #self.sess.run(tf.global_variables_initializer)

        #self.epsilons = np.linspace(epsilon_start,epsilon_end,epsilon_decay_steps)

        #self.policy = self.make_epsilon_greedy_police(len(VALID_ACTIONS))
        #self.epsilon_decay_steps = 100
        #self.total_t = 0

        #self.epsilons =  np.linspace(1,0.025,self.epsilon_decay_steps)
        random.seed(9)
        #pass

    '''
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        action_probs = self.policy(self.sess,observation,self.epsilons[min(self.total_t,self.epsilon_decay_steps-1)])

        #prob = np.argmax(self.predict(self.sess,observation.reshape(1,84,84,4)))

        action = np.random.choice(np.arange(len(action_probs)),p=action_probs)
        print(action)

        self.total_t +=1 
        #return self.env.get_random_action()
        return action
    '''
