from agent_dir.agent import Agent
import scipy
import numpy as np
import tensorflow.contrib.layers as ly 
import os 
import sys 
import pandas as pd 
import tensorflow as tf 
import copy
#import policy_net
from agent_dir import policy_net
up_action = 2
down_action = 3 #### real act.space 
action_dict = {down_action:0,up_action:+1}



def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards


'''
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    gamma = 0.99
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
        if r[t] !=0 :
            running_add =0 
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r
'''

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)  ## this make te environment of PONG

        

        self.args = args
        if args.improved == False : 
            self.up_prob, Inp = self.bulid_model_variable(improved=False)

            self.actions = Inp[1]
            self.observation = Inp[0]
            self.advantage = Inp[2]
            self.loss = self.pg_loss(self.up_prob,self.actions,self.advantage)

            self.merged = tf.summary.merge_all()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0005,epsilon=1e-5)
            self.train_op = optimizer.minimize(self.loss)
        else : 
            self.assign_op = []
            self.gamma = 0.95

            #self.obs = tf.placeholder(tf.float32,shape=[None,6400])
            self.policy ,self.old_policy,self.pi_trainable,self.old_pi_trainable,Inp = self.bulid_model_variable(improved=True)
            self.old_obs = self.old_policy.obs
            self.obs = self.policy.obs
            self.actions = Inp[0]
            self.rewards = Inp[1]
            self.v_preds_next = Inp[2]
            self.gas = Inp[3]
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,epsilon=1e-5)
            self.loss ,self.vf_loss = self.improve_loss(self.policy,self.old_policy,self.actions,self.rewards,self.gas,self.gamma,self.v_preds_next)
            self.train_op = optimizer.minimize(self.loss,var_list = self.pi_trainable)
            self.merged = tf.summary.merge_all()
        
        self.sess = tf.Session()
        self.saver  = tf.train.Saver()
        if args.test_pg:
            self.saver.restore(self.sess,'policy_network.ckpt')
            print('loading trained model')
        ## add training 

        ### bulid model and loss function 
        ##################
        # YOUR CODE HERE #
        ##################
    def bulid_model_variable(self,improved=False):
        if improved:
            with tf.variable_scope('train_inp'):
                actions = tf.placeholder(tf.int32,shape=[None],name='actions')
                rewards = tf.placeholder(tf.float32,shape=[None],name = 'rewards')
                v_preds_next = tf.placeholder(tf.float32,shape=[None] , name = 'v_preds_next')   #### for PPo
                gas = tf.placeholder(tf.float32,shape=[None] , name='gas')  #### for PPo

            policy = policy_net.Policy_net('policy',self.env)
            old_policy = policy_net.Policy_net('old_policy',self.env)  
            pi_trainable = policy.get_trainable_variables()
            old_pi_trainable = old_policy.get_trainable_variables()

            Inp = (actions , rewards ,v_preds_next ,gas)

            return policy , old_policy , pi_trainable ,old_pi_trainable , Inp
        else : 

            with tf.variable_scope('train_inp'):
                observation = tf.placeholder(tf.float32,[None,6400])
                actions = tf.placeholder(tf.float32,[None,1])
                rewards = tf.placeholder(tf.float32,[None,1])

            h = tf.layers.dense(observation,200,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

            #h = ly.fully_connected(h,256,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer())

            prob = tf.layers.dense(h,1,activation=tf.nn.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer())

            #prob = tf.nn.sigmoid(logits,name='sigmoid')

            Inp = (observation , actions ,rewards)

            return prob , Inp
 
    def get_gaes(self,rewards,v_preds,v_preds_next):
        delta = [r_t +self.gamma * v_next - v for r_t,v_next,v in zip(rewards,v_preds_next,v_preds)]
        gaes = copy.deepcopy(delta)
        for t in reversed(range(len(gaes)-1)):
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes
    '''
    def prepro(self,o,image_size=[80,80]):
        """
        Call this function to preprocess RGB image to grayscale image if necessary
        This preprocessing code is from
            https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
        
        Input: 
        RGB image: np.array
            RGB screen of game, shape: (210, 160, 3)
        Default return: np.array 
            Grayscale image, shape: (80, 80, 1)
        
        """
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        return np.expand_dims(resized.astype(np.float32),axis=2).flatten()
    '''
    def prepro(self,I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        
        return I.astype(np.float32).ravel()


    def improve_loss(self,policy,old_policy,action,rewards,gas,gamma,v_preds_next,clip_value=0.2) :
        act_probs = policy.act_probs
        act_probs_old = old_policy.act_probs
        act_probs = act_probs * tf.one_hot(action,act_probs.shape[1])
        #act_probs = act_probs * tf.one_hot(action,2)
        act_probs = tf.reduce_sum(act_probs,1)
        #act_probs = act_probs * tf.one_hot(action,act_probs.shape[1])
        act_probs_old = act_probs_old * tf.one_hot(action,act_probs_old.shape[1])
        #act_probs_old = act_probs_old * tf.one_hot(action,act_probs_old.shape[1])
        #act_probs_old = act_probs_old * tf.one_hot(action,2)
        act_probs_old = tf.reduce_sum(act_probs_old,1)
        with tf.variable_scope('Vc'):
                ratio = tf.exp(tf.log(act_probs)-tf.log(act_probs_old))
                clipped_ratios = tf.clip_by_value(ratio,clip_value_min=1-clip_value,clip_value_max=1+clip_value)
                loss_clip = tf.minimum(tf.multiply(gas,ratio),tf.multiply(gas,clipped_ratios))
                loss_clip = tf.reduce_mean(loss_clip)
                tf.summary.scalar('Vc',loss_clip)

        with tf.variable_scope('entropy'):
                entropy = -tf.reduce_sum(policy.act_probs * tf.log(tf.clip_by_value(policy.act_probs,1e-10,1.0)),axis=1)
                entropy = tf.reduce_mean(entropy,axis=0)
                tf.summary.scalar('entropy',entropy)

        with tf.variable_scope('Vf'):## mse of ??  , that's not in my knowledge
                v_preds = policy.v_preds
                loss_vf = tf.squared_difference(rewards+gamma*v_preds_next , v_preds)
                loss_vf = tf.reduce_mean(loss_vf)
                tf.summary.scalar('Vf',loss_vf)
        c_1 = 1 
        c_2 = 0.01
        with tf.variable_scope('total_loss'):
                loss = loss_clip - c_1 * loss_vf + c_2 * entropy  ### maximum
                loss = -loss #(minimum)
                tf.summary.scalar('total_loss',loss)
        return loss , loss_vf
    def pg_loss(self,prob,sampled_label,reward):
        with tf.variable_scope('total_loss'):
            loss = tf.losses.log_loss(labels=sampled_label,predictions=prob,weights=reward)
            #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=sampled_label,logits=logits)
            #loss = tf.reduce_sum(tf.multiply(reward,cross_entropy))
            tf.summary.scalar('total_loss',loss)
        return loss


    def pg_forward_pass(self,sess,observation):
        return sess.run(self.up_prob,feed_dict={self.observation:observation.reshape(1,-1)})
    

    #testing phase !!!!!!!!!!!!
    def pg_train(self):

        #training detrain ( include each iteration collect actions and so on...)
        # self.env
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.env.seed(9)

        tensorboard_dir = 'RL/'   
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)


        writer = tf.summary.FileWriter(tensorboard_dir)     

        #states , actions , rewards = zip(*state_action_reward_tuples)
        #states = np.vstack(states)
        #actions = np.vstack(actions)
        #rewards = np.vstack(rewards)

        #fedd_dict = {self.observation:states,self.actions:actions,self.advantage:rewards}

        #self.sess.run(self.train_op,feed_dict=feed_dict)   

        count=0
        episode_n = 1 
        smoothed_reward = None #########  reference github !!!! 
        batch_state_action_reward_tuple = []
        while True : 
            print('starting collectinng {}th trajectory !!!'.format(count))

            episode_done =False
            episode_reward_sum = 0
            round_n = 1

            ############ first timestamp ###############

            last_observation = self.env.reset()
            last_observation = self.prepro(last_observation)
            action = self.env.action_space.sample()
            observation , _ , _ , _ = self.env.step(action) 
            observation = self.prepro(observation)
            n_steps = 1 

            ############################################
            while not episode_done :            
                if self.args.render:
                    self.env.render()

                observation_delta = observation - last_observation
                last_obervation = observation 
                up_prob = self.pg_forward_pass(sess,observation_delta)[0]
                #print(up_prob)
                if np.random.uniform() < up_prob : 
                    action = up_action
                else : 
                    action = down_action
                
                observation , reward ,episode_done , info = self.env.step(action)
                observation = self.prepro(observation)
                episode_reward_sum += reward
                n_steps += 1 

                tup =(observation_delta ,action_dict[action],reward)
                batch_state_action_reward_tuple.append(tup)
                if reward == -1 : 
                    print("Round %d: %d time steps; lost..." % (round_n, n_steps))
                elif reward == +1 : 
                    print("Round %d: %d time steps; win..." % (round_n, n_steps))
                if reward != 0 : 
                    round_n +=1 
                    n_steps = 0

            print("Episode %d finished after %d rounds" %(episode_n,round_n))

            if smoothed_reward is None : 
                smoothed_reward = episode_reward_sum
            else : 
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum *0.01

            print("Reward total was %.3f; discounted moving average of reward is %.3f "\
                % (episode_reward_sum , smoothed_reward))  ####like tensorboard 

            if episode_n %1 == 0:
                states , actions , rewards = zip(*batch_state_action_reward_tuple)
                rewards = discount_rewards(rewards,0.99)
                rewards -= np.mean(rewards)   ## normalization 
                rewards /= np.std(rewards)    ## normalization 

                batch_state_action_reward_tuple = list(zip(states,actions,rewards))
                states, actions, rewards = zip(*batch_state_action_reward_tuple)
                
                states = np.vstack(states)
                actions = np.vstack(actions)
                rewards = np.vstack(rewards)

                feed_dict = {self.observation:states,self.actions:actions,self.advantage:rewards}
                merge , loss_1 ,_ =sess.run([self.merged ,self.loss,self.train_op],feed_dict = feed_dict)
                writer.add_summary(merge,count)
                print(loss_1)

                batch_state_action_reward_tuple = []

            if count %5 == 0 :
                saver.save(sess,'RL/check_point'+str(count))
            ### add save check point 
            episode_n +=1
    '''
    def pg_train(self):
        render = False 
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.env.seed(9)
        tensorboard_dir = 'RL/'   
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.FileWriter(tensorboard_dir)
        observation = self.env.reset()
        prev_x = None 
        xs = []
        ys = []
        ws = []
        ep_ws = []
        batch_ws = []

        step = 0
        episode_numbe = 0
        reward_mean = -21
        while  True:
            #print(0)
            cur_x = prepro(observation)

            if prev_x is not None : 
                x = cur_x - prev_x 
            else : 
                x = cur_x

            prev_x = cur_x

            tf_probs = sess.run(self.up_prob,feed_dict = {self.observation:x.reshape(-1,6400)})
            if np.random.uniform() < tf_probs[0,0]:
                action = 2 + 1
            else : 
                action = 2 + 0 

            observation , reward , done , info = self.env.step(action)

            xs.append(x)
            ys.append(action)
            ep_ws.append(reward)
            #print(done)

            if done : 
                episode_number +=1
                print(episode_number)
                discounted_epr = discount_rewards(ep_ws)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                batch_ws += discounted_epr.to_list

                reward_mean = 0.99*reward_mean + (0.01) * sum(ep_ws)

                rs_sum = tf.Summary(value = [tf.Summary.Value(tag='running_reward',simple_value=reward_mean)])
                writer.add_summary(re_sum,global_step = episode_number)

                if reward_mean > 5.0 : 
                    break

                if episode_number % args.batch_size ==0 : 
                    step +=1 
                    print(step)
                    exs = np.vstack(xs)
                    eys = np.vstack(ys)
                    ews = np.vstack(batch_ws)
                    frame_size = len(xs)
                    xs = []
                    ys = []
                    batch_ws = []

                    _,summary = sess.run([self.train_op,self.merged],feed_dict={self.observation:exs,self.actions:eys,self.advantage:ews})
                    writer.add_summary(summary , step )


                if step%5 == 0 :
                    saver.save(sess,'check_point'+str(step))

            observation = self.env.reset()
    '''

    def improved_train(self):

        with tf.variable_scope('assign_op'):
            for v_old , v_new in zip(self.old_pi_trainable,self.pi_trainable):
                self.assign_op.append(tf.assign(v_old,v_new))

        Iteration = 10000
        gamma = 0.95
        #sess = tf.Session()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.env.seed(9)

        tensorboard_dir = 'RL_improved/'   
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.FileWriter(tensorboard_dir)            

        obs = self.env.reset()
        reward = 0
        success_num = 0
        count = 0
        for iteration in range(Iteration):

            obs = self.prepro(obs)  ###########
            last_obs = self.env.reset()
            last_obs = self.prepro(last_obs)
            action = self.env.action_space.sample()

            obs, _ ,_ ,_ = self.env.step(action)

            obs = self.prepro(obs)
            observations = []
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0
            while True : # run_policy_steps which is much less than episode length 
                run_policy_steps +=1
                #obs = np.stack([self.prepro(obs)])
                obs_delta = obs - last_obs
                last_obs = obs
                act , v_pred = self.policy.act(obs_delta.reshape(1,-1),stochastic=True)
                #act , v_pred = self.policy.act(obs,stochastic=True)
                #if np.random.uniform() < act : 
                #    action = up_action
                #else : 
                #    action = down_action
                #print(act)
                if act == 1:
                    temp = up_action
                else:
                    temp = down_action
                #act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                #observations.append(obs)
                observations.append(obs_delta)
                #actions.append(act)
                actions.append(action_dict[temp])
                v_preds.append(v_pred)
                rewards.append(reward)

                #next_obs , reward , done ,info = self.env.step(act)
                obs , reward , done ,info = self.env.step(temp)
                #print(reward)
                if reward == -1 :
                    print('Loss')
                elif reward == +1 :
                    print('win')

                if done : 
                    v_preds_next = v_preds[1:] + [0]
                    obs = self.env.reset()
                    #obs = self.prepro(obs)
                    reward = -1
                    break
                else :
                    #obs = next_obs
                    obs = obs
                    obs = self.prepro(obs)

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 100 : 
                success_num += 1 
                if success_num >= 10:
                    saver.save(sess,'RL_improved/model'+str(iteration))
                    print('Pass !! Model saved.')
                    break
            else : 
                success_num = 0

            gaes = self.get_gaes(rewards=rewards,v_preds=v_preds,v_preds_next=v_preds_next)

            ### what I need is numpy array 

            observations = np.reshape(observations ,newshape=[-1,6400]).astype(np.float32)
            actions  = np.array(actions).astype(np.int32)
            rewards = np.array(rewards).astype(np.float32)
            v_preds_next = np.array(v_preds_next).astype(np.float32)
            gaes = np.array(gaes).astype(np.float32)
            #print(gaes.shape)
            gaes = (gaes - gaes.mean()) / gaes.std()
            #print(gaes.shape)
            #print(observations.shape)
            #print(rewards.shape)
            #print(v_preds_next.shape)

            ####it problem from gae 
            sess.run(self.assign_op)

            inp = [observations, actions , rewards , v_preds_next , gaes]
            for epoch in range(4):
                sample_indices = np.random.randint(low=0,high=observations.shape[0],size=64)

                sample_inp = [np.take(a=a,indices=sample_indices,axis=0) for a in inp]
                #print(sample_inp[0].shape)
                #print(sample_inp[1].shape)
                #print(sample_inp[2].shape)
                #print(sample_inp[3].shape)
                #print(sample_inp[4].shape)
                #print(self.policy.v_preds)
                #print((self.rewards+self.gamma*self.v_preds_next)- self.policy.v_preds )
                #print(self.rewards - self.v_preds_next)
                #print(self.policy.v_preds)
                feed_dict = {self.obs:sample_inp[0],self.old_obs:sample_inp[0],self.actions:sample_inp[1],self.rewards : sample_inp[2],
                            self.v_preds_next:sample_inp[3],self.gas:sample_inp[4]}
                #print(feed_dict[self.obs].shape)
                #print(feed_dict[self.old_obs].shape)
                #print(feed_dict[self.actions].shape)
                #print(feed_dict[self.rewards].shape)
                #print(feed_dict[self.v_preds_next].shape)
                #print(feed_dict[self.gas].shape)
                #loss_2=sess.run(self.policy.v_preds,feed_dict=feed_dict)   ## (64,1)
                loss_2 = sess.run(self.vf_loss ,feed_dict=feed_dict)
                merge , loss_1 , _ = sess.run([self.merged,self.loss,self.train_op],feed_dict=feed_dict)
                #print(loss_2.shape)
                writer.add_summary(merge,count)
                count+=1


    ################ may be the testing phase !!!!
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = self.prepro(observation)
        observation_delta = observation -self.test_last_observation 
        self.test_last_observation = observation 
        up_prob = self.pg_forward_pass(self.sess,observation_delta)[0]

        if np.random.uniform() < up_prob:
            action=up_action
        else : 
            action = down_action
        return action 

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        #self.sess = tf.Session()
        #self.saver = tf.train.Saver()
        #self.saver.restore(self.sess,'policy_network.ckpt')
        self.test_last_observation = self.env.reset()
        self.test_last_observation = self.prepro(self.test_last_observation)
        np.random.seed(9)