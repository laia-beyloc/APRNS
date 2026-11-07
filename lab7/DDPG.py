
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DDPG:
    """The DDPG Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=32, gamma=0.99):
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.      
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma

        # TODO (2): Initialize the Actor and Critic networks. 
        # Initialize Critic network and target network. Should be named self.Critic 
        self.Critic = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        # El constructor del crític (a diferència del de l'actor) no crida
        # initialize_weights(); la cridem aquí per seguir la inicialització de [1], secció 7.
        self.Critic.initialize_weights()
        self.Critic_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic_target, self.Critic)

        # Initialize Actor network and its target network. Should be named self.Actor
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor_target, self.Actor)

        # τ de [1] per a les actualitzacions suaus (TODO 7) i comptadors d'iteracions
        # d'aprenentatge, que es registren com a diagnòstic.
        self.tau = 0.001
        self.critic_updates = 0
        self.actor_updates = 0
        # Claus del dict que retorna update(); el bucle comú les fa servir per fixar el CSV.
        self.metric_names = ("critic_loss", "actor_loss")
        # END TODO (2)

        # Define the optimizers for the actor and critic networks as proposed in the paper
        self.optim_dqn = optim.Adam(self.Critic.parameters(), lr=0.001, weight_decay=0.01) 
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.0001) 


    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []
        timeexit = timesteps

        # We use here OUNoise instead of Gaussian to add some exploration to the agent. OU noise is a stochastic process
        # that generates a random sample from a Gaussian distribution whose value at time t depends on the previous value
        # x(t) and the time elapsed since the previous value y(t). It helps to explore the environment better than Gaussian noise.
        # This line initializes the noise with mean 0 and sigma 0.15 (see Noise.py file)
        OUNoise =  OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_dim))

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.
            epsilon= OUNoise.sample()
            action = np.clip(action + epsilon, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                #TODO (6): if there is enouygh data in the replay buffer, sample a batch and perform an optimization step
                # Batch is sampled from the replay buffer and containes a list of tuples (s, a, r, s', term, trunc)

                # Get the batch data
                batch = self.replay_buffer.get(self.batch_size)

                # Compute the loss for the critic and update the critic network 
                # (fet dins update(): és el mateix pas que fa servir el bucle comú d'experiments)

                # Compute the loss for the actor and update the actor network 
                self.update(batch)

                # END TODO (6)

            # TODO (7): Sync the target networks with soft updates and tau=0.001 according to details of the DDPG paper
            # Les actualitzacions suaus es fan dins update(), just després d'aprendre del
            # minibatch: només tenen sentit quan les xarxes en línia han canviat.
            if terminated or truncated:
                # Figura 1: "Initialize a random process N" a l'inici de cada episodi.
                OUNoise.reset()

            # END TODO (7)

            if timestep % (timesteps-1) == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                pass
            if len(all_rewards_eval)>10 and np.mean(all_rewards_eval[-5:]) > 220:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                break
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        # TODO (3) Implement the function to choose an action given a state. It is deterministic because exploration is added
        # by the OrnsteinUhlenbeckActionNoise in the main loop.
        state = torch.as_tensor(s, dtype=torch.float32, device=device)
        with torch.no_grad():
            a = self.Actor(state).cpu().numpy()

        # END TODO (3)
        return a


    def compute_critic_loss(self, batch):
        """
        The function computes the critic loss using the Mean Squared Bellman Error (MSBE) calculation.
        
        :param batch: The `batch` parameter is a tuple containing the data for computing the loss.
        :return: the critic loss, which is calculated using the mean squared error (MSE) loss between
        the expected Q-values (q_expected) and the target Q-values (target).
        """
        
        # TODO (4): Implement MSBE calculation (need to sample from replay buffer first). Notice that it is VERY 
        # similar to the DQN loss.
        states, actions, rewards, next_states, terminated = self._batch_to_tensors(batch)
        with torch.no_grad():
            # Només una terminació real talla el bootstrap; una truncació per temps
            # no és un estat terminal i s'hi continua estimant el valor de s'.
            next_q = self.Critic_target(next_states, self.Actor_target(next_states))
            target = rewards + self.gamma * (1.0 - terminated) * next_q
        loss = nn.functional.mse_loss(self.Critic(states, actions), target)

        # END TODO (4)
        return loss
    

    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network 
        
        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        # TODO (5) implement the actor loss. You have to sample from the replay buffer first a set of states.
        states = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
        # Minimitzar -Q(s, μ(s)) és pujar pel gradient determinista ∇_a Q · ∇_θ μ (figura 1).
        loss = -self.Critic(states, self.Actor(states)).mean()

        # END TODO (5) 

        return loss


    def update(self, batch):
        """Una iteració d'aprenentatge de DDPG sobre un minibatch (figura 1).

        Actualitza el crític (MSBE), després l'actor (gradient determinista) i fa les
        actualitzacions suaus θ' ← τθ + (1−τ)θ' de les dues xarxes objectiu.
        És el pas que criden tant learn() com el bucle comú d'experiments.

        :param batch: tupla (s [B,obs], a [B,act], r [B], s' [B,obs], terminated [B], truncated [B]).
        :return: dict de mètriques {"critic_loss", "actor_loss"}.
        """
        critic_loss = self.compute_critic_loss(batch)
        self.optim_dqn.zero_grad()
        critic_loss.backward()
        self.optim_dqn.step()

        # El backward de l'actor també deixa gradients al crític; no afecten res
        # perquè optim_dqn.zero_grad() els esborra abans del següent pas del crític.
        actor_loss = self.compute_actor_loss(batch)
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        soft_update(self.Critic_target, self.Critic, self.tau)
        soft_update(self.Actor_target, self.Actor, self.tau)
        self.critic_updates += 1
        self.actor_updates += 1
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


    def _batch_to_tensors(self, batch):
        """Converteix un minibatch a tensors del dispositiu de les xarxes.

        :return: s [B,obs], a [B,act], r [B,1], s' [B,obs], terminated [B,1].
        La marca de truncació no es retorna: no intervé en l'objectiu de Bellman.
        """
        states, actions, rewards, next_states, terminated, _ = batch
        as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return (as_tensor(states), as_tensor(actions), as_tensor(rewards).reshape(-1, 1),
                as_tensor(next_states), as_tensor(terminated).reshape(-1, 1))



    def eval_episodes(self,n=3):
        """ Evaluate an agent performing inside a Gym environment. """
        lr=[]
        for episode in range(n):
            tr = 0.0
            obs, _ = self.env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)




if __name__ == '__main__':
    # Create gym environment
    env = gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array')

    ddpg = DDPG(env,replay_size=1000000, batch_size=64, gamma=0.99)

    ddpg.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array'),'video')    
    video_agent(env, ddpg,n_episodes=5)  
    pass
