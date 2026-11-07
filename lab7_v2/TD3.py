"""Twin Delayed DDPG (TD3), apartat 3 i figura 5 de l'enunciat.

L'enunciat demana implementar TD3 en una CÒPIA de DDPG.py anomenada TD3.py; per
això la classe agent és una còpia (l'única duplicació acceptada del projecte) i
no una subclasse. Canvis respecte de DDPG:
  1. suavitzat de la política objectiu (3.1.1),
  2. dos crítics amb el mínim al target i l'actor sobre Q1 (3.1.2),
  3. actualitzacions retardades de l'actor i de tots els targets (3.1.3),
  4. exploració gaussiana de Noise.py en lloc d'OU.
Els interruptors de l'apartat 4 (use_minimum, target_noise, policy_delay) arriben
com a paràmetres; la classe no coneix el nom de cap variant.
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise
from gymnasium.wrappers import RecordVideo

from config import TARGET_NOISE_CLIP, TD3_EXPLORATION_SIGMA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TD3:
    """The TD3 Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=32, gamma=0.99,
                 use_minimum=True, target_noise=0.2, policy_delay=2, noise_clip=TARGET_NOISE_CLIP):
        """ Initializes the TD3 method.

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
        use_minimum: bool
            Target amb min(Q1', Q2') (True) o només amb Q1' (False, variant "no_minimum").
        target_noise: float
            σ_targ del suavitzat de l'acció objectiu (0 desactiva el suavitzat).
        policy_delay: int
            Iteracions d'aprenentatge del crític per cada actualització de l'actor i dels targets.
        noise_clip: float
            c, retall del soroll del suavitzat.
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_minimum = use_minimum
        self.target_noise = target_noise
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip

        # TODO (2): Initialize the Actor and Critic networks.
        # Initialize Critic network and target network. Should be named self.Critic
        # Dos crítics inicialitzats de manera independent (figura 5). La creació no
        # depèn dels interruptors: amb la mateixa llavor, totes les variants parteixen
        # exactament dels mateixos pesos.
        self.Critic = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic.initialize_weights()
        self.Critic_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic_target, self.Critic)
        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2.initialize_weights()
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic2_target, self.Critic2)

        # Initialize Actor network and its target network. Should be named self.Actor
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor_target, self.Actor)

        # τ igual que a DDPG (3.1.3: no es compensa el retard canviant-lo).
        self.tau = 0.001
        self.critic_updates = 0
        self.actor_updates = 0
        # Claus del dict que retorna update(); el bucle comú les fa servir per fixar el CSV.
        self.metric_names = ("critic_loss", "critic2_loss", "actor_loss",
                             "critic_disagreement", "min_reduction")
        # END TODO (2)

        # Define the optimizers for the actor and critic networks as proposed in the paper
        # El segon crític fa servir el mateix optimitzador i hiperparàmetres que el primer.
        self.optim_dqn = optim.Adam(self.Critic.parameters(), lr=0.001, weight_decay=0.01)
        self.optim_critic2 = optim.Adam(self.Critic2.parameters(), lr=0.001, weight_decay=0.01)
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

        # Apartat 3.1.1: TD3 explora amb soroll gaussià N(0, σ) de Noise.py (sense memòria,
        # per tant no cal reiniciar-lo entre episodis). σ és a config.py.
        noise = NormalActionNoise(np.zeros(self.act_dim), TD3_EXPLORATION_SIGMA)

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.
            epsilon= noise.sample()
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
                # Compute the loss for the actor and update the actor network
                # (tot dins update(): crítics a cada iteració, actor i targets cada policy_delay)
                self.update(batch)

                # END TODO (6)

            # TODO (7): Sync the target networks with soft updates and tau=0.001 according to details of the DDPG paper
            # A TD3 les actualitzacions suaus van lligades a les de l'actor (retardades),
            # per això es fan dins update().
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
        # by the exploration noise in the main loop.
        state = torch.as_tensor(s, dtype=torch.float32, device=device)
        with torch.no_grad():
            a = self.Actor(state).cpu().numpy()

        # END TODO (3)
        return a


    def target_action(self, next_states):
        """Acció objectiu suavitzada (apartat 3.1.1).

        ã' = clip(μ'(s') + clip(ε, −c, c), −1, 1),  ε ~ N(0, σ_targ² I).
        :param next_states: [B,obs]. :return: ã' [B,act].
        El soroll es sorteja sempre, també amb σ_targ = 0, perquè la variant sense
        suavitzat consumeixi la mateixa seqüència aleatòria que les altres.
        """
        actions = self.Actor_target(next_states)
        noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
        return (actions + noise).clamp(-1.0, 1.0)


    def compute_critic_loss(self, batch):
        """
        MSBE dels dos crítics cap al MATEIX target (apartat 3.1.2):
        y = r + γ(1−terminated)·min(Q1', Q2')(s', ã')   (o només Q1' si use_minimum=False).

        :param batch: tupla (s [B,obs], a [B,act], r [B], s' [B,obs], terminated [B], truncated [B]).
        :return: (loss_Q1 + loss_Q2, dict de mètriques). La suma permet un sol backward;
        els paràmetres dels dos crítics són disjunts, així que cada un rep el seu gradient.
        """

        # TODO (4): Implement MSBE calculation (need to sample from replay buffer first). Notice that it is VERY
        # similar to the DQN loss.
        states, actions, rewards, next_states, terminated = self._batch_to_tensors(batch)
        with torch.no_grad():
            next_actions = self.target_action(next_states)
            q1_next = self.Critic_target(next_states, next_actions)
            q2_next = self.Critic2_target(next_states, next_actions)
            min_next = torch.minimum(q1_next, q2_next)
            next_q = min_next if self.use_minimum else q1_next
            # Només una terminació real talla el bootstrap (la truncació per temps no).
            target = rewards + self.gamma * (1.0 - terminated) * next_q
        loss1 = nn.functional.mse_loss(self.Critic(states, actions), target)
        loss2 = nn.functional.mse_loss(self.Critic2(states, actions), target)
        # Diagnòstics de l'apartat 4: es calculen a totes les variants, encara que
        # el mínim no s'utilitzi al target.
        metrics = {"critic_loss": loss1.item(), "critic2_loss": loss2.item(),
                   "critic_disagreement": (q1_next - q2_next).abs().mean().item(),
                   "min_reduction": (q1_next - min_next).mean().item()}
        loss = loss1 + loss2

        # END TODO (4)
        return loss, metrics


    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network

        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        # TODO (5) implement the actor loss. You have to sample from the replay buffer first a set of states.
        states = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
        # L'actor maximitza Q1, no el mínim, a totes les variants (apartats 3.1.2 i 4).
        loss = -self.Critic(states, self.Actor(states)).mean()

        # END TODO (5)

        return loss


    def update(self, batch):
        """Una iteració d'aprenentatge de TD3 sobre un minibatch (figura 5).

        Els dos crítics s'actualitzen sempre. L'actor i els TRES targets només quan el
        comptador d'iteracions del crític és múltiple de policy_delay (3.1.3).

        :param batch: tupla (s [B,obs], a [B,act], r [B], s' [B,obs], terminated [B], truncated [B]).
        :return: dict amb les claus de self.metric_names (actor_loss = None si no toca actor).
        """
        critic_loss, metrics = self.compute_critic_loss(batch)
        self.optim_dqn.zero_grad()
        self.optim_critic2.zero_grad()
        critic_loss.backward()
        self.optim_dqn.step()
        self.optim_critic2.step()
        self.critic_updates += 1

        metrics["actor_loss"] = None
        if self.critic_updates % self.policy_delay == 0:
            # Els gradients que deixa al crític es netegen al següent zero_grad del crític.
            actor_loss = self.compute_actor_loss(batch)
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()
            soft_update(self.Critic_target, self.Critic, self.tau)
            soft_update(self.Critic2_target, self.Critic2, self.tau)
            soft_update(self.Actor_target, self.Actor, self.tau)
            self.actor_updates += 1
            metrics["actor_loss"] = actor_loss.item()
        return metrics


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

    td3 = TD3(env,replay_size=1000000, batch_size=64, gamma=0.99)

    td3.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array'),'video')
    video_agent(env, td3,n_episodes=5)
    pass
