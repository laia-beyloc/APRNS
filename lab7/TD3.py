# Función: adaptación de la solución DDPG a TD3 y sus ablaciones; apartados 3 y 4.
# TODO (3, 4): implementar las tres mejoras de la figura 5 y sus interruptores.
#______________________________
# Se reutiliza DDPG mediante herencia para conservar sus bloques originales.
# Lecture5.pdf, diapositiva 73: mínimo, ruido recortado y actualizaciones retrasadas.
#PROPUESTA: organización de la copia funcional en una subclase, sin duplicar DDPG.
import torch
import torch.nn.functional as F

from ActorCriticNetworks import CriticNetwork, copy_target, soft_update
from DDPG import DDPG


# Apartado 4: cada variante altera exclusivamente el mecanismo indicado.
VARIANTS = {
    'full': {'use_minimum': True, 'target_noise': 0.2, 'policy_delay': 2},
    'no_minimum': {'use_minimum': False, 'target_noise': 0.2, 'policy_delay': 2},
    'no_smoothing': {'use_minimum': True, 'target_noise': 0.0, 'policy_delay': 2},
    'no_delay': {'use_minimum': True, 'target_noise': 0.2, 'policy_delay': 1},
}


class TD3(DDPG):
    def __init__(self, env, replay_size=1000000, batch_size=64, gamma=0.99,
                 variant='full'):
        super().__init__(env, replay_size, batch_size, gamma)
        if variant not in VARIANTS:
            raise ValueError('Variante TD3 desconocida.')
        self.variant = variant
        self.use_minimum = VARIANTS[variant]['use_minimum']
        self.target_noise = VARIANTS[variant]['target_noise']
        self.policy_delay = VARIANTS[variant]['policy_delay']
        self.noise_clip = 0.5
        self.tau = 0.001
        self.update_count = 0
        self.actor_update_count = 0

        # Figura 5: el segundo crítico se inicializa independientemente del primero.
        critic_device = next(self.Critic.parameters()).device
        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(critic_device)
        self.Critic2.initialize_weights()
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(critic_device)
        copy_target(self.Critic2_target, self.Critic2)
        self.Critic2_target.requires_grad_(False)
        self.optim_critic2 = torch.optim.Adam(self.Critic2.parameters(), lr=0.001,
                                              weight_decay=0.01)

    def bellman_target(self, batch):
        # Apartados 3.1.1 y 3.1.2: ruido objetivo recortado y mínimo opcional.
        critic_device = next(self.Critic.parameters()).device
        rewards = torch.as_tensor(batch[2], dtype=torch.float32, device=critic_device).reshape(-1, 1)
        next_states = torch.as_tensor(batch[3], dtype=torch.float32, device=critic_device)
        terminals = torch.as_tensor(batch[4], dtype=torch.float32, device=critic_device).reshape(-1, 1)
        with torch.no_grad():
            next_actions = self.Actor_target(next_states)
            # Se consume el mismo sorteo también sin smoothing para facilitar el control.
            noise = (torch.randn_like(next_actions) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            q1 = self.Critic_target(next_states, next_actions)
            q2 = self.Critic2_target(next_states, next_actions)
            next_q = torch.minimum(q1, q2) if self.use_minimum else q1
            return rewards + self.gamma * (1.0 - terminals) * next_q

    def compute_critic_loss(self, batch):
        # Los dos críticos comparten exactamente el mismo objetivo, incluido su ruido.
        critic_device = next(self.Critic.parameters()).device
        states = torch.as_tensor(batch[0], dtype=torch.float32, device=critic_device)
        actions = torch.as_tensor(batch[1], dtype=torch.float32, device=critic_device)
        target = self.bellman_target(batch)
        return (F.mse_loss(self.Critic(states, actions), target),
                F.mse_loss(self.Critic2(states, actions), target))

    def update(self, batch):
        # Figura 5: ambos críticos aprenden en TODAS las iteraciones de aprendizaje.
        self.optim_dqn.zero_grad(set_to_none=True)
        self.optim_critic2.zero_grad(set_to_none=True)
        loss1, loss2 = self.compute_critic_loss(batch)
        (loss1 + loss2).backward()
        self.optim_dqn.step()
        self.optim_critic2.step()
        self.update_count += 1
        actor_loss = None
        if self.update_count % self.policy_delay == 0:
            # El objetivo heredado del actor utiliza solo Q1, nunca el mínimo.
            self.Critic.requires_grad_(False)
            try:
                self.optim_actor.zero_grad(set_to_none=True)
                actor_loss = self.compute_actor_loss(batch)
                actor_loss.backward()
                self.optim_actor.step()
                self.actor_update_count += 1
            finally:
                self.Critic.requires_grad_(True)
            soft_update(self.Actor_target, self.Actor, self.tau)
            soft_update(self.Critic_target, self.Critic, self.tau)
            soft_update(self.Critic2_target, self.Critic2, self.tau)
        return {'critic1_loss': loss1.item(), 'critic2_loss': loss2.item(),
                'actor_loss': None if actor_loss is None else actor_loss.item()}

    def learn(self, timesteps):
        #PROPUESTA: bucle común con presupuesto fijo, sin la parada anticipada original.
        # La exploración gaussiana y la evaluación separada están en Experimentos.py.
        from Experimentos import train
        return train(self, timesteps=timesteps)
#______________________________
# END TODO (3, 4)
