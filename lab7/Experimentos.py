# Función: comparación, ablaciones, mediciones y checkpoints; apartados 2, 4 y 5.
# TODO (4, 5): ejecutar presupuestos fijos y conservar todas las mediciones.
#______________________________
#PROPUESTA: infraestructura experimental; el PDF no proporciona su pseudocódigo.
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys

import gymnasium as gym
import numpy as np
import torch

from ActorCriticNetworks import soft_update
from DDPG import DDPG
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from TD3 import TD3, VARIANTS


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = 'LunarLander-v3'


def lab_path(path):
    # Todos los artefactos se limitan al directorio autorizado Lab7.
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('La ruta debe estar dentro de Lab7.')
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def update_ddpg(agent, batch):
    # Figura 1: reproduce los TODO (6) y (7) sin cambiar el bucle original.
    agent.optim_dqn.zero_grad(set_to_none=True)
    critic_loss = agent.compute_critic_loss(batch)
    critic_loss.backward()
    agent.optim_dqn.step()
    agent.Critic.requires_grad_(False)
    try:
        agent.optim_actor.zero_grad(set_to_none=True)
        actor_loss = agent.compute_actor_loss(batch)
        actor_loss.backward()
        agent.optim_actor.step()
    finally:
        agent.Critic.requires_grad_(True)
    soft_update(agent.Critic_target, agent.Critic, 0.001)
    soft_update(agent.Actor_target, agent.Actor, 0.001)
    agent.update_count = getattr(agent, 'update_count', 0) + 1
    agent.actor_update_count = getattr(agent, 'actor_update_count', 0) + 1
    return {'critic1_loss': critic_loss.item(), 'critic2_loss': None,
            'actor_loss': actor_loss.item()}


def evaluate(agent, env, episodes, seed):
    #PROPUESTA: cinco episodios con semillas fijas, entorno separado y sin ruido.
    # Se restaura el RNG para que evaluar no altere la trayectoria de entrenamiento.
    saved_rng = random_state()
    returns = []
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed + index)
            total = 0.0
            while True:
                obs, reward, terminated, truncated, _ = env.step(agent.choose_action(obs))
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        restore_random_state(saved_rng)
    return returns


def save_checkpoint(path, agent, training_state):
    # Apartado 4: redes online/objetivo, optimizadores, replay, contador y todos los RNG.
    #PROPUESTA: guardar los atributos del buffer aquí conserva la clase original de Lab5.
    # torch.save serializa su contenido antes de continuar el entrenamiento.
    names = ('Actor', 'Actor_target', 'Critic', 'Critic_target', 'Critic2', 'Critic2_target')
    optimizers = ('optim_actor', 'optim_dqn', 'optim_critic2')
    payload = {
        'networks': {name: getattr(agent, name).state_dict() for name in names if hasattr(agent, name)},
        'optimizers': {name: getattr(agent, name).state_dict() for name in optimizers if hasattr(agent, name)},
        'replay': {'capacity': agent.replay_buffer.capacity,
                   'position': agent.replay_buffer.position,
                   'buffer': agent.replay_buffer.buffer},
        'update_count': getattr(agent, 'update_count', 0),
        'actor_update_count': getattr(agent, 'actor_update_count', 0),
        'rng': random_state(), 'training': copy.deepcopy(training_state),
        'environment_rng': copy.deepcopy(agent.env.unwrapped.np_random.bit_generator.state),
        'action_space_rng': copy.deepcopy(agent.env.action_space.np_random.bit_generator.state),
        'variant': getattr(agent, 'variant', 'ddpg'),
        'hyperparameters': {'batch_size': agent.batch_size, 'gamma': agent.gamma},
    }
    torch.save(payload, lab_path(path))


def load_checkpoint(path, agent):
    # Solo se cargan checkpoints propios generados por este ejecutor.
    payload = torch.load(lab_path(path), map_location=next(agent.Actor.parameters()).device,
                         weights_only=False)
    expected = {'batch_size': agent.batch_size, 'gamma': agent.gamma}
    if payload['hyperparameters'] != expected:
        raise ValueError('El checkpoint tiene otros hiperparámetros.')
    for name, state in payload['networks'].items():
        getattr(agent, name).load_state_dict(state)
    for name, state in payload['optimizers'].items():
        getattr(agent, name).load_state_dict(state)
    # Apartado 4: cada torch.load crea los datos de una rama independiente.
    # Se restauran directamente los tres atributos de la implementación de Lab5.
    agent.replay_buffer.capacity = payload['replay']['capacity']
    agent.replay_buffer.position = payload['replay']['position']
    agent.replay_buffer.buffer = payload['replay']['buffer']
    agent.update_count = payload['update_count']
    agent.actor_update_count = payload['actor_update_count']
    agent.env.unwrapped.np_random.bit_generator.state = payload['environment_rng']
    agent.env.action_space.np_random.bit_generator.state = payload['action_space_rng']
    restore_random_state(payload['rng'])
    # Cada carga crea arrays y tensores independientes para una posible bifurcación.
    return copy.deepcopy(payload['training'])


def write_csv(path, rows):
    if rows:
        with lab_path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def measure_mechanisms(agent):
    #PROPUESTA: diagnósticos sobre hasta 256 transiciones, sin alterar ningún RNG.
    # La reducción del objetivo y el desacuerdo no miden su sesgo respecto al retorno real.
    metrics = {'target_min_reduction': None, 'target_action_perturbation': None,
               'target_critic_disagreement': None}
    if not isinstance(agent, TD3) or not len(agent.replay_buffer):
        return metrics
    saved_rng = random_state()
    try:
        batch = agent.replay_buffer.get(min(256, len(agent.replay_buffer)))
        target_device = next(agent.Actor.parameters()).device
        states = torch.as_tensor(batch[3], dtype=torch.float32, device=target_device)
        with torch.no_grad():
            actions = agent.Actor_target(states)
            noise = (torch.randn_like(actions) * agent.target_noise).clamp(-0.5, 0.5)
            smoothed = (actions + noise).clamp(-1.0, 1.0)
            q1 = agent.Critic_target(states, smoothed)
            q2 = agent.Critic2_target(states, smoothed)
            metrics['target_min_reduction'] = ((q1 - torch.minimum(q1, q2)).mean().item()
                                               if agent.use_minimum else 0.0)
            metrics['target_action_perturbation'] = (smoothed - actions).abs().mean().item()
            metrics['target_critic_disagreement'] = (q1 - q2).abs().mean().item()
    finally:
        restore_random_state(saved_rng)
    return metrics


def train(agent, timesteps=300000, seed=0, output=None, eval_every=1000,
          eval_episodes=5, resume=None, eval_env=None):
    # El presupuesto cuenta SOLO transiciones de entrenamiento, nunca evaluaciones.
    if timesteps <= 0 or eval_every <= 0 or eval_episodes <= 0:
        raise ValueError('Los presupuestos y frecuencias deben ser positivos.')
    variant = getattr(agent, 'variant', 'ddpg')
    folder = lab_path(output or Path('resultados') / f'{variant}_seed{seed}')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'evaluaciones.csv').exists():
        raise FileExistsError(f'Ya hay mediciones en {folder}; elige otra salida.')
    #PROPUESTA: sigma de exploración TD3=0.15, fija en las cuatro variantes.
    noise = (NormalActionNoise(np.zeros(agent.act_dim), 0.15) if isinstance(agent, TD3)
             else OrnsteinUhlenbeckActionNoise(np.zeros(agent.act_dim)))
    own_eval_env = eval_env is None
    if own_eval_env:
        eval_env = gym.make(ENVIRONMENT, continuous=True)
    state = {'step': 0, 'episode': 0, 'seed': seed, 'evaluations': [], 'episodes': [],
             'checkpoint_saved': False, 'eval_every': eval_every, 'eval_episodes': eval_episodes}
    if resume:
        state = load_checkpoint(resume, agent)
        if (state['seed'], state['eval_every'], state['eval_episodes']) != (seed, eval_every, eval_episodes):
            raise ValueError('La semilla o evaluación difieren del checkpoint.')
        if state['step'] >= timesteps:
            raise ValueError('El presupuesto debe superar el paso del checkpoint.')
    metadata = {
        'algorithm': variant, 'seed': seed, 'training_budget': timesteps,
        'evaluation_interval': eval_every, 'evaluation_episodes': eval_episodes,
        'evaluation_seed': 1000000 + seed * 1000, 'episode_seed_rule': 'seed + episode',
        'resume': None if resume is None else str(lab_path(resume)),
        'branch_step': state['step'],
        'environment': agent.env.spec.id if agent.env.spec else type(agent.env).__name__, 'continuous': True,
        'batch_size': agent.batch_size, 'replay_capacity': agent.replay_buffer.capacity,
        'gamma': agent.gamma, 'tau': 0.001, 'actor_lr': 0.0001,
        'critic_lr': 0.001, 'critic_weight_decay': 0.01,
        'architecture': '400-300; LayerNorm; actor tanh; suma estado/accion en critico',
        'exploration': 'Gaussian' if isinstance(agent, TD3) else 'Ornstein-Uhlenbeck',
        'exploration_sigma': 0.15, 'ou_theta': 0.2, 'ou_dt': 0.01,
        'switches': VARIANTS.get(variant), 'target_noise_clip': 0.5 if isinstance(agent, TD3) else None,
        'device': str(next(agent.Actor.parameters()).device), 'torch': torch.__version__,
        'gymnasium': gym.__version__, 'numpy': np.__version__, 'python': sys.version.split()[0],
        'torch_threads': torch.get_num_threads(), 'completed': False,
    }
    metadata_path = folder / 'configuracion.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    obs, _ = agent.env.reset(seed=seed + state['episode'])
    agent.env.action_space.seed(seed)
    episode_return, episode_length = 0.0, 0
    losses = {'critic1_loss': None, 'critic2_loss': None, 'actor_loss': None}
    try:
        for step in range(state['step'] + 1, timesteps + 1):
            action = np.clip(agent.choose_action(obs) + noise.sample(), -1.0, 1.0)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            agent.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1
            if len(agent.replay_buffer) > agent.batch_size:
                batch = agent.replay_buffer.get(agent.batch_size)
                losses = agent.update(batch) if isinstance(agent, TD3) else update_ddpg(agent, batch)
            state['step'] = step
            if step % eval_every == 0 or step == timesteps:
                returns = evaluate(agent, eval_env, eval_episodes, metadata['evaluation_seed'])
                row = {'transitions': step, 'mean_return': float(np.mean(returns)),
                       'std_return': float(np.std(returns)),
                       'evaluation_returns': json.dumps(returns),
                       'critic_updates': getattr(agent, 'update_count', 0),
                       'actor_updates': getattr(agent, 'actor_update_count', 0),
                       **losses, **measure_mechanisms(agent)}
                state['evaluations'].append(row)
                write_csv(folder / 'evaluaciones.csv', state['evaluations'])
                print(f'{variant} semilla={seed} pasos={step}/{timesteps} retorno={row["mean_return"]:.2f}', flush=True)
            if terminated or truncated:
                state['episode'] += 1
                state['episodes'].append({'episode': state['episode'], 'transitions': step,
                                          'return': episode_return, 'length': episode_length,
                                          'terminated': bool(terminated), 'truncated': bool(truncated)})
                write_csv(folder / 'episodios.csv', state['episodes'])
                if hasattr(noise, 'reset'):
                    noise.reset()
                # Se guarda en frontera de episodio: basta reconstruir el siguiente reset.
                if variant == 'full' and step >= timesteps // 3 and not state['checkpoint_saved']:
                    state['checkpoint_saved'] = True
                    save_checkpoint(folder / 'checkpoint_tercio.pt', agent, state)
                obs, _ = agent.env.reset(seed=seed + state['episode'])
                episode_return, episode_length = 0.0, 0
        metadata['completed'] = True
        metadata['actual_transitions'] = state['step']
        metadata['critic_updates'] = getattr(agent, 'update_count', 0)
        metadata['actor_updates'] = getattr(agent, 'actor_update_count', 0)
        metadata['checkpoint_saved'] = state['checkpoint_saved']
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        return state
    finally:
        if own_eval_env:
            eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Apartados 2-4: presupuestos fijos y cinco semillas.')
    parser.add_argument('--variant', choices=['all', 'ddpg', *VARIANTS], default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', default='resultados')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.resume and (args.variant in ('all', 'ddpg') or len(args.seeds) != 1):
        parser.error('--resume requiere una variante TD3 concreta y una semilla.')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Las semillas deben ser distintas y no negativas.')
    torch.set_num_threads(args.threads)
    variants = ['ddpg', *VARIANTS] if args.variant == 'all' else [args.variant]
    for variant in variants:
        # El apartado 2 pide una semilla por algoritmo para la comparación exploratoria.
        seeds = args.seeds[:1] if variant == 'ddpg' else args.seeds
        for seed in seeds:
            seed_everything(seed)
            env = gym.make(ENVIRONMENT, continuous=True)
            try:
                agent = DDPG(env, batch_size=64) if variant == 'ddpg' else TD3(env, variant=variant)
                train(agent, args.steps, seed, Path(args.output) / f'{variant}_seed{seed}',
                      args.eval_every, args.eval_episodes, args.resume)
            finally:
                env.close()


if __name__ == '__main__':
    main()
#______________________________
# END TODO (4, 5)
