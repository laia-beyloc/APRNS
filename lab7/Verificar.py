# Función: validación de ecuaciones, ablaciones y reanudación; apartados 2, 3 y 4.
# TODO (validación): comprobar propiedades matemáticas y reproducibilidad del ejecutor.
#______________________________
#PROPUESTA: pruebas con un entorno mínimo; no equivalen a entrenar LunarLander.
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch

from DDPG import DDPG
from Lab7.APRNS.Diagnosticos import generate, read_runs
from Experimentos import (ROOT, evaluate, lab_path, load_checkpoint, measure_mechanisms,
                          save_checkpoint, seed_everything, train, write_csv)
from ReplayBuffer import ReplayBuffer
from TD3 import TD3, VARIANTS


class EntornoMinimo(gym.Env):
    # Cuatro pasos por episodio permiten comprobar fronteras y presupuestos exactos.
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, (8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.np_random.normal(size=8).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        return (self.np_random.normal(size=8).astype(np.float32),
                float(1 - np.square(action).mean()), False, self.steps == 4, {})


def batch():
    return (np.ones((3, 8), np.float32), np.zeros((3, 2), np.float32),
            np.ones(3, np.float32), np.zeros((3, 8), np.float32),
            np.array([False, True, False]), np.array([False, False, True]))


def constant(network, value=0.0):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        if hasattr(network, 'l4'):
            network.l4.bias.fill_(value)


class Comprobaciones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Las comparaciones bit a bit se realizan en CPU, también desde Colab.
        cls.cuda_patch = patch('torch.cuda.is_available', return_value=False)
        cls.device_patch = patch('DDPG.device', torch.device('cpu'))
        cls.cuda_patch.start()
        cls.device_patch.start()
        cls.temporary = tempfile.TemporaryDirectory(prefix='verificacion_', dir=ROOT)
        cls.folder = Path(cls.temporary.name).resolve()
        # La limpieza recursiva solo afecta al temporal comprobado dentro de Lab7.
        assert cls.folder.is_relative_to(ROOT) and cls.folder != ROOT

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        cls.device_patch.stop()
        cls.cuda_patch.stop()

    def setUp(self):
        seed_everything(7)

    def test_replay_circular_y_referencias_lab5(self):
        replay = ReplayBuffer(2)
        state = np.array([1, 2], np.float32)
        replay.put(state, np.zeros(2), 1, state, False, True)
        state[:] = 99
        # Lab5 guarda referencias: la modificación de entrada también queda almacenada.
        self.assertEqual(replay.get(1)[0][0, 0], 99)
        for value in (2, 3):
            replay.put(np.full(2, value), np.zeros(2), value, np.zeros(2), True, False)
        self.assertEqual(len(replay), 2)
        self.assertEqual(set(replay.get(2)[2]), {2, 3})
        self.assertEqual(replay.position, 1)
        # Se conserva el contrato de Lab5: pedir más muestras devuelve las disponibles.
        self.assertEqual(len(replay.get(3)[0]), 2)

    def test_replay_get_lab5_con_acciones_continuas(self):
        replay = ReplayBuffer(4)
        for index in range(3):
            replay.put(np.full(8, index, dtype=np.float32),
                       np.array([0.1 * index, -0.1 * index]), index,
                       np.full(8, index + 1, dtype=np.float32), index == 1, index == 2)
        actual = replay.get(10)
        self.assertEqual(len(actual), 6)
        self.assertEqual(actual[0].shape, (3, 8))
        self.assertEqual(actual[1].shape, (3, 2))
        for reward, terminated, truncated in zip(actual[2], actual[4], actual[5]):
            self.assertEqual(terminated, reward == 1)
            self.assertEqual(truncated, reward == 2)

    def test_inicializacion_y_accion_determinista(self):
        agent = DDPG(EntornoMinimo(), replay_size=20)
        for online, target in ((agent.Actor, agent.Actor_target), (agent.Critic, agent.Critic_target)):
            for source, destination in zip(online.parameters(), target.parameters()):
                self.assertTrue(torch.equal(source, destination))
                self.assertFalse(destination.requires_grad)
        action = agent.choose_action(np.zeros(8, np.float32))
        np.testing.assert_array_equal(action, agent.choose_action(np.zeros(8, np.float32)))
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.abs(action) <= 1))

    def test_bellman_terminacion_y_truncacion(self):
        agent = DDPG(EntornoMinimo(), replay_size=20)
        constant(agent.Critic)
        constant(agent.Critic_target, 2)
        expected = np.mean(np.square([1 + 0.99 * 2, 1, 1 + 0.99 * 2]))
        loss = agent.compute_critic_loss(batch())
        self.assertAlmostEqual(loss.item(), expected, places=5)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in agent.Critic_target.parameters()))

    def test_ddpg_reinicia_exploracion_entre_episodios(self):
        # La diapositiva 72 pide un proceso de exploración nuevo en cada episodio.
        # Un ruido controlado permite detectar si su estado se arrastra al siguiente.
        env = EntornoMinimo()
        recorded_actions = []
        original_step = env.step

        def record_step(action):
            recorded_actions.append(np.asarray(action).copy())
            return original_step(action)

        class RuidoControlado:
            def __init__(self, mu):
                self.shape = mu.shape
                self.reset()

            def reset(self):
                self.value = 0.0

            def sample(self):
                self.value += 0.1
                return np.full(self.shape, self.value)

        env.step = record_step
        agent = DDPG(env, replay_size=20)
        with patch('DDPG.OrnsteinUhlenbeckActionNoise', RuidoControlado), patch(
                'DDPG.episode_reward_plot'), patch.object(
                agent, 'choose_action', Mock(return_value=np.zeros(2))), patch.object(
                agent, 'eval_episodes', return_value=0.0):
            rewards, _ = agent.learn(8)
        self.assertEqual(len(rewards), 2)
        np.testing.assert_allclose(recorded_actions[:4], recorded_actions[4:])
        np.testing.assert_allclose(np.asarray(recorded_actions)[:, 0],
                                   [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])

    def test_td3_minimo_y_ablacion(self):
        agent = TD3(EntornoMinimo(), replay_size=20)
        constant(agent.Critic_target, 5)
        constant(agent.Critic2_target, 2)
        np.testing.assert_allclose(agent.bellman_target(batch()).cpu().numpy().ravel(),
                                   [2.98, 1, 2.98], rtol=1e-6)
        agent.use_minimum = False
        np.testing.assert_allclose(agent.bellman_target(batch()).cpu().numpy().ravel(),
                                   [5.95, 1, 5.95], rtol=1e-6)

    def test_smoothing_recorta_ruido_y_accion(self):
        agent = TD3(EntornoMinimo(), replay_size=20)
        constant(agent.Actor_target)
        captured = []
        hook = agent.Critic_target.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].detach().clone()))
        with patch('torch.randn_like', side_effect=lambda value: torch.full_like(value, 100)):
            agent.bellman_target(batch())
            self.assertTrue(torch.allclose(captured[-1], torch.full_like(captured[-1], 0.5)))
            with torch.no_grad():
                agent.Actor_target.l3.bias.fill_(float(np.arctanh(0.9)))
            agent.bellman_target(batch())
            self.assertTrue(torch.all(captured[-1] == 1))
            agent.target_noise = 0
            agent.bellman_target(batch())
            self.assertTrue(torch.allclose(captured[-1], torch.full_like(captured[-1], 0.9)))
        hook.remove()

    def test_retraso_contado_en_actualizaciones(self):
        for variant in VARIANTS:
            agent = TD3(EntornoMinimo(), replay_size=20, variant=variant)
            actor_before = copy.deepcopy(agent.Actor.state_dict())
            target_before = copy.deepcopy(agent.Critic_target.state_dict())
            second_before = copy.deepcopy(agent.Critic2.state_dict())
            agent.update(batch())
            self.assertTrue(any(not torch.equal(second_before[key], value)
                                for key, value in agent.Critic2.state_dict().items()))
            if variant != 'no_delay':
                self.assertTrue(all(torch.equal(actor_before[key], value)
                                    for key, value in agent.Actor.state_dict().items()))
                self.assertTrue(all(torch.equal(target_before[key], value)
                                    for key, value in agent.Critic_target.state_dict().items()))
            else:
                self.assertEqual(agent.actor_update_count, 1)
            agent.update(batch())
            self.assertEqual(agent.update_count, 2)
            self.assertEqual(agent.actor_update_count, 2 if variant == 'no_delay' else 1)
            self.assertTrue(any(not torch.equal(actor_before[key], value)
                                for key, value in agent.Actor.state_dict().items()))
            for key, value in agent.Critic_target.state_dict().items():
                if variant != 'no_delay':
                    expected = 0.001 * agent.Critic.state_dict()[key] + 0.999 * target_before[key]
                    torch.testing.assert_close(value, expected)

    def test_actor_utiliza_q1_y_transmite_gradiente(self):
        agent = TD3(EntornoMinimo(), replay_size=20)
        loss = agent.compute_actor_loss(batch()).item()
        constant(agent.Critic2, 100000)
        self.assertEqual(loss, agent.compute_actor_loss(batch()).item())
        agent.Critic.requires_grad_(False)
        agent.compute_actor_loss(batch()).backward()
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in agent.Actor.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.Critic.parameters()))

    def test_evaluacion_y_diagnosticos_no_consumen_rng(self):
        agent = TD3(EntornoMinimo(), replay_size=20)
        for index in range(4):
            agent.replay_buffer.put(np.zeros(8), np.zeros(2), index, np.ones(8), False, False)
        seed_everything(91)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        seed_everything(91)
        first = evaluate(agent, EntornoMinimo(), 2, 100)
        metrics = measure_mechanisms(agent)
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(first, evaluate(agent, EntornoMinimo(), 2, 100))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        self.assertGreaterEqual(metrics['target_min_reduction'], 0)

    def test_checkpoint_reproduce_siguiente_actualizacion(self):
        agent = TD3(EntornoMinimo(), replay_size=20, batch_size=2)
        for index in range(4):
            agent.replay_buffer.put(np.full(8, index), np.zeros(2), index, np.ones(8), False, False)
        agent.update(agent.replay_buffer.get(2))
        checkpoint = self.folder / 'estado.pt'
        save_checkpoint(checkpoint, agent, {'step': 4})
        agent.update(agent.replay_buffer.get(2))
        expected_random = (random.random(), np.random.rand(), torch.rand(1))
        restored = TD3(EntornoMinimo(), replay_size=20, batch_size=2)
        self.assertEqual(load_checkpoint(checkpoint, restored), {'step': 4})
        restored.update(restored.replay_buffer.get(2))
        actual_random = (random.random(), np.random.rand(), torch.rand(1))
        for name in ('Actor', 'Actor_target', 'Critic', 'Critic_target', 'Critic2', 'Critic2_target'):
            for expected, actual in zip(getattr(agent, name).parameters(), getattr(restored, name).parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(expected_random[:2], actual_random[:2])
        torch.testing.assert_close(expected_random[2], actual_random[2], rtol=0, atol=0)
        # Las cargas para las ablaciones mantienen memorias independientes sin ampliar Lab5.
        restored.replay_buffer.buffer[0][0][0] = -100
        self.assertNotEqual(restored.replay_buffer.buffer[0][0][0],
                            agent.replay_buffer.buffer[0][0][0])

    def test_ejecutor_presupuesto_variantes_y_reanudacion(self):
        full, full_state = None, None
        for variant in ('ddpg', *VARIANTS):
            seed_everything(7)
            agent = (DDPG(EntornoMinimo(), replay_size=40, batch_size=2) if variant == 'ddpg'
                     else TD3(EntornoMinimo(), replay_size=40, batch_size=2, variant=variant))
            state = train(agent, 12, 7, self.folder / variant, 4, 2, eval_env=EntornoMinimo())
            self.assertEqual(state['step'], 12)
            self.assertEqual([row['transitions'] for row in state['evaluations']], [4, 8, 12])
            self.assertEqual(agent.update_count, 10)
            self.assertEqual(agent.actor_update_count, 10 if variant in ('ddpg', 'no_delay') else 5)
            if variant == 'full':
                full, full_state = agent, state
        restored = TD3(EntornoMinimo(), replay_size=40, batch_size=2)
        state = train(restored, 12, 7, self.folder / 'reanudada', 4, 2,
                      resume=self.folder / 'full' / 'checkpoint_tercio.pt', eval_env=EntornoMinimo())
        self.assertEqual(state['evaluations'], full_state['evaluations'])
        for expected, actual in zip(full.Actor.parameters(), restored.Actor.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_informe_rechaza_datos_ausentes_y_rutas_externas(self):
        with self.assertRaisesRegex(ValueError, 'Faltan mediciones'):
            read_runs([self.folder / 'sin_mediciones'], [0, 1, 2, 3, 4], 300000)
        with self.assertRaises(ValueError):
            lab_path(ROOT.parent / 'fuera')

    def test_figuras_e_informe_con_fixture_temporal(self):
        # Datos sintéticos exclusivos de esta prueba; se eliminan al terminar.
        # No se guardan en resultados/ ni se presentan como entrenamiento real.
        destination = self.folder / 'diagnosticos'
        destination.mkdir()

        def temporary_path(path):
            path = Path(path)
            return lab_path(path if path.is_absolute() else destination / path)

        for variant in ('ddpg', *VARIANTS):
            for seed in ([0] if variant == 'ddpg' else range(5)):
                folder = destination / 'entradas' / f'{variant}_seed{seed}'
                folder.mkdir(parents=True)
                config = {'algorithm': variant, 'seed': seed, 'completed': True,
                          'environment': 'LunarLander-v3', 'training_budget': 1000,
                          'actual_transitions': 1000, 'evaluation_interval': 1000,
                          'evaluation_episodes': 5, 'branch_step': 0,
                          'critic_updates': 936, 'actor_updates': 936 if variant in ('ddpg', 'no_delay') else 468,
                          'batch_size': 64, 'replay_capacity': 1000000, 'gamma': 0.99,
                          'tau': 0.001, 'actor_lr': 0.0001, 'critic_lr': 0.001,
                          'critic_weight_decay': 0.01, 'architecture': 'fixture',
                          'exploration': 'Gaussian', 'exploration_sigma': 0.15,
                          'target_noise_clip': 0.5, 'switches': VARIANTS.get(variant)}
                (folder / 'configuracion.json').write_text(json.dumps(config), encoding='utf-8')
                write_csv(folder / 'evaluaciones.csv', [{
                    'transitions': 1000, 'mean_return': 10 + seed,
                    'target_min_reduction': 0.0 if variant == 'no_minimum' else 0.1,
                    'target_action_perturbation': 0.0 if variant == 'no_smoothing' else 0.15,
                    'target_critic_disagreement': 0.2}])
        with patch('Diagnosticos.lab_path', side_effect=temporary_path), patch(
                'Diagnosticos.write_csv', side_effect=lambda path, rows: write_csv(temporary_path(path), rows)):
            generate(['entradas'], list(range(5)), 1000)
        report = (destination / 'Informe.md').read_text(encoding='utf-8')
        self.assertIn('12.0', report)
        self.assertTrue((destination / 'graficas' / 'comparacion.png').is_file())
        self.assertTrue((destination / 'graficas' / 'ablaciones.png').is_file())
        self.assertTrue((destination / 'resumen_mediciones.csv').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
#______________________________
# END TODO (validación)
