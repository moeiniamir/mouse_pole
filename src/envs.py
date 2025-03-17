import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from sklearn.mixture import GaussianMixture


class MouseFollowingCartPole(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, max_episode_steps: int, render_mode: Optional[str] = None
    ):
        # self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 50.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.pole_friction = 0.1  # friction coefficient for the pole

        # Angle at which to fail the episode
        self.theta_threshold_radians = 45 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians,
                np.inf,
                self.x_threshold
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None
        self.mouse_x_traj = None
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps 

        # self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot, _ = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp - self.pole_friction * theta_dot / (self.polemass_length)) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array((x, x_dot, theta, theta_dot, self.mouse_x_traj[self._elapsed_steps]), dtype=np.float64)
        self._elapsed_steps += 1

        # terminated = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        # )

        if -self.x_threshold <= self.state[0] <= self.x_threshold:
            if np.abs(self.state[2]) < self.theta_threshold_radians:
                reward = np.exp(-np.abs(self.state[0] - self.state[-1])*2)
            else:
                reward = 0
        else:
            reward = -np.minimum(np.square(x - self.x_threshold), np.square(x + self.x_threshold))
            reward = max(reward, -1)  # Clip the negative reward at -1

        if self.render_mode == "human":
            self.render()

        truncated = self._elapsed_steps >= self._max_episode_steps
        return np.array(self.state, dtype=np.float32), reward, False, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, 0, 0  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))

        min_mouse_x = self.observation_space.low[-1]
        max_mouse_x = self.observation_space.high[-1]
        # Generate a trajectory of mouse positions for the entire episode
        # self.mouse_x_traj = np.full(self._max_episode_steps, 
                                    # np.random.uniform(self.observation_space.low[-1], self.observation_space.high[-1]))
        mean_1 = np.random.uniform(min_mouse_x, max_mouse_x)
        mean_2 = np.random.uniform(min_mouse_x, max_mouse_x)
        first_point = np.clip(np.random.normal(mean_1, 1), min_mouse_x, max_mouse_x)
        self.mouse_x_traj = [first_point]
        while len(self.mouse_x_traj) < self._max_episode_steps:
            if np.random.rand() < 0.25:
                sleep_time = np.random.uniform(5, 15)/100 * self._max_episode_steps
                self.mouse_x_traj.extend([self.mouse_x_traj[-1]] * int(sleep_time))
            else:
                speed = np.random.beta(0.5, 2) * 0.95 + 0.05  # beta(0.5, 1) scaled to [0.05, 1.0]
                choice = np.random.choice([mean_1, mean_2])
                next_point = np.clip(np.random.normal(choice, 1), min_mouse_x, max_mouse_x)
                interp_points = np.linspace(self.mouse_x_traj[-1], next_point, 
                                            num=int(np.abs(self.mouse_x_traj[-1] - next_point) / speed), endpoint=False)
                self.mouse_x_traj.extend(interp_points)
        self.mouse_x_traj = self.mouse_x_traj[:self._max_episode_steps]

        self.state = np.append(self.state, self.mouse_x_traj[0])
        self._elapsed_steps = 0

        # self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # Draw the mouse position as a bold red dot
        mousex = x[4] * scale + self.screen_width / 2.0
        mousey = carty  # Same height as the cart
        mouse_radius = 10  # Larger radius for a bold dot
        gfxdraw.aacircle(
            self.surf,
            int(mousex),
            int(mousey),
            mouse_radius,
            (255, 0, 0),  # Red color
        )
        gfxdraw.filled_circle(
            self.surf,
            int(mousex),
            int(mousey),
            mouse_radius,
            (255, 0, 0),  # Red color
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False