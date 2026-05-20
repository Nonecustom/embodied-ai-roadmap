import gymnasium as gym
import mujoco


ENV_NAME = "InvertedPendulum-v5"                        #？了解这个环境与Catrpolev1,并了解两者的异同点


def make_env():
    return gym.make(ENV_NAME)


def reset_env(env):
    observation, info = env.reset()
    return observation, info


def sample_action(env):
    return env.action_space.sample()


def step_env(env, action):
    next_observation, reward, terminated, truncated, info = env.step(action)
    return next_observation, reward, terminated, truncated, info


def print_space_info(env):
    print(f"mujoco version: {mujoco.__version__}")
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")


def print_step_result(observation, action, next_observation, reward, terminated, truncated):
    print(f"observation: {observation}")
    print(f"action: {action}")
    print(f"next observation: {next_observation}")
    print(f"reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")


def main():
    env = make_env()

    print_space_info(env)

    observation, info = reset_env(env)
    action = sample_action(env)
    next_observation, reward, terminated, truncated, info = step_env(env, action)

    print_step_result(
        observation=observation,
        action=action,
        next_observation=next_observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )

    env.close()


if __name__ == "__main__":
    main()
