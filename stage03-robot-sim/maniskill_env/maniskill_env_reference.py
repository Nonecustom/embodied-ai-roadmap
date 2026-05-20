import gymnasium as gym
import mani_skill.envs


ENV_NAME = "PickCube-v1"


def make_env():
    return gym.make(ENV_NAME, obs_mode="state", render_mode=None)       #？这几个参数的作用


def reset_env(env):
    observation, info = env.reset()
    return observation, info


def sample_action(env):
    return env.action_space.sample()


def step_env(env, action):
    next_observation, reward, terminated, truncated, info = env.step(action)
    return next_observation, reward, terminated, truncated, info


def print_space_info(env):
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")


def print_object_info(name, obj):
    print(f"{name} type: {type(obj)}")
    print(f"{name}: {obj}")


def main():
    env = make_env()

    print_space_info(env)

    observation, info = reset_env(env)
    action = sample_action(env)
    next_observation, reward, terminated, truncated, info = step_env(env, action)

    print_object_info("observation", observation)
    print_object_info("action", action)
    print_object_info("next observation", next_observation)
    print_object_info("reward", reward)
    print_object_info("terminated", terminated)
    print_object_info("truncated", truncated)

    env.close()


if __name__ == "__main__":
    main()
