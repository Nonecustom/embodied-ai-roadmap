# 1. 导入库
import gymnasium as gym
import mujoco

# 2. 定义环境名称
ENV_NAME = "InvertedPendulum-v5"

# 3. reset 环境
def reset_env(env):
    observation,info=env.reset()
    return observation,info

# 4. 随机采样连续 action
def sample_action(env):
    action=env.action_space.sample()
    return action

# 5. step 环境
def step_env(env,action):
    next_observation,reward,terminated,truncated,info=env.step(action)
    return next_observation,reward,terminated,truncated,info

# 6. 打印 MuJoCo 版本和空间信息
def print_mujoco(env):
    print(f"mujoco版本：{mujoco.__version__}")
    print(f"观测空间：{env.observation_space}")
    print(f"动作空间：{env.action_space}")

# 7. 打印 step 结果
def print_step_info(observation,action,next_observation,reward,terminated,truncated):
    print(f"observation:{observation}")
    print(f"action:{action}")
    print(f"next_observation:{next_observation}")
    print(f"reward:{reward}")
    print(f"terminated:{terminated}")
    print(f"truncated:{truncated}")

# 8. main() 串起流程、
def main():
    env=gym.make(ENV_NAME)
    observation,info=reset_env(env)
    action=sample_action(env)
    next_observation,reward,terminated,truncated,info=step_env(env,action)
    print_mujoco(env)
    print_step_info(
        observation=observation,
        action=action,
        next_observation=next_observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated
    )
    env.close()

if __name__=="__main__":
    main()
