# 1. 导入 gymnasium 和 mani_skill.envs
import gymnasium as gym
import mani_skill.envs

# 2. 定义 ENV_NAME = "PickCube-v1"
ENV_NAME="PickCube-v1"

# 3. reset 环境
def reset_env(env):
    observation,info=env.reset()
    return observation,info

# 4. 随机采样 8 维 action
def sample_action(env):
    action=env.action_space.sample()
    return action

# 5. step 环境
def step_env(env,action):
    next_observation,reward,terminated,truncated,info=env.step(action)
    return next_observation,reward,terminated,truncated,info

# 6. 打印 observation/action/reward/terminated/truncated 的 type 和内容
def print_object_info(name,obj):
    print(f"{name} type:{type(obj)}")
    print(f"{name}:{obj}")

def main():
    env=gym.make(ENV_NAME)

    ##打印观测空间和动作空间
    print(f"观测空间：{env.observation_space}")
    print(f"动作空间：{env.action_space}")
    observation,info=reset_env(env)
    action=sample_action(env)
    next_observation,reward,terminated,truncated,info=step_env(env,action)
    print_object_info("observation",observation)
    print_object_info("next_observation",next_observation)
    print_object_info("reward",reward)
    print_object_info("terminated",terminated)
    print_object_info("truncated",truncated)

    env.close()

if __name__=="__main__":
    main()
