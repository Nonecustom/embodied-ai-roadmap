# type()：你是谁？
# print()：你长什么样？
# dir()：你有什么？
# help()：你怎么用？
# hasattr()：你有没有这个东西？
# callable()：这个东西能不能被调用？

# 1. 导入库
import gymnasium as gym 

# 2. 定义环境名称
ENV_NAME="CartPole-v1"

# 3. 创建环境
def make_env():
    return gym.make(ENV_NAME)
# 4. reset 环境
def reset_env(env):
    observation,info=env.reset()
    return observation,info

# 5. 随机采样 action
def sample_action(env):
    return env.action_space.sample()

# 6. step 环境
def step_env(env,action):
    next_observation,reward,terminated,truncated,info=env.step(action)
    return next_observation,reward,terminated,truncated,info                
    
# 7. 打印空间信息
def print_space_info(env):
    print(f"观测空间{env.observation_space}")
    print(f"动作空间{env.action_space}")

# 8. 打印 step 结果
def print_step_result(observation,action,next_observation,reward,terminated,truncated):
    print(f"初始位置：{observation}")
    print(f"动作：{action}")
    print(f"新位置：{next_observation}")
    print(f"奖励：{reward}")
    print(f"结束信息：{terminated}")
    print(f"超时信息：{truncated}")


# 9. main() 串起流程
def main():
    env=make_env()
    print_space_info(env)
    observation,info=reset_env(env)
    action=sample_action(env)
    next_observation,reward,terminated,truncated,info=step_env(env,action)
    print_step_result(observation,action,next_observation,reward,terminated,truncated)
    env.close()

if __name__=="__main__":
    main()

