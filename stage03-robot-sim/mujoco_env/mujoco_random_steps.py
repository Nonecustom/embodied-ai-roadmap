# 任务要求：

# 1. 创建 InvertedPendulum-v5 环境
# 2. reset 环境
# 3. 连续运行 10 步
# 4. 每一步随机采样 action
# 5. 执行 env.step(action)
# 6. 打印 step 编号、action、reward、terminated、truncated
# 7. 如果 terminated 或 truncated 为 True，提前 break
# 8. env.close()
import gymnasium as gym 
import mujoco

ENV_NAME="InvertedPendulum-v5"

def reset_env(env):
    observation,info=env.reset()
    return observation,info

def sample_action(env):
    action=env.action_space.sample()
    return action

def step_env(env,action):
    next_observation,reward,terminated,truncated,info=env.step(action)
    return next_observation,reward,terminated,truncated,info

def main():
    env=gym.make(ENV_NAME)
    observation,info=reset_env(env)
    for index in range(10):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=step_env(env,action)
        print(f"step 0{index+1}")
        print(f"action:{action}")
        print(f"reward:{reward}")
        print(f"terminated:{terminated}")
        print(f"truncated:{truncated}")
        print(f" ")

        if terminated or truncated:
            break

if __name__=="__main__":
    main()