# 要求：

# 1. 创建 InvertedPendulum-v5
# 2. reset
# 3. total_reward = 0
# 4. step_count = 0
# 5. while not (terminated or truncated)
# 6. 每一步随机 action
# 7. 累加 total_reward
# 8. 累加 step_count
# 9. 结束后打印 total_reward、step_count、terminated、truncated
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
    total_reward=0.0
    step_count=0
    env=gym.make(ENV_NAME)
    observation,info=reset_env(env)
    terminated=False
    truncated=False
    while not (terminated or truncated):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=step_env(env,action)
        total_reward+=reward
        step_count+=1
    
    print(f"total_reward:{total_reward}")
    print(f"step_count:{step_count}")
    print(f"terminated:{terminated}")
    print(f"truncated:{truncated}")

if __name__=="__main__":
    main()
