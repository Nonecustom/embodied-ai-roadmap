# 要求
# 1. 创建 CartPole-v1
# 2. reset
# 3. total_reward = 0
# 4. step_count = 0
# 5. while 循环直到结束
# 6. 每一步随机 action
# 7. 累加 total_reward
# 8. 结束后打印总步数和总奖励
# 9. env.close()

import gymnasium as gym

ENV_NAME="CartPole-v1"

def reset_env(env):
    observation,info=env.reset()
    return observation,info

def sample_action(env):
    action=env.action_space.sample()
    return action

def get_step(env,action):
    next_observation,reward,terminated,truncated,info=env.step(action)
    return next_observation,reward,terminated,truncated,info

def main():
    env=gym.make(ENV_NAME)
    total_reward=0.0
    total_episode=0
    observation,info=reset_env(env)
    terminated=False
    truncated=False
    while not (terminated or truncated):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=get_step(env,action)
        total_reward+=reward
        total_episode+=1
    
    print(f"total_reward:{total_reward}")
    print(f"total_episode:{total_episode}")
    env.close()

if __name__=="__main__":
    main()
