# 要求：

# 1. 跑 5 个 episode
# 2. 每个 episode 记录 total_reward 和 step_count
# 3. 打印每轮结果
# 4. 计算 average_reward
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

def run_one_episode(env):
    episode_reward=0.0
    episode_step_count=0
    observation,info=reset_env(env)
    terminated=False
    truncated=False
    while not (terminated or truncated):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=step_env(env,action)
        episode_reward+=reward
        episode_step_count+=1
    
    return episode_reward,episode_step_count

def main():
    env=gym.make(ENV_NAME)
    total_reward=[]
    for index in range(5):
        episode_reward,episode_step_count=run_one_episode(env)
        print(f"episode {index+1} reward:{episode_reward}, steps:{episode_step_count}")
        total_reward.append(episode_reward)
    
    average_reward=sum(total_reward)/len(total_reward)
    print(f"average reward:{average_reward}")

        


if __name__=="__main__":
    main()