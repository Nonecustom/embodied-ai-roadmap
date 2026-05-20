# 要求：
# 1. 跑 5 个 episode
# 2. 每轮记录 total_reward
# 3. 每轮记录 step_count
# 4. 每轮记录 terminated / truncated
# 5. 计算 average_reward
import gymnasium as gym
import torch
import mani_skill.envs

ENV_NAME="PickCube-v1"

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
    env=gym.make(ENV_NAME,obs_mode="state",render_mode=None)
    episode_reward=[]
    episode_steps=[]
    for index in range(5):
        observation,info=reset_env(env)          #k 必须要在循环内，每轮重置环境状态
        total_reward=0.0
        step_count=0
        terminated=torch.tensor(False)          
        truncated=torch.tensor(False)
        while not (terminated.item() or truncated.item()):
            action=sample_action(env)
            next_observation,reward,terminated,truncated,info=step_env(env,action)
            total_reward+=reward.item()
            step_count+=1

        episode_reward.append(total_reward)
        episode_steps.append(step_count)
        print(f"episode {index+1} reward:{total_reward},steps:{step_count},terminated:{terminated.item()},truncated:{truncated.item()}")
    
    average_reward=sum(episode_reward)/len(episode_reward)
    print(f"average reward:{average_reward}")

if __name__=="__main__":
    main()