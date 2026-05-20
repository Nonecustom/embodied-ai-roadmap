# 任务要求：

# 1. 创建 PickCube-v1 环境
# 2. 设置 render_mode="human"
# 3. reset
# 4. 随机运行 200 步
# 5. 每一步 sample action
# 6. 执行 env.step(action)
# 7. 加 time.sleep(0.03)
# 8. 如果 terminated 或 truncated，重新 reset
# 9. env.close()
import gymnasium as gym
import mani_skill.envs
import torch
import time 

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
    env=gym.make(ENV_NAME,obs_mode="state",render_mode="human")
    episode_reward=[]
    episode_steps=[]
    for index in range(200):
        observation,info=reset_env(env)         
        env.render()
        time.sleep(1)
        total_reward=0.0
        step_count=0
        terminated=torch.tensor(False)          
        truncated=torch.tensor(False)
        while not (terminated.item() or truncated.item()):
            action=sample_action(env)
            next_observation,reward,terminated,truncated,info=step_env(env,action)
            env.render()
            time.sleep(0.03)
            total_reward+=reward.item()
            step_count+=1

        episode_reward.append(total_reward)
        episode_steps.append(step_count)
        
    
    average_reward=sum(episode_reward)/len(episode_reward)
    print(f"average reward:{average_reward}")
    env.close()

if __name__=="__main__":
    main()