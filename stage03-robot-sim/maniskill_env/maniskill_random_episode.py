# 任务要求：

# 1. 创建 PickCube-v1 环境
# 2. reset
# 3. total_reward = 0
# 4. step_count = 0
# 5. while 循环运行，直到 terminated 或 truncated
# 6. 每一步随机 sample action
# 7. 执行 env.step(action)
# 8. 累加 reward
# 9. 累加 step_count
# 10. 结束后打印 total_reward、step_count、terminated、truncated
# 11. env.close()
import torch
import gymnasium as gym
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
    total_reward=0.0
    step_count=0
    observation,info=reset_env(env)
    terminated=torch.tensor(False)          #k terminated，truncated是tensor所以初始时也必须是tensor
    truncated=torch.tensor(False)
    while not (terminated.item() or truncated.item()):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=step_env(env,action)
        total_reward+=reward.item()
        step_count+=1
    
    print(f"total_reward:{total_reward}")
    print(f"steps:{step_count}")
    print(f"terminated:{terminated}")
    print(f"truncated:{truncated}")
    env.close()

if __name__=="__main__":
    main()