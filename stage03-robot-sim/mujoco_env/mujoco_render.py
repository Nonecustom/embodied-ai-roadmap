# 任务要求：

# 1. 创建带 render_mode="human" 的环境
# 2. reset
# 3. 随机运行 200 步
# 4. 每一步 env.step(action)
# 5. 如果 terminated 或 truncated，重新 reset 或 break
# 6. 能看到窗口动画
# 7. 记录是否能正常显示
#？遇到问题200太小并且很快就结束了
import gymnasium as gym 
import mujoco
import time

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
    env=gym.make(ENV_NAME,render_mode="human")
    observation,info=reset_env(env)
    for index in range(1000):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=step_env(env,action)
        time.sleep(0.03)

        if terminated or truncated:
            observation,info=reset_env(env)
    
    env.close()

if __name__=="__main__":
    main()