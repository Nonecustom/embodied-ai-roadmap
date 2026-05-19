# 让 CartPole 随机运行 10 步
# 每一步打印：
# step编号
# action
# reward
# terminated
# truncated

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
    observation,info=reset_env(env)
    for step in range(10):
        action=sample_action(env)
        next_observation,reward,terminated,truncated,info=get_step(env,action)

        print(f"step编号{step+1}")
        if action==0:
            print(f"action:左移")
        else:
            print(f"action:右移")
        print(f"reward:{reward}")
        print(f"netx_observation{next_observation}")
        print(f"terminated:{terminated}")
        print(f"truncated:{truncated}")
        if terminated or truncated:
            break
    env.close()

if __name__=="__main__":
    main()