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

def run_one_episode(env):
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
    return total_reward

def run_many_episodes(env,num_episodes):
    episode_rewards=[]
    for index in range(num_episodes):
        total_reward=run_one_episode(env)
        episode_rewards.append(total_reward)
        print(f"episode {index+1} reward:{total_reward}")
    
    print(f"average reward:{sum(episode_rewards)/len(episode_rewards)}")
    return episode_rewards


def main():
    env=gym.make(ENV_NAME)
    print(f"--测试跑一轮--")
    run_one_episode(env)
    print(f"--测试跑多轮--")
    run_many_episodes(env,num_episodes=5)
    env.close()

if __name__=="__main__":
    main()
