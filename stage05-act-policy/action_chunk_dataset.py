from pathlib import Path
import numpy as np


CHUNK_SIZE = 4

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR.parent
DATASET_PATH = (
    REPO_DIR
    / "stage04-behavior-cloning"
    / "cartpole_bc"
    / "data"
    / "cartpole_demo.npz"
)


def load_demonstration_dataset(dataset_path):
    data=np.load(dataset_path,allow_pickle=True)
    episodes=data["episodes"]
    return episodes


def build_chunks_from_episode(episode, chunk_size):
    observations=episode['observations']
    actions=episode['actions']
    chunk_observations=[]
    action_chunks=[]
    max_start=len(actions)-chunk_size+1
    for i in range(max_start):
        now_obs=observations[i]
        now_act=actions[i:i+chunk_size]
        chunk_observations.append(now_obs)
        action_chunks.append(now_act)

    return np.array(chunk_observations),np.array(action_chunks)



def build_chunk_dataset(episodes, chunk_size):
    all_chunk_observations=[]
    all_action_chunks=[]
    for i in range(len(episodes)):
        episode=episodes[i]
        chunk_observations,action_chunks=build_chunks_from_episode(episode,chunk_size)
        all_chunk_observations.append(chunk_observations)
        all_action_chunks.append(action_chunks)
    
    all_chunk_observations = np.concatenate(all_chunk_observations, axis=0)
    all_action_chunks = np.concatenate(all_action_chunks, axis=0)

    return all_chunk_observations,all_action_chunks



def inspect_chunk_dataset(chunk_observations, action_chunks):
    print(f"chunk observations shape:{chunk_observations.shape}")
    print(f"action chunks shape:{action_chunks.shape}")
    print(f"first observation:{chunk_observations[0]}")
    print(f"first action chunk:{action_chunks[0]}")


def main():
    episodes = load_demonstration_dataset(DATASET_PATH)
    chunk_observations, action_chunks = build_chunk_dataset(episodes, CHUNK_SIZE)
    inspect_chunk_dataset(chunk_observations, action_chunks)


if __name__ == "__main__":
    main()
