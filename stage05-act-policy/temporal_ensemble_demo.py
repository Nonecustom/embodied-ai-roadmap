# 1. 导入库
import torch
import torch.nn as nn


# 2. 构造 toy action chunks
action_chunks = [
    [0, 0, 1, 1],  # t=0 预测 actions[0:4]
    [0, 1, 1, 1],  # t=1 预测 actions[1:5]
    [1, 1, 0, 0],  # t=2 预测 actions[2:6]
]

# 3. 收集每个时间步的候选动作 collect_votes
def collect_votes(action_chunks):
    votes={}                                  ##用字典表示，每个时间会有多个动作预测，因此字典有每个时间的list
    for chunk_index in range(len(action_chunks)):
        action_chunk=action_chunks[chunk_index]         ##读出每个chunk
        for action_index in range(len(action_chunk)):   ##遍历每个chunk的预测的每个action
            global_time=chunk_index+action_index        ##计算出属于预测总时间的哪个时间
            action=action_chunk[action_index]           ##读出当前动作预测

            if global_time not in votes:                ##当votes里面没有这个时间的list创建一个
                votes[global_time]=[]

            votes[global_time].append(action)

    return votes

# 4. 对每个时间步投票 vote_actions
def vote_actions(votes):
    final_actions=[]
    for index in range(len(votes)):
        candidate_actions=votes[index]
        predicted_0=0
        predicted_1=0
        for action_index in range(len(candidate_actions)):
            if candidate_actions[action_index]==0:
                predicted_0+=1
            else:
                predicted_1+=1
        if predicted_0>=predicted_1:
            final_actions.append(0)
        else:
            final_actions.append(1)

    return final_actions

# 5. 打印结果 inspect_results
def inspect_results(action_chunks, votes, final_actions):
    print("action chunks:")
    for chunk_index in range(len(action_chunks)):
        print(f"chunk {chunk_index}: {action_chunks[chunk_index]}")

    print("\nvotes:")
    for time_step in range(len(votes)):
        print(f"time {time_step}: {votes[time_step]}")

    print("\nfinal actions:")
    print(final_actions)

# 6. 主函数 main
def main():
    votes=collect_votes(action_chunks)
    final_actions=vote_actions(votes)
    inspect_results(action_chunks,votes,final_actions)

# 7. 程序入口
if __name__=="__main__":
    main()