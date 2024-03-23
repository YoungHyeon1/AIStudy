import json
import numpy as np
with open('result_data_embedding_window.json', 'r') as f:
    loaded_data = json.load(f)

# 리스트를 다시 NumPy 배열로 변환
loaded_data_numpy_ready = [{k: np.array(v) for k, v in item.items()} for item in loaded_data]
import torch
# 데이터 확인
search_item = ['happy', 'pleasant', 'furious', 'sad', 'glad', 'angry']
temp = {}
for item in loaded_data_numpy_ready[:50]:
    for key, value in item.items():
        # if key in search_item:
        temp[key]=value

# # Euclidean Similarity 계산
# print("Eucliean Similarity")
# for i in search_item:
#     for j in search_item:
#             if i == j:continue
#             tensor_a = torch.tensor(temp[i])
#             tensor_b = torch.tensor(temp[j])
#             print(f"{i}, {j} Distance:  {torch.norm(tensor_a - tensor_b)}")


# # 유클리디언 거리 계산
# max_value = 0
# min_value = 10e10
# print("코사인 계산")
# for i in search_item:
#     for j in search_item:
#         if i == j:continue
#         tensor_a = torch.tensor(temp[i])
#         tensor_b = torch.tensor(temp[j])
#         euclidean_distance = torch.norm(tensor_a - tensor_b)
#         print(f"{i}, {j} Distance:  {euclidean_distance.item()}")
#         max_value = max(max_value,euclidean_distance.item())
#         min_value = min(min_value,euclidean_distance.item())

# print(max_value)
# print(min_value)
# print(temp)
        # if key in search_item:
        #     for i in search_item:
        #         tensor_a = torch.tensor(value)
        #         tensor_b = torch.tensor(value)
        #         print(f"{key}, {i} Distance:  {torch.norm(tensor_a - tensor_b)}")

        # print(f"{key}: {value}")
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 예시 데이터: 단어 임베딩 벡터를 나타냅니다.
# 실제로는 학습된 모델에서 임베딩을 추출하여 사용합니다.

labels = list(temp.keys())
embeddings = np.array([temp[word] for word in labels])

# t-SNE 모델 생성 및 고차원 데이터를 2D로 매핑
tsne_model = TSNE(n_components=2, perplexity=5, learning_rate=100, n_iter=1000, random_state=42)
reduced_embeddings = tsne_model.fit_transform(embeddings)

# 결과 시각화
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    x, y = reduced_embeddings[i, :]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.title('Embedding & window')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()


def solution(n):
    num=set(range(2,n+1))

    for i in range(2,n+1):
        if i in num:
            num-=set(range(2*i,n+1,i))
    return len(num)
from itertools import permutations