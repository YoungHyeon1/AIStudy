# CBOW(Continumous Bag of Words) WOrd to Vector Embedding

### conda update -n base -c defaults conda
# conda create -n word2vec nltk pandas
# conda activate word2vec
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#       또는 conda install pytorch torchvision torchaudio cpuonly -c pytorch
# (word2vec) python cbow.py

# https://didu-story.tistory.com/101

import nltk
from os import path
from tqdm import tqdm
import re, sys
import pandas as pd

# Clean sentences
def clean_text(text) :
            #change capital letters to lower
    text = ' '.join(word.lower() for word in text.split(" "))
            # re.sub('패턴', '바꿀문자열', '문자열', 바꿀횟수)
    text = re.sub(r"([.,!?])", r" \1 ", text)   # 모든 마침표, 쉼표, 느낌표, 물음표 다음에 공백을 추가
    
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)# 알파벳(a-z, A-Z), 마침표, 쉼표, 느낌표, 물음표를
                                                #     제외한 모든 문자를 제거
    return text

#============================================================================    
# Tokenizer downloading and setting
#===========================================================================
            # install nltk's data (영어 텍스트를 문장단위로 분할 하는 사전 훈련된 모델)
if not path.exists('./tokenizers/punkt/english.pickle'):
            # nltk.download('popular')
    nltk.download('popular')

            # set tokenizer
tokenizer = nltk.data.load('./tokenizers/punkt/english.pickle')

#===========================================================================
# data sentenses reading and cleaning
#===========================================================================
            # read data text for training
            # https://www.gutenberg.org/files/84/84-h/84-h.htm (frankenstein book)
with open("./book.txt",encoding='cp949') as fp:
    book = fp.read()
    
            # Split the raw text book into sentences
            # tokenize the data    
sentences = tokenizer.tokenize(book)
            
            # check the read sentences, and print some sentences
print (f'totally {len(sentences)} sentences')
for i in range(5):
    sentence = sentences[i]
    print (f"*** sentence[{i}]: {sentence}\n")

        # check the cleaned sentences
cleaned_sentences = [clean_text(sentence) for sentence in sentences]
print('='*80)
for i in range(5):
    sentence = cleaned_sentences[i]
    print (f"*** cleaned sentence[{i}]: {sentence}\n")
    
#=======================================================================
# CBOW data creation 
#=======================================================================
MASK_TOKEN = "<MASK>"
windowSize = 2          # target token 앞뒤로 각각 2개씩, 총 앞뒤로 주변에 4개의 token을 사용한다

        # 각 문장의 토큰 리스트를 순회하면서 지정된 크기(길이 5개)의 window로 묶어주기
        # lambda 매개변수 : 표현식의 예:
        # matrix = [[1, 2], [3, 4], [5, 6]]
        # squared = [num ** 2 for row in matrix for num in row]
        # print(squared)  # Output: [1, 4, 9, 16, 25, 36]
        # 2차원-->1차원으로 펴기
flatten = lambda outer_list: [item for inner_list in outer_list for item in inner_list]
        # nltk.ngrams([1,2,3,4,5], 3)) ===> [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
        # 전체 cleaned_sentenses로부터, 
        
        # cleaned_sentences로부터 가져온 각 문장에서 5개의 단어(토큰)로 이루어진 windowx 리스트를 가져와라 
        
        # cleaned_sentences에서 sentence를 하나씩 가져와서 nltk.ngram을 통해 
        # sentence 맨 앞과 맨 뒤에 각각 2개씩의 <MASK> 토큰을 추가한 후, 5개 단어 길이의 wind리스트로
        # 바꾼 후, flatten 하라
        # I iove you --> [<MASK>, <MASK>, 'I', 'love', 'you',<MASK>, <MASK>]
windows = flatten([list(nltk.ngrams([MASK_TOKEN] * windowSize + sentence.split(' ') + [MASK_TOKEN] * windowSize, 
                                    windowSize * 2 + 1)) \
                                    for sentence in tqdm(cleaned_sentences)])

print('='*80)
print(' First 10 windows data for trainig')
print('='*80)
for i in range(10):
    print(windows[i])

# CBOW 데이터로 만들어주기
data = []
for window in tqdm(windows):
    target_token = window[windowSize]   # windowSize번째, 즉 3번째 토큰이 target token이 됨
    context = []
                                        # 가운데 위치한 target token을 제외한 주변 10갸ㅏ 토큰을 구함
    for i, token in enumerate(window):
        if token == MASK_TOKEN or i == windowSize:
            continue
        else:
            context.append(token)
    # '구분자'.join(리스트)
    data.append([' '.join(token for token in context), target_token])
    
# data를 column 제목이 있는 DataFrame의 형태로 변환
cbow_data = pd.DataFrame(data, columns=["context", "target"])

print('='*80)
print(' First 20 CBOW data for trainig')
print('='*80)
print(cbow_data[:20])

#=======================================================================
# CBOW Training
#=======================================================================

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n***** Device : {device}")

if device != 'cpu' :    # 'cuda'
    GPU = True
else :
    GPU = False
    
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        if GPU:
            # embedding layer 생성 함수 정의 (one-hot encoded vector->dense embedding layer)
            # vocab_size * embedding_dim (4543*100) 크기의 lookup _table 내부적으로 생성
            self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)  # GPU로 이동
            # Linear transformation layer 생성 함수 정의 (임베딩 디멘젼->vocaburary size layer)
            # linear layer는 tensorflow의 dense layer와 유사
            self.linear = nn.Linear(embedding_dim, vocab_size).to(device)  # GPU로 이동            
        else :
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embedded = self.embeddings(inputs).mean(dim=0)  # 각 동일 단어 임베딩의 평균치 구함
        out = self.linear(embedded)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs

# word_ti_ix 집합(사전)을 참고로 하여 index 값들로 이루어진 context vector들을 리턴
# 예: [32, 2324, 156, 33] 로 변환
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    if GPU:
        return torch.tensor(idxs, dtype=torch.long).to(device)
    else :
        return torch.tensor(idxs, dtype=torch.long)


# 가정: cbow_data DataFrame과 word_to_ix 사전이 이미 준비되어 있음
# unique한 단어 집합 생성 : set() 사용
vocab = set (word for _, row in cbow_data.iterrows() for word in row['context'].split(' ') + list(row['target']))
print('='*80)
print(f"vocaburary set : {vocab}")
print('='*80)
vocab_size = len(vocab)
print(f"vocaburary size: {vocab_size}")

embedding_dim = 100  # 임베딩 차원 설정

# word 기준으로 index 알수 있게 set로 표현
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(f"word_to_ix: {word_to_ix}")

model = CBOW(vocab_size, embedding_dim)
if GPU : 
    model = model.to(device)

# Negative Log Likelihood Loss 손실함수 설정, 아래 코드에서 손실 계산 시 사용
loss_function = nn.NLLLoss()    
# 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 옵티마이저 생성, 아래 코드에서 사용
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 학습 시작
SHOW_FLAG = True
WRITE_FLAG = True
EPOCH_NO = 300       # 300

print("*"*80)
print(f'\n총 windows(학습용 데이터) 갯수 : {len(windows)} \n')
print('='*80)
for epoch in tqdm(range(EPOCH_NO),desc="Epoch 진행률", unit="iter"):  # 에포크 설정
    total_loss = 0
    windowNo = 0
    for _, row in tqdm(cbow_data.iterrows(),desc="Sentence 진행률", unit="iter"):     # 전체 데이터에 대하여
        
        # 학습용 window 데이터에 대한 context vector 생성
        context_vec = make_context_vector(row['context'].split(' '), word_to_ix)
        
        # 타겟 벡터 생성
        target_vector = [0 for i in range(vocab_size)]
        target_vector[word_to_ix[row['target']]] = 1.

        if GPU :
            target_value = torch.tensor(target_vector, dtype=torch.long).to(device)
        else :
            target_value = torch.tensor(target_vector, dtype=torch.long)
            
        model.zero_grad()                   # 그래디언트 초기화 (gradient 누적 방지)

        # 입력 벡터 생성: 주변 단어들의 임베딩 벡터를 평균내어 하나의 벡터로 합칩니다. 
        # 이 평균 벡터가 CBOW 모델의 입력으로 사용됩니다.
        log_probs = model(context_vec)      # 실제 출력값 (log probability 계산)

        if SHOW_FLAG and windowNo % 10000 == 0:
            print('\n','='*80)
            print(f'epoch #: {epoch}, window # : {windowNo}')
            print(f'context word, context vector 값 [{len(context_vec)}] 개: {row["context"]} :: {context_vec}')
            print(f'targe_value tensor 값 [{len(target_value)}] 개: {target_value}')
            print(f'학습 중 log_probabilities tensor 값 [{len(log_probs)}] 개: {log_probs}')
            print('='*80)
                
        #loss = loss_function(log_probs, torch.tensor([word_to_ix[row['target']]], dtype=torch.long).to(device))
        loss = loss_function(log_probs, target_value)   # loss 계산
        loss.backward()                                 # 손실 및 gradient 계산
        optimizer.step()                                # weights update
        
        total_loss += loss.item()
        #print(f"Sentence [{windowNo}]'s total_loss : {total_loss}")
        windowNo += 1
    print(f'\n\n*** Epoch {epoch}: Total Loss: {total_loss}\n')


# 임베딩 값 추출
# 임베딩 행렬의 차원 크기를 확인
print("*"*80)
print('\n*** 임베딩 Weights의 크기: ',model.embeddings.weight.size(),'\n')

# 임베딩 행렬의 실제 값(데이터) 출력
embeddings_data = model.embeddings.weight.data  # .data는 텐서의 데이터를 가져옵니다.
print('=== 최종 임베딩 Weights === \n')
print(embeddings_data) 

print('\n=== 첫번째 단어 임베딩 Weight === \n')
print(embeddings_data[0],'\n') 

# 모델의 임베딩 레이어 가중치를 복사하여 계산 그래프로부터 분리된 상태로 embeddings 변수에 저장
# detach() 실행 이후에는 gradient 계산에 참여 안함
embeddings = model.embeddings.weight.detach()

# 관심 단어만 출력해보기 위해, 일단 embedding 단어를 리스트 type의 word에 저장
i=0
word = [];
for key, value in word_to_ix.items():
    word.append(key)
    i += 1

result_data = []
print("*"*80, "\n")
print('\n=== 샘플 몇 단어에 대한 Word Embedding === \n')
for ndx, embed in tqdm(enumerate(embeddings)):
    embedded_word = word[ndx]
    if embedded_word in ['happy', 'pleasant', 'furious', 'sad', 'glad', 'angry']:
        print(f"\nembeddings[{ndx}], {embedded_word} : {embed}")
    # embedding 결과를 파일로 저장
    if GPU :
        result_data.append({embedded_word: embed.cpu().numpy()})
    else :
        result_data.append({embedded_word: embed.numpy()})
import json
data_json_ready = [{k: v.tolist() for k, v in item.items()} for item in result_data]
with open('result_data_embedding.json', 'w') as f:
    json.dump(data_json_ready, f, indent=4)

print("*"*80, "done.\n")



# with open("result_data.txt", 'r') as f:


'''
[참고자료]
Euclidean Similarity 계산의 예
import torch

# 두 텐서 정의
tensor_a = torch.tensor([1.0, 2.0, 3.0])
tensor_b = torch.tensor([4.0, 5.0, 6.0])

# 유클리디언 거리 계산
euclidean_distance = torch.norm(tensor_a - tensor_b)

print(f'Euclidean Distance: {euclidean_distance.item()}')



# Cosine Similarity 계산의 예
import torch.nn.functional as F

# 두 텐서 정의
tensor_a = torch.tensor([1.0, 2.0, 3.0])
tensor_b = torch.tensor([4.0, 5.0, 6.0])

# 코사인 유사도 계산
cosine_sim = F.cosine_similarity(tensor_a.unsqueeze(0), tensor_b.unsqueeze(0))

print(f'Cosine Similarity: {cosine_sim.item()}')
'''