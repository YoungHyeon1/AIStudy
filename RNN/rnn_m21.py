# Many-to-one (2-to-1) RNN 예

#### conda create -n word2vec nltk pandas
# conda activate word2vec
#### conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#       또는 conda install pytorch torchvision torchaudio cpuonly -c pytorch
# (word2vec) python rnn_m21.py train 500    # 학습시
# (word2vec) python rnn_m21.py i love    # Prediction 시

import numpy as np
import torch.nn as nn
import torch, sys
import torch.optim as optim
from tqdm import tqdm
from TextRNN import TextRNN
from utils import make_batch
#=====================================================
# Rnn Training 함수
#=====================================================
def train_rnn(totalEpoch) :  
  criterion = nn.CrossEntropyLoss() # 손실함수를 CrossEntropy로 선택
  optimizer = optim.Adam(model.parameters(), lr=0.01) # Optimizer를 Adam으로 선택

  #---------------학습 500회 
  for epoch in tqdm(range(totalEpoch)):
            # requires_grad=True: backpropagation시에 gradients 누적하라
            # back propagation 시에 batch_size * sequence_length 만큼 gradient를 계산 및 반영하도록 
            #   hidden vector들의 변수가 필요 (0으로 초가화)
    hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)

    for i in range(len(input_batch)//batch_size):
      inputs = input_batch[i*batch_size:(i+1)*batch_size]
      targets = target_batch[i*batch_size:(i+1)*batch_size]
      # inputs_padded = rnn_utils.pad_sequence(inputs, batch_first=True)
      # targets_padded = rnn_utils.pad_sequence(targets, batch_first=True)
      output = model(hidden, inputs)
    #if epoch == totalEpoch-1 :
      print(f'\n*** Epoch # {epoch}일 때 학습 직후 배치데이터 전체에 대한 output layer 출력값 output:\n{output}')
      print('-'*80)

      loss = criterion(output, targets)

      if totalEpoch <= 3 or (epoch % 100) == 0:
            print('Epoch:', '%04d' % (epoch), 'cost =', '{:.6f}'.format(loss))
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

#=====================================================
# [1] Word Processing
#=====================================================
sentences = [
  "i like dogs", "i love coffee", "i hate milk", "i enjoy tea",
  "you like cats", "you love milk", "you hate coffee", "you want snowing",
  "he expects promotion", "he wants money", "he waits snowing",
  'I enjoy cats', 'I like promotion', 'you like promotion',
 'you like coffee', 'I expect tea', 'I enjoy milk', 'he enjoy tea', 'I hate promotion',
 'he expect money', 'I love dogs', 'I like money', 'I hate money',
 'you like tea', 'he hate coffee', 'he enjoy money',
 'he want money', 'you like dogs', 'he like milk', 'you love money', 'I like coffee',
  'I want dogs', 'he want cats', 'you enjoy money'
]

windowSize = 2
dtype = torch.float
word_list = list(set(" ".join(sentences).split()))
word_list.sort()    # 매 실행시마다 동일한 word index 번호를 갖게
word_dict = {w: i for i, w in enumerate(word_list)}
print('\n','='*80)
print("word dictionary:",word_dict,'\n')
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)
#=====================================================
# [2] Input, Target batch data 생성
#=====================================================
      # sentence 문장들로 부터 입력tensor, 출력tensor 구함
input_batch, target_batch = make_batch(sentences, word_dict)
input_batch = torch.tensor(np.array(input_batch), dtype=torch.float32, requires_grad=True)
print('\n','='*80)
print("input batch:",input_batch)

target_batch = torch.tensor(target_batch, dtype=torch.int64)
print('\n','='*80)
print("target batch:",target_batch)

#=====================================================
# main 
#=====================================================
if __name__ == "__main__" :
  batch_size = min(len(sentences), 32)
  n_hidden = 10  # 은닉층의 길이
  model = TextRNN(n_class, n_hidden)                 # rnn 모델 객체 생성
  print('\n','='*80)
  print('\nUsage [학습]: python rnn_m21.py train 100')
  print('Usage [Prediction]: python rnn_m21.py word1 word2\n')
        # [4] textRNN 모델 학습
  if len(sys.argv) >= 3 :
      #---------------------------------- 학습
    if 'train' in sys.argv[1] and int(sys.argv[2])> 0:
      train_rnn(int(sys.argv[2]))
      
      # ------------------------------
      # 전체 학습 문장에 대해 prediction 해보기
            # 전체 문장에 대해 입력 벡터 생성
      # ------------------------------
      print('\n','='*80)
      print('학습 완료 했으니, 전체 입력 데이터에 대해, prediction으로 학습 결과 확인')
      model.reset_count()    # hidden_vectors 의 출력 index를 0으로 reset
      input = [sen.split()[:2] for sen in sentences]

            # hidden layer 값 초기화
            # 1 : hidden layer 수
            # batch_size : 저장할 hidden matrix의 행, 즉, backward propagation 대상이 되는 time 수
            # n_hidden : hidden matrix의 열, 즉 hidden vector 한개의 노드 수
      hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
            # input 값에 대한 prediction 실행
      for i in range(len(input_batch)//32):
            predict = model(hidden, input_batch[i*batch_size:(i+1)*batch_size]).data.max(1, keepdim=True)[1]
            # 결과 출력
            print('\n','.'*80, '\n[학습후 테스트 결과]\n')
            print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
            # 학습 결과 파일로 저장
      torch.save(model.state_dict(), 'text_rnn.pth')
      
    else :
      #---------------------------------- 매개변수 문장에 대해 Prediction
      queryStr = sys.argv[1] +' '+ sys.argv[2] + ' ' + 'i'
      sentence = []
      sentence.append(queryStr)
      
      query_input_batch, query_target_batch = make_batch(sentence, word_dict)
      input_batch = torch.tensor(np.array(query_input_batch), dtype=torch.float32, requires_grad=False)
      
      print('\n','-'*80)
      print("query input batch:",input_batch)
      
      # model = TextRNN()  # Initialize the model the same way you did before
      model.load_state_dict(torch.load('text_rnn.pth')) # 저장된 파일로 부터 모델 및 파라미터  불러오기
      model.eval()

      hidden = torch.zeros(1, 1, n_hidden, requires_grad=False)
            # input 값에 대한 prediction 후 output layer 출력값(softmax값) 출력해보기 
      output = model(hidden, input_batch)
      model.print_output_softmax(output)
            # 답만 가져오기
      predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
            # 결과 출력
      print('\n','='*80)
      print(f'학습용 sentences : {sentences}\n')
      print(f'predict 결과:{predict}')
      print(f'*** {sys.argv[1]}  {sys.argv[2]} ==> {number_dict[predict.item()]}')

      
    
  
