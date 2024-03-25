#=====================================================
# TextRNN Class
#=====================================================
import torch.nn as nn
import torch
dtype = torch.float

class TextRNN(nn.Module):
  forward_pass = 0
  def __init__(self, n_class, n_hidden, n_output):
    super(TextRNN, self).__init__()
          # dropout 옵션은 hidden layer갯숫가 1 초과일때만 사용
    #self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.3)
          # input_size=n_class : 입력값을 클래스 크기만큼 길이의 one-hot vector로 할거니까
          # hidden_size : hidden layer node 수 = 5 로 미리 정함
    print(n_class)
    self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
          # trainable weight matrix initialized with random values
          # nn.Parameter() : 해당 파라미터가 학습시에 참여하는 weight가 됨
#     self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype)) # n_hiddex * n_class 크기 matrix
          # n_class 크기 bias용 vector  
#     self.b = nn.Parameter(torch.randn([n_class]).type(dtype))
    self.fc = nn.Linear(n_hidden, n_output)  # 출력을 위한 선형 레이어
          # raw 출력값들에 적용할 함수
    self.Softmax = nn.Softmax(dim=1)

  def add_count (self, n=1):
    self.forward_pass += n
    
  def reset_count (self):
    self.forward_pass = 0
    
      # X : 입력 시퀀에서의 변하는 X값
      # hidden : 계속 update 되는 hidden matrix 값
      # 이 forward 함수는 epoch 횟수만큼 호출, 
  def forward(self, hidden, X):
          # input tensor to have a shape of (seq_len, batch, feature), 
          # batch와 seq_length 위치를 교체
          #          이 예제의 경우 (seq_len, batch_size, hidden_size)형식으로 리턴되는데
          #           (2, 11, 5) 크기가 됨
    X = X.transpose(0, 1)
          # hidden_vectors: 각 time step 마다의 hidden vector 노드들의 값(h(t))들 저장
          #           나중에 back-propagation시에 gradient 값 계산을 위하여 저장해 둠
          #           (seq_len, batch, hidden_size) = (2, 11, 5)
          # hidden : 마지막 time step 직후의  hidden vector 값 
          #           hidden layer가 다층일 경우에는 layer 층의 갯수 만큼 값을 가짐
        # 한번의 호출로 한 batch, 모든 시퀀스, 모든 히든벡터에 대한 벡터값들을 리턴
        #   2 시퀀스 * 11 batch * 5 vector nodes 값을

    hidden_vectors, hidden = self.rnn(X, hidden)    # 입력층 batch 값들과 히든 벡터를 이용하여
                                                    # 히든 벡터 값들을 계산
    print(f'\n*** hidden_vectors[{self.forward_pass}]: {hidden_vectors}')                                        
    print(f'\n*** hidden[{self.forward_pass}]: {hidden}')                                        

#     last_hidden_vector = hidden_vectors[-1]   # many-to-one 모델이므로 hidden 노드중 마지막 노드값만 필요

            # torch.mm() :  matrix multiplication, 실제 계산 출력값을 여기서 리턴
#     model = torch.mm(last_hidden_vector, self.W) + self.b  
    model = self.fc(hidden_vectors[:, -1, :])
    self.add_count(1)
    
    return model

  def print_output_softmax(self, output):
        softmax_values = self.Softmax(output)
        print("Softmax Results:", softmax_values)
