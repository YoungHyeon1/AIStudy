import numpy as np
import nltk
from tqdm import tqdm
#=====================================================
# Input Target batch 데이터 생성 함수
# #===================================================
def make_batch(sentences, word_dict):

  input_batch = []
  target_batch = []

  for sen in sentences:
    word = sen.split()
    input = [word_dict[n] for n in word[:-1]]
    target = word_dict[word[-1]]
    input_batch.append(np.eye(len(word_dict))[input])  # One-Hot Encoding
    target_batch.append(target)
    print(word)
    print(target_batch)
    raise

  return input_batch, target_batch


#=======================================================
# windows data creation
#=======================================================
def make_windows(cleaned_sentences, windowSize=2) :
  MASK_TOKEN = "<MASK>"

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
  windows = flatten(nltk.ngrams(sentence[:-windowSize].split(' '),windowSize + 1) \
                                      for sentence in tqdm(cleaned_sentences))
  return windows
