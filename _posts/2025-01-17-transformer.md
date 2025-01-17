---
layout: single
title:  "[paper] Attention is all you need (Transformer)"
permalink: paper/transformer
toc: true
toc_sticky: true
categories: 
  - paper
---



## Attention is all you need (Transformer)



2017년 Transformer가 등장한 이후 많은 모델들이 Transformer를 사용하고 내가 사용한 대부분의 모델이 Transformer 기반으로 이루어져있다는 사실을 잘 알면서도 대충만 알고 있었다. 기초의 중요성을 느끼고 차근차근 뜯어본다. <br>  

***

### Transformer

Attention is all you need 논문과 Transformer 관련 자료를 공부하고나니 Transformer를 한 문장으로

 > **"각 예측 시점에서 어떤 단어에 집중할지 찾는다"**

라고 요약할 수 있었다.

<br>  

기존(RNN)에는 어느 시점이던 마지막 단어 임베딩만 집중했으나, 뒤쪽 단어 정보에 비해 앞쪽 단어 정보는 적게 담겨져 있는 상태였고 이는 성능의 하락을 불러왔다.

다음 단어를 예측하는데 있어서 마지막 단어 임베딩만 중요하지는 않다는 사실을 우리는 이미 알고 있다.

예를 들어

나는 + 학생 + 입니다  -> I + am + a + student

예제에서

student 라는 값을 가져올 때 "학생" 이라는 단어 임베딩이 가장 큰 의미를 부여할 것이라는 것을 알 수 있다.

<br>  

Decoder의 단어 임베딩이 입력된 Encoder의 단어 임베딩을 모두 살피고 중요도에 따라 가중치를 가해 Context vector를 찾는다. 이것이 기존 방식에 **Attention**이라는 개념을 더한 것이다.

이처럼 마지막 단어 임베딩에만 집중하지 않게끔 Decoder 에서 입력된 단어 임베딩을 모두 살피는 것을 **Attention** 이라고 한다.

<br>   



**Self attention**

그럼 Self attention은 또 무엇인가?

단어 임베딩이 단어 간 거리에 영향을 받지 않는 방법, 즉 attention을 Encoder, Decoder 안에서도 self로 사용하자! 라는 개념이다.

한 단어 임베딩의 context vector를 찾을 때 입력된 모든 단어 임베딩을 살펴 각각에 맞는 가중치(해당 단어에서 중요한 정도)를 찾고 곱해서 찾는 방법이다.

그렇게 되면 한 단어 임베딩이 문장 전체의 의미를 다 가질 수 있다. 물론 뒤에 등장하는 단어 임베딩의 의미도 가질 수 있게 된다!



이렇게 Encoder, Decoder, Encoder-Decoder 간의 attention의 구조를 가진 모델이 바로 **Transformer**인 것이다.



<img src="https://blog.kakaocdn.net/dn/dV8WTb/btrtNSGjFLN/RR3zQ4bUFdIivy5py7ypdk/img.png" alt="img" style="zoom:50%;" />

<br>  

***

지금까지 Transformer의 대략적인 개념을 살펴보았다.

이제 논문을 통해 자세한 구조를 알아보겠다.<br/>  



### **1 Introduction**

RNN, LSTM, GRU 와 같은 Sequential한 model이 대세를 이루어 왔다. recurrent 하고 Encoder, Decoder 형태의 모델의 한계를 극복하기 위한 노력이 이어져 왔다. 순차적으로 계산이 이루어지는 특성은 parallelization(병렬화)를 불가능하게 하고 이는 긴 시퀀스에서 더 문제가 될 수 있다. factorization, conditional computation 등의 방법으로 극복하고 있으나 근본적인 문제는 해결되지 않은 상태이다.

Attention은 입력, 출력 시퀀스 간의 거리에 의존하지 않게 한다. 하지만 이런 attenstion machanism은 아직 recurrent한 신경망과의 결합으로 사용된다.

이 논문에서는 recurrent한 구조를 배제하고 온전히 attention mechanism을 사용하여 Transformer를 제안한다. 

<br/>  


### 2 Background

Sequential한 계산을 줄이려는 CNN 기반의 Extended Neural GPU, ByteNet, ConvS2S 모델들은 병렬로 계산하지만 임의의 입출력 간의 연결을 위해 계산량이 늘어난다. Transformer는 이러한 연산량을 줄이지만 해상도를 떨어뜨린다. 하지만 이는 Multi-Head Attentio으로 극복한다.

self-attention은 단일 시퀀스 내의 다른 position들을 연결하여 해당 시퀀스를 표현하는 것이다.

Transfomer는 RNN과 convolution 없이 self-attention만으로 이루어진 첫 모델이다.

<br/>  


### 3.1 Encoder and Decoder Stacks

 <img src="https://blog.kakaocdn.net/dn/dV8WTb/btrtNSGjFLN/RR3zQ4bUFdIivy5py7ypdk/img.png" alt="img" style="zoom: 33%;" />

**Encoder** 

Encoder는 6개의 동일한 layer로 이루어져 있고 각각의 layer는 두 개의 하위 layer로 구성되어있다.(그림에는 6개 중 하나의 layer만 표현되어있음) multi-head self-attention과  positionwise fully connected feed-forward network 이다. 각 하위 layer는 residual connection이 적용되고 이후 layer normalization가 이루어진다. 
$$
LayerNorm(x+Sublayer(x))
$$
그림 속 [Add & Norm] 에 해당하는 부분이다.

residual connection을 위해 모델의 모든 하위 layer와 embedding의 차원을 512로 통일한다.

**Decoder**

Decoder도 6개의 동일한 layer로 이루어져 있지만 Encoder에서의 두 개의 하위 layer에 하나의 하위 layer가 추가된다. 세번째 하위 layer는 Encoder의 출력값으로 multi-head attention를 수행한다(그림 속 Decoder layer의 중간 하위 layer). Encoder에서와 같이 각 하위 layer는 residual connection이 적용되고 이후에 layer normalization 가 이루어진다. 

Decoder에서는 특정 position 이후의 position에서는 attention 하지 못하게 하기 위하여 masking을 설정한다. 이는 예측하려는 값을 미리 attention하여 제대로된 학습을 못하지 않도록 방지하는 방법이다. 그림 속 Decoder layer의 맨 아래 하위 layer의 이름은 Masked Multi-head Attention임을 볼 수 있다.

<br/>  


### 3.2 Attention

<img src="https://aiml.com/wp-content/uploads/2023/09/Self-Attention-vs-Multi-headed-Attention.png" alt="What is Multi-head Attention and how does it improve model performance over  single Attention head? - AIML.com" style="zoom: 25%;" />

#### 3.2.1 Scaled Dot-Product Attention

입력은 차원이 dk인 query, key 그리고 차원이 dv인 value로 구성된다. query와 모든 key 간의 dot product을 계산한 후 root(dk)로 나누고 softmax 함수를 적용하여 가중치를 얻는다. 
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$
이 방법은 dot-product attention에서 scale을 적용한 것이다.

attention은 주로 additive attention, dot-product attention을 사용하는데 dk값의 크기가 크면 additive attention이 더 좋은 성능을 보인다. dk값이 클수록 dot-product값이 커져 softmax 함수가 매우 작은 gradient를 가지는 지역으로 밀려나기 때문(?)이라고 추측한다고 한다. 이를 해결하기 위해 위에서와 같이 scale 작업을 더해주었다. 

<br/>  


#### 3.2.2 Multi-Head Attention

query, key, value 를 이용해 single attention을 진행하는 것 보다 query, key, value를 각각 dk, dk, dv 차원으로 linear project 하는 것이 유리하다. 이때 h개의 서로 다른 학습된 linear projection을 적용한다. projection 된 query, key, value를 병렬적으로 attention을 적용하고 dv 차원의 출력값을 생성한다. 이 생성값들을 concatenate 한 후 한번 더 linear project하여 최종 출력값을 생성한다.

single attention 보다 multi-head attention을 사용할 때 다양한 위치의 정보를 동시에 attention 할 수 있다.


$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(Q W^Q_i, K W^K_i, V W^V_i)  
$$

$$
W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad 
   W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad 
   W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}, \quad 
   W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}
$$

이 논문에서는 h=8 parallel attention layer로 설정하고 
$$
d 
k
​
 =d 
v
​
 =d 
model
​
 /h=64
$$
로 설정하였다. 각 헤드의 차원이 줄었기 때문에 full(512) 차원을 가진 self attention과 계산비용은 유사하다.

<img src="https://blog.kakaocdn.net/dn/dV8WTb/btrtNSGjFLN/RR3zQ4bUFdIivy5py7ypdk/img.png" alt="img" style="zoom: 33%;" />

<br>  


#### 3.2.3 Applications of Attention in our Model

해당 모델에서 Attention은 총 3부분에서 사용된다.

- "encoder-decoder attention

  query는 이전 단계의 출력값에서 가져오고 key와 vlaue는 encoder의 결과값에서 가져온다. 이 과정으로 Decoder의 모든 position에서 encoder의 모든 position의 정보를 attention 할 수 있다. 이 논문 이전에도 사용되던 기본적인 attention 방법과 유사하다.

-  self-attention layers in encoder

  encoder 내부에서 일어나는 attention으로 query, key, value 모든 값을 encoder의 이전 layer에서 가져온다. encoder 내부 모든 position에서 그 자체 모든 position에 attention할 수 있다.

-  self-attention layers  in decoder

  decoder 내부에서 일어나는 attention으로 위와 동일하지만 마스킹이라는 개념이 추가된다. decoder는 출력 단어 시퀀스를 생성하는 역할이기 때문에 아직 출력되지 않은 미래의 값(정답)을 미리 attention 하는 것은 학습을 방해하게 된다. 따라서 미래의 단어에 해당하는 부분을 -inf로 마스킹하여 계산하게 된다. -inf로 마스킹하는 이유는 여러 연산이 끝난 후 마지막에 softmax 함수를 적용하게 되는데 이때 0값을 가지게 하기 위함이다.
  
<br/>  

  

### 3.3 Position-wise Feed-Forward Networks

ecoder와 decoder의 하위 layer 중 Position-wise Feed-Forward Networks가 있다. 그림 속 Feed Forward 부분이다. 두 번의 linear transformation과 그 사이 ReLU 활성화 함수로 이루어져 있다.
$$
FFN(x)=max(0,xW_{1}+b_{1})W_{2}+b_{2}
$$

<br/>  


### 3.4 Embeddings and Softmax

다른 시퀀스 변환 모델들과 유사하게, 입력, 출력 토큰을 d(model) 차원의 벡터로 임베딩화 한다. 

또한 linear transformation과 softmax 함수를 통해 decoder의 출력값을 다음 토큰 예측 확률로 변환한다.

<br/>  


### 3.5 Positional Encoding

transformer model은 위에서 봤듯이 attention으로 위치에 영향받지 않고 관계를 계산할 수 있었다. 따라서 단어 임베딩에는 어떠한 절대적인 위치정보가 포함되어있지 않아 시퀀스의 순서를 사용하기 위해 위치정보를 주입해야한다. encoder와 decoder의 하단 부분에 positional encoding을 embedding에 추가한다. positional encoding을 embedding에 단순히 더하는 방식이다. 두 벡터가 동일한 차원을 가지기 때문에 가능하다.

positional encoding 값은 다양한 방법으로 구현이 가능하나 해당 모델에서는 다음과 같이 구한다. 
$$
P E(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

$$
P E(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$
저자는 이 방법을 선택한 이유를 "모델이 상대적인 위치에 따라 attention 하는 방법을 쉽게 학습할 수 있도록 하기위함"이라고 표현한다. 고정된 오프셋 k에 대해 PE(pos+k)가 PE(pos)의 선형 함수로 표현될 수 있기 때문이라고 한다.

 이 뜻에 대해서는 [https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/) 참고

<br/>  



**Reference**

[VASWANI, A. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017](https://doi.org/10.48550/arXiv.1706.03762)

[youtube 혁펜하임 | AI & 딥러닝 강의](https://www.youtube.com/watch?v=8E6-emm_QVg)

