---
layout: single
title:  "[paper] Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram"
permalink: paper/LeadAgnostic
toc: true
toc_sticky: true
categories: 
  - paper
---

**Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram (Jungwoo Oh et al.)**

이 논문은 ECG foundation model인 ECG-FM 논문을 리뷰하다 발견한 ECG-FM의 기본이 되는 논문이다.  

## Background
ECG 분야에서는 대부분 contrastive learning 을 통한 학습 방법을 주로 사용해왔다. Contrastive learning은 유사한 데이터 샘플(positive pairs)끼리는 거리를 최소화 하는 방향으로, 유사하지 않은 데이터 샘플(negative pairs)끼리는 거리를 최대화 하는 방향으로 모델을 학습하는 Self-supervised learning의 한 방법이다.  

대부분의 Contrastive learning에서는 유사한 데이터(positive pairs)를 만들기 위해 데이터 증강을 진행한다. 증강된 데이터들과 원본 데이터를 positive pair로 하여 학습하는 방식이다. 하지만 ECG 분야에서는 생리학적으로 중요한 패턴이 손상되거나 semantic meaning이 잘못 정렬될 수 있어 증강이 까다롭다.  

Contrastive learning을 위한 다양한 방법이 존재했다. 먼저 3KG (Gopal et al., 2021) 방식은 ECG 데이터로 3D 차원의 VCG 데이터를 생성하여 이를 positive pair로 사용하는 contrastive learning이다. 그리고 Contrastive Multi-Segment Coding (CMSC) 방식은 한 개의 ECG 데이터에서 시간적으로 인접한 데이터들끼리를 positive pair로 정의하여 contrastive learning을 진행하는 방식이다. 이 방법은 데이터의 증강이 필요 없기 때문에 증강 과정에서의 패턴 손상 문제를 해결한 방법이다.  

본 논문에서 제시한 Self-supervised 모델에는 크게 세가지 포인트가 있다.

<img width="692" height="424" alt="image" src="https://github.com/user-attachments/assets/8d6431f3-bdd7-4861-9ceb-7ffca556a55e" />  


### 1. wav2vec2.0
음성분야 self-supervised learning model 중 하나인 wav2vec2.0의 구조를 그대로 가져와 적용하였다. wav2vec2.0 모델에 대한 리뷰는 이전에 진행하여 간단히 소개한다.  
[참고](https://hyunoim.github.io/paper/wav2vec2)

<img width="420" height="228" alt="image" src="https://github.com/user-attachments/assets/cda9491b-8313-4500-b923-451fb39c18fa" />  
 

raw waveform (X)를 CNN based Encoder를 통해 latent speech representation(Z) 로 만든다. 그리고 이것을 일부(6.5% 확률로 start token을 정한 후 그 뒤에 10개의 토큰을 masking) 마스킹하여 Transformer 아키텍처로 구성된 context network를 통과시켜 Contextualized Representations(C)를 만든다. 그리고 masking되지 않은 이전의 latent speech representation(Z)로 quantization 과정을 거쳐 quantized representation(Q)을 생성하여 Contextualized Representations(C)와의 contrastive loss를 가지고 학습하는 식의 방법이다.

### 2. Contrastive Multi-Segment Coding (CMSC)

<img width="236" height="102" alt="image" src="https://github.com/user-attachments/assets/a885bc7e-5375-4309-9fd6-98e30dab22dd" />  

위에서 언급한 내용과 같이 데이터 증강없이 시간을 기준으로 인접한 데이터들 끼리 postitive pair로 묶어 학습하는 contrastive learning이다. 본 논문에서 사용한 데이터는 대략 10s 길이의 ECG데이터 인데, 이를 5s 간격으로 겹치지 않게 나누어 이 쌍을 positive pair로 사용하는 방식을 취했다.

### 3. Random Lead Masking (RLM)

<img width="447" height="131" alt="image" src="https://github.com/user-attachments/assets/8960693b-8c31-45bc-8324-a34dd45ca378" />  

한 개의 ECG 데이터는 여러 lead로 이루어진 경우가 대부분이다. 이는 심전도 측정 시 동시에 여러 센서를 통해 입력을 받아 생기는 여러 채널들이다. 본 논문에서 이 여러 lead를 가진 ECG 데이터를 input으로 넣을 때 lead별로 50%의 확률로 lead 전체를 masking하는 방식을 취했다.  

왜 그런 방식을 취했나 싶지만, 이는 fine-tunning 시의 모델의 일반화 성능을 위한 방법이다. ECG 데이터들이 12-lead 데이터가 많지만 항상 그러한 것은 아니다. 6-lead, 3-lead, 2-lead, 심지어는 1-lead 데이터도 존재한다. Fine-tunning 시에 각각의 상황에 맞게 학습을 시켜야하는데, pre-training을 할 때 12-lead 데이터로만 학습을 진행할 경우 모델의 일반화 성능이 떨어질 수 있다는 점을 겨냥한 것이다. 12-lead 데이터에서 각각의 lead를 50% 확률로 lead 전체를 masking하는 방법을 진행하여 6-lead, 3-lead, 2-lead 와 같은 여러 조합의 representation을 학습하는 듯한 효과를 가져올 수 있다고 말한다.

---

모델의 loss는 local과 global한 representation을 전부 학습할 수 있도록 설정하였다.  

<img width="273" height="95" alt="image" src="https://github.com/user-attachments/assets/c8cb4d4e-d7db-49e7-a841-5fbd1cd3b0a5" />  

Wav2vec2.0에서 사용한 loss와 동일한 local loss이다. quantized representation(Q)와 Contextualized Representations(C)의 차이를 줄이는 방향으로 Loss를 설정한 것을 볼 수 있다. 여기서 M은 마스킹된 부분을 의미하고, 따라서 L(local)은 마스킹된 부분에서의 loss 평균이라 할 수 있다.  

<img width="292" height="99" alt="image" src="https://github.com/user-attachments/assets/8adf59b8-9851-4547-bacf-09356208c8fc" />  

Global loss는 model framework 그림을 통해 보면 알 수 있듯 local representations에서 avr pooling 과정을 거쳐 얻은 global representation 사이의 contrastive learning을 의미한다. Raw ECG signal을 자르기 전에 같은 신호로부터 잘려진 데이터들끼리는 positive pair로 contrastive learning을 하는 것이다.  

<img width="311" height="48" alt="image" src="https://github.com/user-attachments/assets/fd6be0ac-86ec-442f-8d25-e63bc76570b6" />  

그래서 결론적으로는 L(local)과 L(global)을 합쳐서 loss로 사용하게된다.

---

## Fine-tunning
이 논문에서 해당 pre-training model의 성능을 확인하기위해 두가지 fine-tunning task로 실험을 진행한다.  

- **Cardiac Arrhythmia Classification**  
  심장의 부정맥 진단을 예측하는 분류 task  
- **Patient Identification**  
  같은 환자로부터 생성된 ECG데이터인지 맞추는 task  

---

## Experiment
실험 방법은 크게 세가지로 진행되었다.  

1. 12-lead 데이터로 pre-training을 진행한 뒤, 12-lead, 6-lead, 3-lead, 2-lead, 1-lead 데이터로 각각 finetuning 하여 task별로 평가하는 방법.  
2. 12-lead, 6-lead, 3-lead, 2-lead, 1-lead 데이터로 각각 pre-training과 fine-tunning을 모두 진행하는 방법. 예를 들면 6-lead로 pretraining, 6-lead로 fine-tunning  
3. Random lead masking이 없을 때와 있을 때의 비교  

---

## Result

### Experiment 1.

<img width="528" height="397" alt="image" src="https://github.com/user-attachments/assets/a6ee113f-d7d6-4539-92e6-19176af2ef3e" />  

W2V+CMSC+RLM은 분류 작업의 DX Task 점수에서 상당한 개선을 보여줬다. 이는 local representation (Wav2Vec)과 global representation(CMSC)을 결합하고 Random lead masking(RLM)을 적용하는 것이 리드 수가 적더라도 심장 이상을 진단하는 데 매우 효과적임을 나타낸다.

### Experiment 2.
<img width="612" height="356" alt="image" src="https://github.com/user-attachments/assets/2e219aca-0d59-4e8e-82fa-baf063ab0d65" />  

Wav2Vec 2.0 (W2V)가 12-리드를 제외한 모든 리드 조합(6-리드, 3-리드, 2-리드 등)에서 가장 좋은 점수를 보이는데, 이는 classification이 ECG 신호의 local representations)에 집중하기 때문이라고 저자들은 말하고 있다.

### Experiment 3.
<img width="656" height="311" alt="image" src="https://github.com/user-attachments/assets/4751e8f9-6107-4431-9634-9fbfa197b7a1" />  

 RLM을 W2V에 적용하면 classification 성능은 향상되지만, identification 성능은 일부 리드 조합에서 감소하는 모습을 보인다. RLM이 W2V의 local represention을 포착하는 능력을 강화하기 때문이라고 추측하고 있다.

---

## Limitation
본 논문에서 제시하는 두 가지 한계점이 존재하는데,  

1. **Hyperparameter가 optimal 하지 않다.**  
   Random Lead Masking에서의 0.5 파라미터는 실험을 통해 계산된 파라미터가 아니라고 한다. 따라서 최적의 파라미터가 아닐 가능성이 있다.
   
2. **Resampling 과정에서의 데이터 손실.**  
   ECG 데이터는 출처별로 sampling rate이 각각 다르기 때문에 전부 통일 시켜주는 resampling 과정이 필요하다. 해당 과정에서의 데이터 손실이 있을 수 있다고 말하고 있다.  

