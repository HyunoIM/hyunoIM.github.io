---
layout: single
title:  "[Study] Optimizer"
permalink: study/optimizer
toc: true
toc_sticky: true
categories: 
  - study
---

## **Optimizer**

앞에서 본 loss function은 단순히 평가 지표로만 사용되지 않는다. 이 손실함수 값을 줄여나가기 위해 모델의 parameter를 조정하는 작업을 **optimization(최적화)**라고 한다.

그리고 이를 수행하는 알고리즘이 **optimizer** 이다.

***

<img src="https://velog.velcdn.com/images/freesky/post/57e14895-6eb0-4c86-a9d1-0acdb0398406/image.png" alt="img" style="zoom:50%;" />
[https://www.slideshare.net/slideshow/ss-79607172/79607172#49](https://www.slideshare.net/slideshow/ss-79607172/79607172#49)

<br>  


gradient descent(경사 하강법)과 같은 optimizer는 일반적으로 손실 함수의 gradient를 사용한다. gradient는 다변수 함수의 도함수(derivative)로, 특정 지점에서 함수 출력값이 변화하는 정도와 방향을 설명한다. 따라서 손실 함수가 모든 지점에서 gradient를 가질 수 있도록, 즉 미분 가능하도록 설계되는 것이 중요하다.

딥러닝에서는 nonlinear activation function를 가진 여러 layer의 상호 연결된 뉴런으로 구성된 대규모 neural network을 활용하기 때문에 전체를 미분하려면 엄청나게 많은 양의 변수와 활성화 함수 각각에 대해 편미분을 계산해야 한다.

이 지점에서 나오는 개념이 **backpropagation(역전파)**이다!

출력 층에서 입력 층까지 네트워크를 역방향으로 통과하며, chain rule을 사용하여 네트워크의 각 weight와 bias가 전체 손실에 어떻게 기여했는지를 계산한다.

역전파에 대한 자세한 설명은 [[cs231n] 4. Introduction to Neural Networks](https://jdvvd.github.io/cs231n/4)에 정리해두었다.


![img](https://velog.velcdn.com/images/freesky/post/607dea70-976d-4dfb-b875-cd75ea348aa8/image.gif)


<br>  


**Reference**

[IBM/think/topics(https://www.ibm.com/think/topics/loss-function)](https://www.ibm.com/think/topics/loss-function)

[https://www.slideshare.net/slideshow/ss-79607172/79607172#49](https://www.slideshare.net/slideshow/ss-79607172/79607172#49)
