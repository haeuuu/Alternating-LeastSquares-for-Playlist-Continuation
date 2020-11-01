# :melon: Melon playlist continuation :musical_note:



### Introduction

`축 처진 퇴근길을 위한 그루브`, `기분 좋은 밤 산책을 위한 적당한 텐션의 음악`, `무드등 켜고 혼술하며 듣는 감각적인 그루브` ...

이렇게 만들어진 플레이 리스트가 과연 오직 30개의 장르에만 의존할까요?

실제 유저들은 대분류 30개나 소분류 224개보다도 **더 세세하게 음악을 분류할 것**이라고 생각했습니다.

K-means clustering을 통해 **더욱 세분화된 장르 군집 1000개**를 찾고, 각 군집 내에서 상위 노래를 추천합니다.



:heavy_check_mark: [GitHub에서 확인하기 (model1-2) ](https://github.com/haeuuu/RecSys-For-Melon-playlist-continuation)

:heavy_check_mark: [GitHub에서 확인하기 (model3) ](https://github.com/haeuuu/RecSys-For-Melon-playlist-continuation2)

:heavy_check_mark: [Tistory에서 확인하기](https://hhhaeuuu.tistory.com/111?category=941513)



### Keywords

- K-means clustring
- PCA
- SVD

 

## 0. 데이터 전처리 및 song embedding 생성



### song에 대한 one-hot vector 생성

> 노래의 meta 정보에 playlist에서 추출한 태그 정보를 이용하여 one-hot vector ( d = (n , 457) ) 를 생성합니다.



:green_heart: **사용한 feature : 년 / 월 / 장르 대분류 / 장르 소분류 / 고빈도 태그**

1. Issue data를 년/월로 잘라 총 96개의 범주를 생성

   ##### 결측치 처리

   - 월 정보가 `00` 이거나 `51`인 경우, 또는 년도가 `0000`이거나 `2020`인 노래가 존재한다.
   - 총 2801개의 결측치 중에서 **해당 노래가 포함된 앨범을 찾아** 총 8개의 결측치를 채웠다.
   - 나머지 결측치에 대해서는 정보 손실을 감안하고 `0000/00/00`으로 수정하여 년도 범주와 월 범주에 각각 `0000`,`00`을 추가하였다.

   ##### 월로 잡을 수 있는 정보와 년도로 잡을 수 있는 정보는 다르다.

   - 6~7월에는 여름에 듣기 좋은 댄스곡이, 12월에는 크리스마스를 겨냥한 캐롤이 발매된다. 월/계절 정보가 핵심이 된다.
   - 제국의 아이들 등 오래전 아이돌 노래를 위한 플레이 리스트 또는 80년대 추억의 노래 등은 년도 정보가 중요한 역할을 한다.

2. 같은 앨범에 있는지 아닌지에 대한 정보는 사용하지 않았다.

   - 동일 앨범 id를 가진 노래의 갯수에 대해 3분위수(75%)값이 2였다.

     즉 대부분이 다른 앨범에 들어가 있기 때문에 앨범 정보는 유용하지 않다고 판단하였다.

   - 또한 같은 앨범일지라도 다양한 장르와 분위기의 노래가 수록된다. 제외한다.

3. 소분류, 대분류 정보 30 + 224개를 이용한다.

4. train set을 이용하여 노래에 tag를 매칭시킨다.

   * 해당 노래가 속한 play list를 모두 찾아 태그를 추출한다.
   * 이 때 tag는 train data 전체에서 가장 많이 등장한 상위 110개만을 사용하였다.



### PCA를 통한 차원 축소

* 457차원의 one-hot vector를 180차원으로 축소합니다.
* explained_variance_ratio = 0.9438



### K-means clustering

> k-means clustering을 통해 1800/1000개의 새로운 장르를 추출합니다.

* ==군집 평가 결과==



## 1. Matrix Factorization

`playlist`를 `user`로, k-means clustering으로 얻어낸 새로운 장르(군집)을 `item`으로 보고 `user-item matrix`를 생성합니다.



### model 1 : 빈도로 정의된 rating을 이용한다.

* `playlist_i`의 `cluster_j`에 대한 implicit feedback은 해당 playlist에 속한 cluster의 갯수로 정의합니다.
  * ex : `[song1 (cluster1), song10(cluster2), song32(cluster2), song32(cluster2), song97(cluster1)]` 은 cluster 1, 2에 대해 2점, 3점을 가지게 됩니다.
* ==점수의 분포는? 그대로 써도 괜찮은가?==



* surprise 패키지를 이용하여 SVD를 진행합니다.

* latent factor = 512 , epoch = 50
* RMSE = 0.1926



==점수 넣기==

```

```



### model 2 : log(rating + k)를 이용한다.





## 2. implicit ALS

* Alternating Least Square model



```
Music nDCG: 0.085352
Tag nDCG: 0.261288
Score: 0.111742
```



## 3. AutoEncoder



```
Music nDCG: 0.0087537
Tag nDCG: 0.260441
Score: 0.0465068
```



# :art: 소감

실습 수업때도, 연구실에서도 항상 다짐했던 것이 있었는데 바로 '이론만 알지 말고 조그맣게라도 적용해보자!' 였다.

이론상으로는 별탈 없이 잘될거같지만 실제로 적용해보면 예상치 못한 부분에서 이런저런 문제를 만난다. 행렬이 너무 커진다던가, 모델이 너무 무거워진다던가, 혹은 데이터 가공부터가 난관인 경우도 있다.

movielens 데이터를 이용하여 이론 공부와 조그마한 실습을 병행했는데 아무래도 모르는 영화가 대부분이다보니 EDA를 할 때도, 나만의 가설을 세울 때도, 추천 결과를 해석할 때도 어려움이 있었다. 내가 주도적으로 분석하고 모델을 진행시켜나간다는 느낌보다는 영화/감독/배우 등 정형화된 틀을 벗어나지 못하고 그대로 따라하고 있다는 느낌을 받았다.

좀 더 나에게 와닿는, 내가 약간의 지식도 가지고 있는 그런 데이터가 없을까 하다가 때마침 멜론 데이터를 만나게 되었다 !

직접 추천 결과를 확인할 때도 어떤식으로 추천되는지를 파악할 수 있어 훨씬 재미있었고, 내가 노래를 듣는 사람의 입장이 되어 여러가지 가설을 세울 때도 수월했다 !

제대로된 첫 도전이기 때문에 '높은 점수를 얻어서 순위를 올리자'보다는 실제 데이터에 적용할 때 어떤 문제를 만날 수 있는지, 그리고 이를 해결하려면 어떤 테크닉을 써야 하는지 등을 얻어가야겠다는 목표를 가지고 천천히 도전했다.

낮은 점수로라도 제출을 해보고 싶었지만 연구 인턴에서도 진행하고 있는 것들이 있었기 때문에 기간을 맞추지는 못했다.
