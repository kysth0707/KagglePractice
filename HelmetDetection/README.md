-1. 쓸 라이브러리 고르기 ( 1.WhichLibraryToUse.ipynb )  
- 어느 라이브러리를 쓸 지에 대해 고민  
결과 
https://github.com/kairess/torchvision_walkthrough/blob/master/keypoints.ipynb  
위를 반드시 사용해야 함!  
밑의 코드를 추가하는 것도 좋아 보임.
https://github.com/kairess/torchvision_walkthrough/blob/master/segmentation.ipynb
  
-실패 Check helmet region ( Failure )  
- ( -2.CheckHelmetRegion.ipynb )  


-2. 헬맷 구역 판별 ( 2.CheckHelmetRegion.ipynb )  
- 라이브러리를 어떻게 활용할 지에 대해 고민  
결과  
객체 판별 라이브러리 + 사람 형태 확인 라이브러리 섞어 쓰기
  
-2-1. 함수화 ( HelmetRegionChecker.py, 2-1.FunctionCheck.ipynb )

-3. 이미지 사람만 추출 ( 3.ImageClearing.ipynb )
- 이미지에서 사람만 추출합니다

-4. 헬맷 씌우기 ( 4.AddHelmetToFace.ipynb )
- 이미지에서 얼굴을 인식해 흰색 안전모를 덮어씌워 줌  
결과  
잘 됨. 하지만 이로 인해 좋은 학습 결과가 나올지는 미지수  
  
-4-1. 헬맷 씌워주기 함수화 ( HelmetGenerator.py, 4-1.HelmetGenerator.ipynb )  
  
-5.모든 사진 헬맷 씌워서 저장시키기 ( 5.ConvertAllImage.ipynb )  
- 이미지 생성  
결과    
정상적으로 잘 됨 

-6. 모델 제작 및 학습 ( 6.Learning.ipynb )
- 모델을 제작하고 학습시킵니다 ( Tensorflow.model.Sequential )  
정확도 99% 에서 학습을 중지함

-7. 테스트 ( 7.Test.ipynb )
- 위에서 적용한 함수들을 전부 사용해보며 테스트합니다.  
결과  
실패  

-8. 이미지 변경 2차
- with_helmet 이미지와 without_helmet 이미지의 사이즈 크기가 원인일 것 같아 다시 변경했습니다

-9. 학습 2차
- 동일 학습 코드로 재 학습합니다

-10. 테스트 2차
- 다시 테스트를 진행합니다  
결과  
정확도가 굉장히 낮습니다.. 4명 중 1명 만 착용 중일 것이라고 예측함.  

10 까지의 결과  
초록 : 헬맷 씀 / 빨강 : 헬맷 안 씀
![image](https://user-images.githubusercontent.com/83905675/213908226-929b2ab6-1829-495c-a8ce-017d597b3073.png)
![image](https://user-images.githubusercontent.com/83905675/213908229-72aca9bc-02c7-49c0-98ac-bee2f6c4497e.png)


References  
  
Image  
http://www.newspeak.kr/news/articleView.html?idxno=220494  
  
.etc  
https://www.youtube.com/watch?v=MC6jm28_LHo  
https://www.kaggle.com/datasets/ashwingupta3012/human-faces?select=Humans  
  
https://www.youtube.com/watch?v=ncIyy1doSJ8  
https://github.com/kairess/mask-detection  
https://github.com/prajnasb/observations  
  
https://pinkwink.kr/1124  
https://github.com/opencv/opencv/tree/master/data/haarcascades  
  
https://www.youtube.com/watch?v=WgsZc_wS2qQ  
https://github.com/kairess/torchvision_walkthrough  
https://eehoeskrap.tistory.com/463  
https://wikidocs.net/52846  
  
https://www.youtube.com/watch?v=C6qZXg4fLPY  
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
https://webnautes.tistory.com/1410  
  
https://codetorial.net/tensorflow/classifying_the_cats_and_dogs.html  