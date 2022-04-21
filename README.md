# CloseCV
## 휴스타 4기 이미지 프로세싱 1조 CloseCV

얼굴 인식 후 화면 중앙과 사용자의 얼굴 사이의 x, y, z의 좌표 차이값 및 roll 각도 차이값을 계산해주는 라이브러리.

## 설치
프로젝트 클론.
```shell
git clone https://github.com/Junu-Park/CloseCV.git
```

## Pip에서 설치
NumPy, OpenCV, Dlib을 설치 해야함.
```shell
pip install -r requirements.txt
```

## 아나콘다에서 설치
NumPy, OpenCV, Dlib을 설치 해야함.
```shell
conda env create --file enviroment.yml
conda activate CloseCV
```

## 실행
```shell
python close_cv_example.py
```
