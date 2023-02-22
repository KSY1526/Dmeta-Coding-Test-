# Dmeta-Coding-Test

## 코드 재현 방법
* command line 내 다음 내용 타이핑
~~~
conda create -n dmeta2 python=3.8
conda activate dmeta2
pip install -r requirements.txt

python cut_image.py
python merge_image.py

# 옵션 사용시
python cut_image.py --image cat.1(파일이름) --nums 2(이미지 분할 크기)
python merge_image.py --image cat.1(파일이름) --nums 2(이미지 분할 크기)
~~~

## 폴더 구조

- ./sample : 사용하는 이미지 담는 폴더
- ./save : 분할 이미지 저장되는 폴더 
- requirements.txt : 필수 패키지 버전 관리
- cut_image.py : 원본 이미지를 자른 뒤 ./save에 저장
- merge_image.py : 분할 이미지를 이용해 원본 이미지 복원 시도