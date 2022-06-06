# Realtime_Voice_Activity_Detection
Making Real time voice activity detection model

입력으로 들어오는 오디오가 말소리인지 아닌지를 실시간으로 판단하는 모델을 만듭니다.

데이터셋 선정 - audio slicing - data augmentation - training - test의 순으로 진행합니다.

1. 데이터셋 선정
preprocessing/1.OtherSound_audio_classification.ipynb 파일로 진행(음성 데이터 선정 코드는 작업중)

2. audio slicing: 학습 데이터 파일을 일정 길이로 자릅니다.
preprocessing/Slicing_audio.ipynb 파일로 진행

3. data augmentation: RIR 생성, 음높이 조절, 진폭 조절 등
preprocessing/Audio_augmentation.ipynb 파일로 진행

4. training: CNN 학습
training/4. create_log_mel_spectrogram_and_training.ipynb 파일로 진행

5. test
```
python3 realtime_detection.py
```

### Dataset link
- Speaking: AIHUB 등
- OtherSound: Freesound https://freesound.org/
