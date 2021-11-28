import numpy as np
# from numpy_ringbuffer import RingBuffer
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import pyaudio
import wave # for save audio file
import datetime
import os
from collections import deque
import math

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load segment audio classification model
model_path = r"audio_model/"
model_name = "audio_CNN_2021_09_07_12_19_57_acc_97.53"

# Model reconstruction from JSON file
with open(model_path + model_name + '.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['Speaking', 'OtherSound'])

# Some Utils

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4)) #figsize=20,4
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

# 스펙트로그램: 가로축은 시간, 세로축은 주파수, 진한 정도는 진폭을 나타냅니다.
def draw_spectrogram(X):
    X = librosa.effects.preemphasis(X)
    clip, index = librosa.effects.trim(X, top_db=20, hop_length=256)
    stfts = librosa.stft(clip, n_fft=512, hop_length=256, win_length=512)   
    stftsdb = librosa.amplitude_to_db(abs(stfts))
    plt.figure(figsize=(20,4))
    librosa.display.specshow(stftsdb, sr=22050, 
                             hop_length=256,
                             x_axis='s', y_axis='hz')
    plt.colorbar()
    
# 입력 오디오를 normalization하는 함수입니다.
def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

# 입력 오디오 buffer가 말소리인지 아닌지를 판별하는 함수입니다.
def predictSound(X):

    #Triming: 입력 오디오에서 무음 구간을 제거합니다.
    clip, index = librosa.effects.trim(X, top_db=20, frame_length=512, hop_length=256) 

    # Trimming
    X, index = librosa.effects.trim(X, top_db=20, frame_length=512,hop_length=256)
    
    #get mel-spectrogram: 입력 오디오로부터 mel-spectrogram feature 추출
    X = librosa.feature.melspectrogram(y=X, sr=16000, n_fft=512, hop_length=256, win_length=512)
    X = librosa.power_to_db(X, ref=np.max)
    X = X.T
    X = np.mean(X, axis=0)
    X = minMaxNormalize(X)
    X = np.reshape(X, (1, 16, 8))

    # get prob
    result = model.predict(np.array([X]))
    predictions = [np.argmax(y) for y in result]
    prob = np.max(result)
    result = lb.inverse_transform([predictions[0]])[0]
    #print('predict: ', result, round(prob, 2))
    return result, prob


# 현재 입력값의 dB를 출력하는 함수입니다.
def showdB(y):  # y, sr =librosa.load(...) 
    clip, index = librosa.effects.trim(y, top_db=20, frame_length=512, hop_length=256)
    stfts = librosa.stft(clip, n_fft=512, hop_length=256, win_length=512)
    dB = librosa.amplitude_to_db(abs(stfts), ref=1/1000)
    dB = np.mean(dB)
    return dB 


def pcm2float(sig, dtype='float32'):
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

# 발화 끝으로 갈수록 에너지가 약해지므로 정확도가 낮아집니다.
# 이를 보완하기 위해 후처리 과정에서 이동평균을 이용합니다.
class MovingAverage:
    def __init__(self, size: int):
        self.data = deque(maxlen = size)

    def next(self, val: int) -> float:
        self.data.append(val)
        return sum(self.data)/len(self.data)


class RealtimeRecording():
    def __init__(self):
        self.CHUNKSIZE = 8192 # 8192: 256ms. 입력 오디오 신호를 256ms 단위로 받습니다.
        self.RATE = 16000 # sample rate
        self.FORMAT = pyaudio.paInt16 
#       self.FORMAT = pyaudio.paFloat32 # original = paFloat32 
        self.CHANNELS = 1 # mono

        self.audio_buffer = bytes()
        self.ma = MovingAverage(3)
        self.STATE = False

        # for saving speaking buffers
        self.speaking_buffer = np.array([])
        self.SAVE = False
        self.previous_result = ''
        self.category = 0.0

        
    def start(self):
        # initialize portaudio
        print("Stream Start")
        now = datetime.datetime.now()
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
        channels=self.CHANNELS,
        rate=self.RATE, input=True,
        frames_per_buffer=self.CHUNKSIZE)

        if not self.STATE:
            self.audio_buffer += audio_data.data

            if len(self.audio_buffer) == 20480: #10240 = 640ms. np.frombuffer를 거치면 320ms
                data = self.audio_buffer
                self.noise_sample = np.frombuffer(data, dtype=np.int16)

                self.noise_sample = np.nan_to_num(self.noise_sample)
                self.noise_sample_float = pcm2float(self.noise_sample)
                #plotAudio2(self.noise_sample_float)
                self.audio_buffer = bytes()
                self.STATE = True
                print('Noise reduction setting complete')


        if self.STATE:
            self.audio_buffer += audio_data.data
            
            if len(self.audio_buffer) == 8192: #8192: 256ms

                data = self.audio_buffer
                self.sample = np.frombuffer(data, dtype=np.int16)
                self.sample_float = pcm2float(self.sample)

                # nan 값 발견 시 제거
                if not np.isfinite(self.sample_float).all():
                   self.sample = np.nan_to_num(self.sample)

                # 노이즈 샘플로 노이즈 제거
                noisy_part = self.noise_sample_float
                self.current_window = nr.reduce_noise(y=self.sample_float, 
                y_noise=noisy_part, prop_decrease=1.0, sr=16000)

            
                # dB Threshold. 특정 dB 이상의 오디오에 대해서만 판별을 수행합니다.
                current_dB = showdB(self.current_window)
                dB_threshold = 16 
                
                # predict
                self.pred, self.prob = predictSound(np.array(self.current_window))

                # dB filtering and hangover
                # 이전 buffer의 상태에 따라서 speaking 판단 여부를 조금씩 조정합니다.
                if current_dB > dB_threshold:
                    # false positive를 줄이기 위해 설정한 값입니다. 사용 환경에 따라서 조정할 수 있습니다.
                    if self.pred == 'Speaking' and self.prob > 0.75: 
                        #print('pred: ', self.pred, round(self.prob,2))
                        self.result = 'Speaking'
                        self.category = self.ma.next(1)
                        #print('result: ', self.result, self.category, 'loud speaking')

                    else:
                        #print('pred: ', self.pred, round(self.prob,2))
                        if self.previous_result == 'Speaking' and self.category >0.7:
                            #print('previous: ', self.previous_result)
                            self.result = 'Speaking'
                            self.category = self.ma.next(0)
                            #print('result: ', self.result, self.category, 'possible speaking')
                        else:
                            #print('previous: ', self.previous_result)
                            self.result = 'OtherSound'
                            self.category = self.ma.next(0)
                            #print('result: ', self.result, self.category, 'loud othersound')

                else:
                    #print('pred: ', self.pred, round(self.prob,2))
                    if self.previous_result == 'Speaking' and self.category >0.5:
                        #print('previous: ', self.previous_result)
                        self.result = 'Speaking'
                        self.category = self.ma.next(0)
                        #print('result: ', self.result, self.category, 'quite speaking')
                    else:
                        #print('previous: ', self.previous_result)
                        self.result = 'OtherSound'
                        self.category = self.ma.next(0)
                        #print('result: ', self.result, self.category, 'quite othersound')

                now = datetime.datetime.now()
                print('final result: ', self.result, round(self.category,2))
                #print('dB: ', round(current_dB, 2))
                #print('*'*20)

                # maximum length of speaking buffer
                max_buffer_len = 16000 * 10 # 10S
                if self.category != 0 and len(self.speaking_buffer) < max_buffer_len:
                    self.speaking_buffer = np.concatenate((self.speaking_buffer, self.current_window))
                    self.SAVE=True
                else:
                    self.SAVE=False

                     
                
                self.audio_buffer = bytes()

        # Saving speaking buffer (optional)
        try:
            now = datetime.datetime.now()
            if self.SAVE == False and len(self.speaking_buffer) !=0:
                speaking_length = len(self.speaking_buffer) / 16000.
                print('speaking_length: ', speaking_length, 's')
                self.speaking_buffer = np.array([])
        except AttributeError:
            pass

        # audio_buffer = []
        # frames = []
        

        # for i in range(0, int(self.RATE / self.CHUNKSIZE * self.RECORD_SECONDS)):
        #     data = stream.read(self.CHUNKSIZE)
        #     current_window = np.frombuffer(data, dtype=np.int16) # dtype=np.float32

        #     audio_buffer = np.concatenate((audio_buffer, current_window))
     
        # noisy_part = audio_buffer[0:20480] # 주변 소음을 수집한 뒤 noise reduction을 수행합니다.
        # audio_buffer = nr.reduce_noise(y = audio_buffer, y_noise=noisy_part, sr=16000)

        
        
        # close stream
        now = datetime.datetime.now()
        print(now)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('End.')
        
        return audio_buffer

if __name__ == '__main__':
    rr = RealtimeRecording()
    audio_data = rr.start()
    predicted = predictSound(np.array(audio_data))
    print(predicted)
