import numpy as np
import cv2
import time
import threading
import queue
from loguru import logger
import soundfile as sf
import sounddevice as sd
from io import BytesIO
from torchvision import transforms
import torch
import wave

def geneHeadInfo(sampleRate, bits, sampleNum):
    import struct
    rHeadInfo = b'\x52\x49\x46\x46'
    fileLength = struct.pack('i', sampleNum + 36)
    rHeadInfo += fileLength
    rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'
    rHeadInfo += struct.pack('i', sampleRate)
    rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))
    rHeadInfo += b'\x02\x00'
    rHeadInfo += struct.pack('H', bits)
    rHeadInfo += b'\x64\x61\x74\x61'
    rHeadInfo += struct.pack('i', sampleNum)
    return rHeadInfo

class liteAvatar(object):
    def __init__(self, data_dir=None, num_threads=1, fps=30, use_gpu=True):
        logger.info('Initializing liteAvatar...')
        self.data_dir = data_dir
        self.fps = fps
        self.num_threads = num_threads
        self.device = "cuda" if use_gpu else "cpu"
        
        # 同步参数
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.playing = False
        self.current_frame = None
        self.bg_frame = None
        self.audio_data = None
        self.video_running = True
        self.param_res = []
        self.audio_start_time = 0
        self.frame_event = threading.Event()
        
        # 初始化模型
        self.load_models()
        self.load_data()
        
        # 启动后台线程
        self.bg_running = True
        self.bg_thread = threading.Thread(target=self._bg_video_loop)
        self.bg_thread.start()
        
        # 启动独立视频线程
        self.video_thread = threading.Thread(target=self._video_main_loop)
        self.video_thread.start()
        
        self.frame_lock = threading.Lock()

    def load_models(self):
        logger.info("Loading models...")
        from audio2mouth_cpu import Audio2Mouth
        self.audio2mouth = Audio2Mouth(use_gpu=True)
        
        self.encoder = torch.jit.load(f'{self.data_dir}/net_encode.pt').to(self.device)
        self.generator = torch.jit.load(f'{self.data_dir}/net_decode.pt').to(self.device)

    def load_data(self):
        logger.info("Loading data...")
        self.bg_cap = cv2.VideoCapture(f'{self.data_dir}/bg_video.mp4')
        _, self.bg_frame = self.bg_cap.read()
        
        self.neutral_pose = np.load(f'{self.data_dir}/neutral_pose.npy')
        
        self.ref_img_list = []
        transforms_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        for ii in range(150):
            img_path = f'{self.data_dir}/ref_frames/ref_{ii:05d}.jpg'
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (384, 384))
            tensor_img = transforms_list(image).unsqueeze(0).to(self.device)
            self.ref_img_list.append(self.encoder(tensor_img))

    def _bg_video_loop(self):
        while self.bg_running:
            ret, frame = self.bg_cap.read()
            if not ret:
                self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self.frame_lock:
                self.bg_frame = frame
            time.sleep(1/self.fps)

    def _video_main_loop(self):
        """持续显示视频的主循环"""
        cv2.namedWindow('Avatar', cv2.WINDOW_NORMAL)
        start_time = time.time()
        frame_count = 0
        
        while self.video_running:
            try:
                frame = self.video_queue.get_nowait()
            except queue.Empty:
                with self.frame_lock:
                    frame = self.bg_frame.copy() if self.bg_frame is not None else None
            
            if frame is not None:
                display_frame = cv2.resize(frame, (393, 717))
                cv2.imshow('Avatar', display_frame)
                
                expected_time = start_time + frame_count / self.fps
                actual_delay = max(1, int(1000*(expected_time - time.time())))
                key = cv2.waitKey(actual_delay)
                if key & 0xFF == 27:
                    break
                
                frame_count += 1
            else:
                cv2.waitKey(1)

    def _play_audio(self, audio_data, sr):
        """优化后的音频播放方法"""
        def audio_callback(outdata, frames, time_info, status):
            nonlocal audio_pos
            if audio_pos >= len(audio_data):
                raise sd.CallbackStop
            chunksize = min(len(audio_data) - audio_pos, frames)
            outdata[:chunksize] = audio_data[audio_pos:audio_pos + chunksize]
            audio_pos += chunksize
            
            # 记录音频开始时间
            if audio_pos == chunksize:
                self.audio_start_time = time.time()
                self.frame_event.set()
        
        audio_data = audio_data.reshape(-1, 1)
        audio_pos = 0
        with sd.OutputStream(samplerate=sr, channels=1, callback=audio_callback):
            sd.sleep(int(1000 * len(audio_data) / sr))

    def _generate_frames(self):
        """视频帧生成方法"""
        self.frame_event.wait()  # 等待音频开始
        start_time = self.audio_start_time
        
        for idx, param in enumerate(self.param_res):
            # 计算目标时间
            target_time = start_time + idx / self.fps
            while time.time() < target_time:
                time.sleep(0.001)
            
            # 生成嘴部图像
            bg_id = idx % 150
            param_tensor = torch.tensor([param[str(i)] for i in range(32)]).float().to(self.device)
            mouth_img = self.generator(self.ref_img_list[bg_id], param_tensor.unsqueeze(0))
            mouth_img = (mouth_img / 2 + 0.5).clamp(0, 1).cpu()[0].permute(1,2,0).detach().numpy() * 255
            
            # 融合背景
            final_frame = self.blend_mouth(mouth_img.astype(np.uint8))
            self.video_queue.put(final_frame)

    def process_audio(self, audio_path):
        import librosa
        logger.info("Processing audio...")
        try:
            
            wav_bytes = self.read_wav_to_bytes(audio_path)
            if wav_bytes is None:
                return None, None
                
            headinfo = geneHeadInfo(16000, 16, len(wav_bytes))

            audio_data = headinfo + wav_bytes
            
            # input_audio, sr = sf.read(BytesIO(audio_data))
            # 修改点：增加重采样逻辑
            with BytesIO(audio_data) as bio:
                input_audio, sr = sf.read(bio)
                
                # 新增2：自动重采样到16000Hz
                orig_sr=24000
                if orig_sr != 16000:
                    input_audio = librosa.resample(
                        input_audio.T if input_audio.ndim > 1 else input_audio,  # 处理多声道
                        orig_sr=orig_sr,
                        target_sr=16000,
                        # res_type='soxr_hq'
                    )
                    sr = 16000  # 强制更新采样率
                    
                    # 幅度标准化
                    input_audio = input_audio / np.max(np.abs(input_audio))
            
            # if input_audio is None:
            #     raise ValueError("Failed to read audio data")
            
            self.param_res = self.audio2mouth.inference(input_audio=input_audio)[0]
            return input_audio, sr
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return None, None

    def blend_mouth(self, mouth_img):
        with open(f'{self.data_dir}/face_box.txt') as f:
            y1, y2, x1, x2 = map(int, f.readline().split())
        
        mask = np.ones((y2-y1, x2-x1, 3)) * 255
        mask[10:-10, 10:-10] = 0
        mask = cv2.GaussianBlur(mask, (21,21), 15) / 255
        
        with self.frame_lock:
            bg = self.bg_frame.copy()
        mouth_resized = cv2.resize(mouth_img, (x2-x1, y2-y1))[:,:,::-1]
        bg[y1:y2, x1:x2] = mouth_resized * (1 - mask) + bg[y1:y2, x1:x2] * mask
        return bg.astype(np.uint8)

    def handle(self, audio_path):
        # 停止当前播放
        self.playing = False
        self.frame_event.clear()
        
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        if hasattr(self, 'generate_thread'):
            self.generate_thread.join()
        
        # 清空旧帧
        while not self.video_queue.empty():
            self.video_queue.get()
        
        # 处理音频
        audio_data, sr = self.process_audio(audio_path)
        if audio_data is None:
            return
        
        # 启动音频和视频生成线程
        self.playing = True
        self.audio_thread = threading.Thread(target=self._play_audio, args=(audio_data, sr))
        self.generate_thread = threading.Thread(target=self._generate_frames)
        
        self.audio_thread.start()
        self.generate_thread.start()
        
        self.audio_thread.join()
        self.generate_thread.join()
        self.playing = False

    @staticmethod
    def read_wav_to_bytes(file_path):
        try:
            with wave.open(file_path, 'rb') as wav_file:
                return wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            logger.error(f"Error reading WAV file: {e}")
            return None

    def release(self):
        """资源释放"""
        self.bg_running = False
        self.video_running = False
        self.frame_event.set()
        self.bg_thread.join()
        self.video_thread.join()
        cv2.destroyAllWindows()
        self.bg_cap.release()

if __name__ == '__main__':
    import os
    avatar = liteAvatar(
        data_dir='./data/preload',
        num_threads=2,
        fps=30,
        use_gpu=True
    )
    wav_file="./wav"
    wav_path=[wav_file+"/"+x for x in os.listdir(wav_file) if x.endswith(".wav")]
    try:

        for x in wav_path:
            avatar.handle(x)
        while True:
            time.sleep(1)  
    except KeyboardInterrupt:
            pass
    finally:
            avatar.release()
         
         
# 下个版本
# 使用队列的方式，在播放上一个的时候，直接处理下一个


    # try:
    #     test_file = 'asr_example.wav'
    #     print("=== 第一次处理 ===")
    #     avatar.handle(test_file)
        
    #     print("=== 第二次处理 ===")
    #     avatar.handle(test_file)
        
    #     print("=== 第三次处理 ===")
    #     avatar.handle(test_file)
        
    #     # 保持窗口打开
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     avatar.release()