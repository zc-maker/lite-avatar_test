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
        
        # 音视频同步参数
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.playing = False
        self.current_frame = None
        self.bg_frame = None
        self.audio_data = None
        
        # 初始化模型
        self.load_models()
        self.load_data()
        
        # 启动背景线程
        self.bg_running = True
        self.bg_thread = threading.Thread(target=self._bg_video_loop)
        self.bg_thread.start()
        self.frame_lock = threading.Lock()
        

    def load_models(self):
        logger.info("Loading models...")
        from audio2mouth_cpu import Audio2Mouth
        self.audio2mouth = Audio2Mouth(use_gpu=True)
        
        # 加载Torch模型
        self.encoder = torch.jit.load(f'{self.data_dir}/net_encode.pt').to(self.device)
        self.generator = torch.jit.load(f'{self.data_dir}/net_decode.pt').to(self.device)

    def load_data(self):
        logger.info("Loading data...")
        # 加载背景视频
        self.bg_cap = cv2.VideoCapture(f'{self.data_dir}/bg_video.mp4')
        _, self.bg_frame = self.bg_cap.read()
        
        # 加载面部参数
        self.neutral_pose = np.load(f'{self.data_dir}/neutral_pose.npy')
        
        # 加载参考图像
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
            self.bg_frame = frame
            time.sleep(1/self.fps)

    def _play_audio(self, audio_data, sr):
        """修复后的音频播放方法"""
        
        audio_data = audio_data.reshape(-1, 1) 
        def audio_callback(outdata, frames, time, status):
            nonlocal audio_pos
            if audio_pos >= len(audio_data):
                raise sd.CallbackStop
            chunksize = min(len(audio_data) - audio_pos, frames)
            outdata[:chunksize] = audio_data[audio_pos:audio_pos + chunksize]
            audio_pos += chunksize
            
        audio_pos = 0
        with sd.OutputStream(samplerate=sr, channels=1, callback=audio_callback):
            sd.sleep(int(1000 * len(audio_data) / sr))  # 这里使用传入的audio_data参数

    def _play_video(self):
        cv2.namedWindow('Avatar', cv2.WINDOW_NORMAL)
        start_time = time.time()
        frame_count = 0
        
        while self.playing:
            # 计算预期帧时间
            expected_time = start_time + frame_count / self.fps
            
            # 优先获取生成的视频帧
            try:
                frame = self.video_queue.get_nowait()
            except queue.Empty:
                frame = self.bg_frame  # 使用背景帧
            
            # 显示帧
            if frame is not None:
                # display_frame = frame
                display_frame = cv2.resize(frame, (393, 717))

                cv2.imshow('Avatar', display_frame)
                
                # 计算实际延迟
                actual_delay = max(1, int(1000*(expected_time - time.time())))
                if cv2.waitKey(actual_delay) & 0xFF == 27:
                    break
            
            frame_count += 1
        
        cv2.destroyAllWindows()

    def process_audio(self, audio_path):
        logger.info("Processing audio...")
        try:
            # 读取并预处理音频
            wav_bytes = self.read_wav_to_bytes(audio_path)
            headinfo = geneHeadInfo(16000, 16, len(wav_bytes))
            audio_data = headinfo + wav_bytes
            
            # 提取音频特征
            input_audio, sr = sf.read(BytesIO(audio_data))
            if input_audio is None:
                raise ValueError("Failed to read audio data")
            
            param_res = self.audio2mouth.inference(input_audio=input_audio)[0]
            
            # 生成视频帧
            for idx in range(len(param_res)):
                bg_id = idx % 150
                param = param_res[idx]
                param_tensor = torch.tensor([param[str(i)] for i in range(32)]).float().to(self.device)
                
                # 生成嘴部图像
                mouth_img = self.generator(self.ref_img_list[bg_id], param_tensor.unsqueeze(0))
                mouth_img = (mouth_img / 2 + 0.5).clamp(0, 1).cpu()[0].permute(1,2,0).detach().numpy() * 255
                
                # 与背景融合
                final_frame = self.blend_mouth(mouth_img.astype(np.uint8))
                self.video_queue.put(final_frame)
            
            return input_audio, sr
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return None, None

    def blend_mouth(self, mouth_img):
        # 从数据文件读取融合参数
        with open(f'{self.data_dir}/face_box.txt') as f:
            y1, y2, x1, x2 = map(int, f.readline().split())
        
        # 创建融合蒙版
        mask = np.ones((y2-y1, x2-x1, 3)) * 255
        mask[10:-10, 10:-10] = 0
        mask = cv2.GaussianBlur(mask, (21,21), 15) / 255
        
        # 融合图像
        bg = self.bg_frame.copy()
        mouth_resized = cv2.resize(mouth_img, (x2-x1, y2-y1))[:,:,::-1]
        bg[y1:y2, x1:x2] = mouth_resized * (1 - mask) + bg[y1:y2, x1:x2] * mask
        return bg.astype(np.uint8)

    def handle(self, audio_path):
        # 停止当前播放
        self.playing = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        
        # 处理音频并生成视频
        audio_data, sr = self.process_audio(audio_path)
        if audio_data is None:
            logger.error("No audio data to play")
            return
        
        # 启动播放线程
        self.playing = True
        self.audio_thread = threading.Thread(target=self._play_audio, args=(audio_data, sr))
        self.video_thread = threading.Thread(target=self._play_video)
        
        self.audio_thread.start()
        self.video_thread.start()
        
        # 等待播放完成
        self.audio_thread.join()
        self.playing = False
        self.video_thread.join()

    @staticmethod
    def read_wav_to_bytes(file_path):
        try:
            with wave.open(file_path, 'rb') as wav_file:
                return wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            logger.error(f"Error reading WAV file: {e}")
            return None

if __name__ == '__main__':
    # 初始化系统
    avatar = liteAvatar(
        data_dir='./data/preload',
        num_threads=2,
        fps=30,
        use_gpu=True
    )
    
    # 测试三次调用
    test_file = 'asr_example.wav'
    try:
        print("=== 第一次处理 ===")
        avatar.handle(test_file)
        
        print("=== 第二次处理 ===")
        avatar.handle(test_file)
        
        print("=== 第三次处理 ===")
        avatar.handle(test_file)
    finally:
        avatar.bg_running = False
        avatar.bg_thread.join()
        cv2.destroyAllWindows()