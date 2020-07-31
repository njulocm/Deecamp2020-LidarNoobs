import pandas as pds
import numpy as np
import sys

from threading import Timer
import threading
import _thread
import time
from queue import Queue

import wave
import pyaudio


def alert_count(frames_info, MIN_NUM, ALERT_DISTANCE):
    '''用来检测是否达到警报标准，其实也就是判断有危险的帧数有没有达到阈值'''
    alert_levels = []
    for level in range(len(ALERT_DISTANCE)):
        temp_frames_info = frames_info[frames_info['dist'] <= ALERT_DISTANCE[level]]
        if len(temp_frames_info['image_idx'].unique()) >= MIN_NUM:
            alert_levels.append(level)
    return alert_levels


def play_alert_audio(wave_filename):
    '''播放wavefile'''
    chunk = 1024
    wavefile = wave.open(wave_filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wavefile.getsampwidth()),
                    channels=wavefile.getnchannels(),
                    rate=wavefile.getframerate(),
                    output=True)
    data = wavefile.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wavefile.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()


def alert_car(frames_car_info):
    '''对car的边框进行判断，发出警报'''
    pass


def alert_truck(frames_truck_info):
    '''对truck的边框进行判断，发出警报'''
    pass


def alert_tricar(frams_tricar_info):
    '''对tricar的边框进行判断，发出警报'''
    pass


def alert_cyclist(frames_cyclist_info):
    '''对cyclist的边框进行判断，发出警报'''
    global alert_audio_now
    # print('检测周围是否有自行车...')
    frames_cyclist_info = frames_cyclist_info[frames_cyclist_info['dist']
                                              <= ALERT_DISTANCE_CYCLIST[-1]]

    # 分成左前方、右前方、左后方和右后方分别检测
    # (1) 先是检测左前方
    frames_left_front = frames_cyclist_info[
        np.logical_and(frames_cyclist_info['y'] >= 0,
                       frames_cyclist_info['x'] >= 0)]
    alert_content_left_front = ['左前方自行车一级警报',
                                '左前方自行车二级警报',
                                '左前方自行车三级警报']
    alert_level_left_front = alert_count(frames_left_front,
                                         MIN_CYCLIST_FRAMES,
                                         ALERT_DISTANCE_CYCLIST)
    # 处理警报，这边只要发出最高级别的警报
    if len(alert_level_left_front) > 0:
        print(alert_content_left_front[alert_level_left_front[0]])
        # 音频警报部分
        if alert_level_left_front[0] <= ALERT_AUDIO_LEVEL:
            alert_audio_path = cyclist_audios_path[0]
            alert_audio_now.append(alert_audio_path)
            if not alert_audio_path in alert_audio_before:
                alert_queue.put(alert_audio_path)

    # (2) 再检测右前方
    frames_right_front = frames_cyclist_info[
        np.logical_and(frames_cyclist_info['y'] < 0,
                       frames_cyclist_info['x'] >= 0)]
    alert_content_right_front = ['右前方自行车一级警报',
                                 '右前方自行车二级警报',
                                 '右前方自行车三级警报']
    alert_level_right_front = alert_count(frames_right_front,
                                          MIN_CYCLIST_FRAMES,
                                          ALERT_DISTANCE_CYCLIST)
    # 处理警报，这边只要发出最高级别的警报
    if len(alert_level_right_front) > 0:
        print(alert_content_right_front[alert_level_right_front[0]])
        # 音频警报部分
        if alert_level_right_front[0] <= ALERT_AUDIO_LEVEL:
            alert_audio_path = cyclist_audios_path[1]
            alert_audio_now.append(alert_audio_path)
            if not alert_audio_path in alert_audio_before:
                alert_queue.put(alert_audio_path)

    # (3) 检测左后方
    frames_left_behind = frames_cyclist_info[
        np.logical_and(frames_cyclist_info['y'] >= 0,
                       frames_cyclist_info['x'] <= 0)]
    alert_content_left_behind = ['左后方自行车一级警报',
                                 '左后方自行车二级警报',
                                 '左后方自行车三级警报']
    alert_level_left_behind = alert_count(frames_left_behind,
                                          MIN_CYCLIST_FRAMES,
                                          ALERT_DISTANCE_CYCLIST)
    # 处理警报，这边只要发出最高级别的警报
    if len(alert_level_left_behind) > 0:
        print(alert_content_left_behind[alert_level_left_behind[0]])
        # 音频警报部分
        if alert_level_left_behind[0] <= ALERT_AUDIO_LEVEL:
            alert_audio_path = cyclist_audios_path[2]
            alert_audio_now.append(alert_audio_path)
            if (not alert_audio_path in alert_audio_before) \
                    and (not cyclist_audios_path[0] in alert_audio_before):
                alert_queue.put(alert_audio_path)

    # (4) 最后检测右后方
    frames_right_behind = frames_cyclist_info[
        np.logical_and(frames_cyclist_info['y'] <= 0,
                       frames_cyclist_info['x'] <= 0)]
    alert_content_right_behind = ['右后方自行车一级警报',
                                  '右后方自行车二级警报',
                                  '右后方自行车三级警报']
    alert_level_right_behind = alert_count(frames_right_behind,
                                           MIN_CYCLIST_FRAMES,
                                           ALERT_DISTANCE_CYCLIST)
    # 处理警报，这边只要发出最高级别的警报
    if len(alert_level_right_behind) > 0:
        print(alert_content_right_behind[alert_level_right_behind[0]])
        # 音频警报部分
        if alert_level_right_behind[0] <= ALERT_AUDIO_LEVEL:
            alert_audio_path = cyclist_audios_path[3]
            alert_audio_now.append(alert_audio_path)
            if (not alert_audio_path in alert_audio_before) \
                    and (not cyclist_audios_path[1] in alert_audio_before):  # 从右前方变成右后方就别报警了
                alert_queue.put(alert_audio_path)


def alert_pedestrian(frames_pedestrian_info):
    '''对pedestrian的边框进行判断，发出警报'''
    global alert_audio_now
    # print('检测周围是否有行人...')
    frames_pedestrian_info = frames_pedestrian_info[frames_pedestrian_info['dist']
                                                    <= ALERT_DISTANCE_PEDESTRIAN[-1]]

    # 分成左前方和右前方分别检测
    # (1) 先是检测左前方
    frames_left_front = frames_pedestrian_info[
        np.logical_and(frames_pedestrian_info['y'] >= 0,
                       frames_pedestrian_info['x'] >= 0)]
    alert_content_left_front = ['左前方行人一级警报',
                                '左前方行人二级警报',
                                '左前方行人三级警报']
    alert_level_left_front = alert_count(frames_left_front,
                                         MIN_PEDESTRIAN_FRAMES,
                                         ALERT_DISTANCE_PEDESTRIAN)
    # 处理警报，这边只要发出最高级别的警报
    if len(alert_level_left_front) > 0:
        print(alert_content_left_front[alert_level_left_front[0]])
        # 音频警报部分
        if alert_level_left_front[0] <= ALERT_AUDIO_LEVEL:
            alert_audio_path = pedestrian_audios_path[0]
            alert_audio_now.append(alert_audio_path)
            if not alert_audio_path in alert_audio_before:
                alert_queue.put(alert_audio_path)

    # (2) 再检测右前方
    frames_right_front = frames_pedestrian_info[
        np.logical_and(frames_pedestrian_info['y'] < 0,
                       frames_pedestrian_info['x'] >= 0)]
    alert_content_right_front = ['右前方行人一级警报',
                                 '右前方行人二级警报',
                                 '右前方行人三级警报']
    alert_level_right_front = alert_count(frames_right_front,
                                          MIN_PEDESTRIAN_FRAMES,
                                          ALERT_DISTANCE_PEDESTRIAN)
    # 处理警报，这边只要发出最高级别的警报
    if len(alert_level_right_front) > 0:
        print(alert_content_right_front[alert_level_right_front[0]])
        if alert_level_right_front[0] <= ALERT_AUDIO_LEVEL:
            alert_audio_path = pedestrian_audios_path[1]
            alert_audio_now.append(alert_audio_path)
            if not alert_audio_path in alert_audio_before:
                alert_queue.put(alert_audio_path)


def Event_play_alert_audio():
    '''播放audio的事件'''
    e = threading.Event()

    def go():
        while (1):
            e.wait()  # 等待触发
            e.clear()
            while (not alert_queue.empty()):  # 把当前队列里面的音频都放出来
                play_alert_audio(alert_queue.get())

    threading.Thread(target=go).start()
    return e


def all_alerts():
    '''对所有项目进行检测并警报的函数'''
    global video_result, timestamp, time_interval, frames_info
    print('*' * 20)
    print('进行检测，timestamp={}'.format(timestamp))

    frames_car_info = frames_info[frames_info['name'] == 'Car']
    alert_car(frames_car_info)

    frames_truck_info = frames_info[frames_info['name'] == 'Truck']
    alert_truck(frames_truck_info)

    frames_tricar_info = frames_info[frames_info['name'] == 'Tricar']
    alert_tricar(frames_tricar_info)

    frames_cyclist_info = frames_info[frames_info['name'] == 'Cyclist']
    alert_cyclist(frames_cyclist_info)

    frames_pedestrian_info = frames_info[frames_info['name'] == 'Pedestrian']
    alert_pedestrian(frames_pedestrian_info)

    event_play_alert_audio.set()

    # 下一次警报
    if timestamp < STOP_TIME:
        # 更新参数
        global alert_audio_before, alert_audio_now
        alert_audio_before = alert_audio_now
        alert_audio_now = []
        timestamp += time_interval
        frames_info = video_result[
            np.logical_and(video_result['image_idx'] >= img_idx_list[FPS * (timestamp - time_interval)],
                           video_result['image_idx'] <= img_idx_list[FPS * timestamp - 1])]

        global alert_timer
        alert_timer = Timer(time_interval - TIME_RECORRECTION, all_alerts, ())
        alert_timer.start()
    else:
        end_run_time = time.time()
        print('检测结束，共耗时{}秒'.format(end_run_time - start_run_time))
        sys.exit(0)


if __name__ == "__main__":
    # 参数设置
    FPS = 10
    STOP_TIME = 200
    # 前一秒至少多少张影像检测到就报警，用来提高容错性
    MIN_CAR_FRAMES = MIN_TRUCK_FRAMES = MIN_TRICAR_FRAMES = MIN_CYCLIST_FRAMES = MIN_PEDESTRIAN_FRAMES = 2
    # 报警距离，可以分等级
    ALERT_DISTANCE_CAR = []
    ALERT_DISTANCE_TRUCK = []
    ALERT_DISTANCE_TRICAR = []
    ALERT_DISTANCE_CYCLIST = [5, 10, 15]
    ALERT_DISTANCE_PEDESTRIAN = [5, 10, 15]
    # 校正时间，扣除程序运行时间
    TIME_RECORRECTION = 6.8672528 / 200
    # TIME_RECORRECTION = 0  # 最后测一下

    ALERT_AUDIO_LEVEL = 2 - 1  # 发出声音报警的LEVEL,LEVEL从0计数

    # 配置alert_audio音频路径
    # 行人报警音频
    pedestrian_audios_path = ['alert_audio/pedestrian{}.wav'.format(i) for i in range(2)]
    # 自行车报警视频
    cyclist_audios_path = ['alert_audio/cyclist{}.wav'.format(i) for i in range(4)]

    alert_queue = Queue(maxsize=10)  # 设置一个播放alert_audio的队列
    event_play_alert_audio = Event_play_alert_audio()

    alert_audio_before = []  # 用来存上一秒的报警，如果当前秒还有该警报，就别报了
    alert_audio_now = []

    # 读取数据
    filename = sys.argv[1]  # filename = 'dbsample3_size3_lr1_batch8_video_result.csv'
    video_result = pds.read_csv(filename)
    img_idx_list = video_result['image_idx'].unique().tolist()

    # 设置时间触发器，定时检测当前行车状态
    time_interval = int(sys.argv[2])
    timestamp = time_interval  # 当前时间戳
    frames_info = video_result[
        np.logical_and(video_result['image_idx'] >= img_idx_list[FPS * (timestamp - time_interval)],
                       video_result['image_idx'] <= img_idx_list[FPS * timestamp - 1])]

    start_run_time = time.time()
    alert_timer = Timer(time_interval, all_alerts, ())
    alert_timer.start()
