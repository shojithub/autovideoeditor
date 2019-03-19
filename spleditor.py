import datetime
"""
spleditorはsplatoon2の動画編集をサポートするプログラムです。
Opencvの画像認識を利用し、ゲームの下記の場面を検出します。
プレイ区間
キル
やられた
Win-Lose
"""

import subprocess
import cv2
import numpy

# opencvのパラメータ
CV_CAP_PROP_POS_FRAMES = 1   # 現在のフレーム数
CV_CAP_PROP_FPS = 5          # フレームレート
CV_CAP_PROP_FRAME_COUNT = 7  # 最大のフレーム数

TEMP_IMAGE_KILL = "kill_image.png"
TEMP_IMAGE_DEATH = "death_image.png"
TEMP_IMAGE_WIN = "win_image.png"

# テスト用の変数を用意
TESTV0 = "../Segment_0001.mp4"


def test_timer(loop=1):
    start = datetime.datetime.now()
    for i in range(loop):
        a = check_frames_of_video(TESTV0)
        for i in a:
            ffmpeg_segment(TESTV0, i[0] - 1, i[1] + 1)

    end = datetime.datetime.now()
    print("start:", start.strftime("%Y%m%d%H%M%S"), ", end:",
          end.strftime("%Y%m%d%H%M%S"), ":", end-start)


# 対象のフォルダから、指定の名前のファイルのpathを調べる

def search_file(search_folder, file_name):
    data = subprocess.run(["find", search_folder, "-name", file_name],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = data.stdout.decode("utf8")
    data = data.split('\n')
    data.remove("")  # 空白行の削除
    return data

#  画面の黒さを調べる


def check_frames_of_video(video, check_interval_seconds=1):
    cap = cv2.VideoCapture(video)
    max_flame = int(cap.get(CV_CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(CV_CAP_PROP_FPS)

    flame_count = max_flame - 1
    black_point_sec = []
    kill_point_sec = []
    death_point_sec = []
    win_point_sec = []

    #  画像ごとに各チェックを行う
    while (flame_count >= 0):
        cap.set(CV_CAP_PROP_POS_FRAMES, flame_count)  # フレーム数指定
        frame = cap.read()[1]                         # フレームを画像として読み込む、リストの2番目

        # 黒い画面化かチェックする
        check = _black_point_check(frame, flame_count, frame_rate)
        if check is not False:
            black_point_sec.append(check)

        check_kill = _image_match_check(frame, TEMP_IMAGE_KILL, flame_count,
                                        frame_rate, height=[660, 695],
                                        width=[500, 785])

        # kill画面かチェックする
        if check_kill is not False:
            kill_point_sec.append(check_kill)

        check_death = _image_match_check(frame, TEMP_IMAGE_DEATH, flame_count,
                                         frame_rate, height=[640, 680],
                                         width=[1000, 1280])

        # death画面かチェックする
        if check_death is not False:
            death_point_sec.append(check_death)

        check_win = _image_match_check(frame, TEMP_IMAGE_WIN, flame_count,
                                       frame_rate, height=[0, 100],
                                       width=[0, 190])

        # 勝利画面かチェックする
        if check_win is not False:
            win_point_sec.append(check_win)

        flame_count = flame_count - \
            (check_interval_seconds * frame_rate)  # フレーム数を移動

    black_point_sec = _game_cut_point(black_point_sec,
                                      check_interval_seconds)   # 重複を削除

    kill_point_sec = _cut_point(kill_point_sec,
                                check_interval_seconds)   # 重複を削除

    death_point_sec = _cut_point(death_point_sec,
                                 check_interval_seconds)   # 重複を削除

    win_point_sec = _cut_point(win_point_sec,
                               check_interval_seconds)   # 重複を削除

    return [black_point_sec, kill_point_sec, death_point_sec, win_point_sec]


def _black_point_check(frame, flame_count, frame_rate):
    point = numpy.sum(frame)
    if point < 1000000:         # 0に近いほど黒い
        flame_seconds = int(flame_count // frame_rate)
        return flame_seconds
    else:
        return False


def _image_match_check(frame, image, flame_count, frame_rate,
                       height=[0, 720], width=[0, 1280]):
    videoframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    videoframe = videoframe[height[0]:height[1], width[0]:width[1]]
    temp = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    point = cv2.matchTemplate(videoframe, temp, cv2.TM_CCOEFF_NORMED)
    point = cv2.minMaxLoc(point)[1]
    if point > 0.8:         # 0に近いほど黒い
        flame_seconds = int(flame_count // frame_rate)
        return flame_seconds
    else:
        return False


def match(img, temp):
    # 比較方法はcv2.TM_CCOEFF_NORMEDを選択
    result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    # 結果のmax_valが欲しい　0-1 1に近いほど似てる
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return (max_val)


def _cut_point(input_list, CHECK_INTERVAL, time_range=180):
    input_list.sort()
    # 連続している箇所をまとめる
    result = []
    for i in range(len(input_list)):
        try:
            if (input_list[i] - input_list[i-1]) != CHECK_INTERVAL:
                result.append(input_list[i])

        except(IndexError):
            result.append(input_list[i])

    return result


# 黒い画像ポイントを整理し、動画から抽出する秒数の取得
def _game_cut_point(input_list, CHECK_INTERVAL, time_range=180):
    input_list.sort()
    # 連続している箇所をまとめる
    count = 0
    dictionary = {}
    for i in range(len(input_list)):
        try:
            if count == 0:
                black_start = input_list[i]

            count = count + 1

            if (input_list[i+1] - input_list[i]) != CHECK_INTERVAL:
                dictionary[black_start] = count
                count = 0

        except(IndexError):
            dictionary[black_start] = count

    # 動画から抽出する箇所を計算する
    result = []
    keys = list(dictionary)
    start = keys[0] + dictionary[keys[0]]
    for i in keys[1:]:
        end = i
        if end - start > time_range:
            result.append([start, end])

        start = i + dictionary[i]

    return result


# ffmpegを呼び出し指定した秒数の動画を切り出す。output_fileは開始秒数_終了時間秒数が追記される。
def ffmpeg_segment(video, start=0, end=10):
    time_range = end - start
    output_video = video.replace(".mp4",
                                 "_" + str(start) + "_" + str(end) + ".mp4")
    command = ["ffmpeg", "-y", "-ss", str(start), "-i", video,
               "-t", str(time_range), "-vcodec", "copy", "-acodec", "copy",
               output_video]
    data = subprocess.Popen(command)
    return data


a = check_frames_of_video(TESTV0)
print(a)
#a = [[[42, 389], [613, 969]], [81, 163, 206, 241, 258, 287, 345, 463, 477, 541, 646, 673, 711, 728, 801], [105, 182, 208, 243, 269, 294, 313, 336, 467, 503, 521, 545, 655, 687, 773, 839, 891], [0, 364]]
