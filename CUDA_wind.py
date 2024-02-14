import json
import random
import urllib3
import numba
import numpy as np
from datetime import datetime
import certifi
import re
import tkinter as tk
from tkinter import ttk


area_dic = {'北海道/釧路':'014100',
            '北海道/旭川':'012000',
            '北海道/札幌':'016000',
            '青森県':'020000',
            '岩手県':'030000',
            '宮城県':'040000',
            '秋田県':'050000',
            '山形県':'060000',
            '福島県':'070000',
            '茨城県':'080000',
            '栃木県':'090000',
            '群馬県':'100000',
            '埼玉県':'110000',
            '千葉県':'120000',
            '東京都':'130000',
            '神奈川県':'140000',
            '新潟県':'150000',
            '富山県':'160000',
            '石川県':'170000',
            '福井県':'180000',
            '山梨県':'190000',
            '長野県':'200000',
            '岐阜県':'210000',
            '静岡県':'220000',
            '愛知県':'230000',
            '三重県':'240000',
            '滋賀県':'250000',
            '京都府':'260000',
            '大阪府':'270000',
            '兵庫県':'280000',
            '奈良県':'290000',
            '和歌山県':'300000',
            '鳥取県':'310000',
            '島根県':'320000',
            '岡山県':'330000',
            '広島県':'340000',
            '山口県':'350000',
            '徳島県':'360000',
            '香川県':'370000',
            '愛媛県':'380000',
            '高知県':'390000',
            '福岡県':'400000',
            '佐賀県':'410000',
            '長崎県':'420000',
            '熊本県':'430000',
            '大分県':'440000',
            '宮崎県':'450000',
            '鹿児島県':'460100',
            '沖縄県/那覇':'471000',
            '沖縄県/石垣':'474000'
            }


class Analysis:
    def __init__(self, selected_area):
        self.selected_area = selected_area

    def run(self):
        selected_area_code = area_dic.get(self.selected_area)
        if selected_area_code is not None:
            jma_data = self.get_jma_data(selected_area_code)
            if jma_data:
                result_text = self.generate_result_text(jma_data)
                return result_text
            else:
                return "気象データの取得に失敗しました。"
        else:
            return "選択した都道府県はサポートされていません。"

    def get_jma_data(self, area_code):
        jma_url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"
        
        # urllib3のHTTPプールマネージャを作成
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            ca_certs=certifi.where()
        )
        
        try:
            # urllib3を使用してデータを取得
            response = http.request('GET', jma_url)
            jma_data = json.loads(response.data.decode('utf-8'))
            return jma_data
        except Exception as e:
            print(f"データの取得中にエラーが発生しました: {e}")
            return 
        
    # 2D配列を生成する関数
    def create_2d_array(self, json_data):
        # 各地域のweatherCodesを取得
        names = []
        weather_codes = []
        date = []
        for entry in json_data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                dates = time_data.get("timeDefines", [])
                date.append(dates)
                areas = time_data.get("areas", [])
                for i, area in enumerate(areas):
                    area_weather_codes = area.get("weatherCodes", [])
                    name = area["area"]["name"]
                    weather_codes.append(area_weather_codes) 
                    names.append(name)                
    
        # 最大の地域数を取得
        max_area_count = max(len(area_codes) for area_codes in weather_codes)
    
        # 2D配列を初期化
        result = np.empty((len(weather_codes), max_area_count), dtype=np.float32)
    
        # weatherCodesを2D配列にセット
        for i, area_codes in enumerate(weather_codes):
            result[i, :len(area_codes)] = area_codes
    
        return result, names, date
    
    def generate_result_text(self, jma_data):
        # 入力値の取得
        selected_area = self.selected_area
    
        result_text = f"気象庁データ: {selected_area}\n"
        result_text += f"今日の天気: {jma_data[0]['timeSeries'][0]['areas'][0]['weathers'][0]}\n"
        result_text += f"今日の風: {jma_data[0]['timeSeries'][0]['areas'][0]['winds'][0]}\n"
    
        weather_matrix, names, dates = self.create_2d_array(jma_data)

        date_order_count = 0

        names = list(set(names))

        for date in dates:
            if date_order_count == 4:
                continue
            date_order_count += 1
            for name in names:
                if re.fullmatch(r'.*島.*', name):
                    continue
                day_count = 0
                if datetime.fromisoformat(date[day_count]).hour == 0 and len(date) == 7:
                    result_text += f"----------------------------------[地域, 一週間ごと]---------------------------------------\n"
                    day_count += 1
                else:
                    result_text += f"----------------------------------[地域, 時間ごと]-----------------------------------------\n"
                    day_count += 1     
                for i in range(len(date)):
                    # ISO 8601形式の日付と時刻を解析してPythonのdatetimeオブジェクトに変換
                    input_datetime = datetime.fromisoformat(date[i])
                    weekly = input_datetime.weekday()
                    # 24時間制のフォーマットで日付と時刻を文字列に変換
                    format_month = str(int(str(input_datetime.month))) + "月"
                    format_hour = str(int(str(input_datetime.hour))) + "時"
                    format_time = str(int(str(input_datetime.minute))) + "分"
                    if weekly == 0:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　月曜日")
                    elif weekly == 1:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　火曜日")
                    elif weekly == 2:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　水曜日")
                    elif weekly == 3:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　木曜日")
                    elif weekly == 4:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　金曜日")
                    elif weekly == 5:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　土曜日")
                    else:
                        formatted_datetime = input_datetime.strftime("%Y年" + format_month + "%d日" + format_hour + format_time + "　日曜日")    
                    result_text += f"日付: {formatted_datetime}\n"
                    result_text += f"地域名: {name}\n"
    
                    # 天気コードからなる2D配列を作成
                    code = weather_matrix
                    results = np.array([[float(code) for code in row] for row in code], dtype=np.float32)
    
                    result = self.cuda_ridge_detection(results, 0.5)  # 調整後の闘値を使用して再度リッジ検出を実行
    
                    # 日時系列ごとにトータルリッジ検出結果と闘値を計算
                    total_ridge, threshold = self.calculate_total_ridge_and_threshold(result , i)
                    
                    # 他の条件に応じて闘値を調整
                    if total_ridge >= 10:
                        threshold += 0.5  # トータルリッジ検出値が10以上の場合、闘値を0.5増加させる
                    
                    # 降水確率が高い場合に闘値を下げる
                    jma_rainfalls = self.get_precipitation_probability(jma_data)

                    # トータルリッジが2以上の場合
                    if total_ridge >= 2:
                        # 降水確率を50%以上に設定
                        jma_rainfalls[i] = min(max(total_ridge * 10, 50), 100)
                    # トータルリッジが0の場合
                    elif total_ridge == 0:
                        # 降水確率を0%に設定
                        jma_rainfalls[i] = 0
                    # トータルリッジが1未満の場合
                    else:
                        # 降水確率をトータルリッジの値に応じて設定
                        jma_rainfalls[i] = min(total_ridge * 10, 100)
                    
    
                    # total_ridge = np.sum(result[0])
        
                    # 降水確率と平均風速を取得
                    average_wind_speed = self.get_winds(jma_data)
        
                    # 天候予測
                    low_temperature, up_temprature = self.calculate_average_temperature(jma_data)
                    average_rainfalls = self.calculate_average_rainfall(jma_rainfalls)
                    snow_predicted, predicted_weather, snow_probability = self.predict_weather(
                        1.0,
                        5.0,
                        10.0,
                        low_temperature,
                        up_temprature,
                        average_rainfalls,
                        total_ridge,
                        jma_rainfalls,
                        average_wind_speed,
                        i
                    )
        
                    # 修正：snow_probabilityを考慮して雪の確率を上げる
                    if snow_probability > 0.1:
                       snow_probability += 0.2  # 雪の確率が一定値以上ならばさらに上げる（適宜調整）
                    elif snow_predicted:
                       snow_probability += 0.2  
        
                    result_text += f"降水確率: {jma_rainfalls[i]}%\n"
                    result_text += f"天気予測：{predicted_weather}\n"
    
        return result_text
    
    def calculate_total_ridge_and_threshold(self, results, index):
    # ここで日時系列ごとにトータルリッジ検出結果と闘値を計算
        total_ridge = np.sum(results[0][index])
        mean_value = np.mean(results[0][index])
        std_deviation = np.std(results[0][index])
        threshold = (mean_value + 2 * std_deviation) / 10 ** 34  # 例: (平均値 + 2倍の標準偏差) / 10の34乗で少数点数にする
        return total_ridge, threshold


    def get_precipitation_probability(self, data):
        pops = []
        for entry in data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                areas = time_data.get("areas", [])
                for area in areas:
                    pops.extend(area.get("pops", []))
        # 空文字列や空の要素を取り除く
        pops = [value for value in pops if value]
        return pops

    def calculate_average_rainfall(self, jma_rainfalls):
        pops = []  # リストとして初期化
        if jma_rainfalls:
            for entry in jma_rainfalls:
                try:
                    # 各要素を数値に変換してリストに追加
                    pops.append(float(entry))
                except (ValueError, TypeError):
                    pops = self.get_precipitation_probability(jma_rainfalls)
        # 空文字列や空の要素を取り除く
        pops = [value for value in pops if value]
        # リストに要素があれば平均を計算
        return np.mean(pops, dtype=np.float32) if pops else None


    def calculate_average_temperature(self, data):
        temperatures = []
        lower_temperatures = []
        upper_temperatures = []
        for entry in data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                areas = time_data.get("areas", [])
                for area in areas:
                    temperatures.extend(area.get("temps", []))
                    lower_temperatures.extend(area.get("tempsMinLower", []))
                    upper_temperatures.extend(area.get("tempsMaxLower", []))
        # 空文字列や空の要素を取り除く
        low_temperatures = [float(value) for value in lower_temperatures if value]
        up_temperatures = [float(value) for value in upper_temperatures if value]
        for i in range(len(temperatures)):
            if i % 2 == 0:
                low_temperatures.insert(0, float(temperatures[i]))
            else:
                up_temperatures.insert(0, float(temperatures[i]))
        return low_temperatures, up_temperatures

    def cuda_ridge_detection(self, data, thres):
        rows, cols = data.shape
        count = np.zeros_like(data, dtype=np.float32)
        for i in numba.prange(1, rows - 1):
            for j in range(1, cols - 1):
                if (
                    i > 0
                    and j > 0
                    and i < (rows - 1)
                    and j < (cols - 1)
                    and data[i, j] > thres
                    and not np.isnan(data[i, j])
                ):
                    step_i = i
                    step_j = j
                    for k in range(1000):
                        if (
                            step_i == 0
                            or step_j == 0
                            or step_i == (rows - 1)
                            or step_j == (cols - 1)
                        ):
                            break
                        index = 4
                        vmax = -np.inf
                        for ii in range(3):
                            for jj in range(3):
                                value = data[step_i + ii - 1, step_j + jj - 1]
                                if value > vmax:
                                    vmax = value
                                    index = jj + 3 * ii
                        if index == 4 or vmax == data[step_i, step_j] or np.isnan(vmax):
                            break
                        row = index // 3
                        col = index % 3
                        count[step_i - 1 + row, step_j - 1 + col] += 1
                        step_i, step_j = step_i - 1 + row, step_j - 1 + col

        # weather_array_normalizedの処理
        weather_normalized = np.mean(data)

        # 平均値が特定の閾値を超えるかどうかの判定
        threshold_exceeded = weather_normalized > 0.5

        # 閾値の超過判定結果を返す
        return count, threshold_exceeded
    
    def get_winds(self, data):
        winds = []
        for entry in data:
            time_series = entry.get("timeSeries", [])
            for time_data in time_series:
                areas = time_data.get("areas", [])
                for area in areas:
                    winds.extend(area.get("waves", []))
        # 空文字列や空の要素を取り除く
        winds = [value for value in winds if value]
        try:
           if winds:
                   wind_speeds = []
                   for wind in winds:
                       if wind is not None:
                           # 風速の文字列から数字と単位（メートル）を取り除いて浮動小数点数に変換
                           for wind in winds:
                               wind_values = re.findall(r'[\d.]+', wind)  # 風速値を正規表現で抽出
                               for wind_speed in wind_values:
                                   wind_speeds.append(float(wind_speed))  # 風速を浮動小数点数に変換してリストに追加
                   if wind_speeds:
                       average_wind_speed = min(wind_speeds)
                       return average_wind_speed
        except Exception as e:
            print(f"風速情報の抽出中にエラーが発生しました: {e}")
        return None
            

    def predict_weather(self, low_temperature_threshold, up_temperature_threshold, precipitation_threshold, low_average_temperature, up_average_temperature, average_rainfall, total_ridge, jma_rainfalls, winds, i):

        # 降水確率を取得
        precipitation_probability = float(jma_rainfalls[i])

        snow_probability = precipitation_probability

        # 平均気温の閾値を調整
        if low_average_temperature[i] <= -2.0:
            snow_probability += 0.3  # 平均気温が-2.0度以下の場合、雪の確率を増加
    
        # 降水量が影響する条件を調整
        if average_rainfall >= 5.0:
            snow_probability += 0.2  # 平均降水量が5.0mm以上の場合、雪の確率を増加
    
        # 雪の予測ロジックを調整
        if (
            low_average_temperature[i] <= low_temperature_threshold
            and up_average_temperature[i] <= up_temperature_threshold
            and float(precipitation_probability) >= precipitation_threshold
            and 10 <= snow_probability <= 30
        ):
            snow_predicted = True
        else:
            snow_predicted = False
            snow_probability = 0.0    
    
        # 天気予測のロジック
        if snow_predicted:
            predicted_weather = "雪"
        else:
            if total_ridge == 0 or precipitation_probability <= 20:
                predicted_weather = "晴れ"
                snow_probability = 0.0
            elif total_ridge == 1 or 20 <= precipitation_probability <= 40:
                predicted_weather = "曇り"
                snow_probability = 0.0
            elif total_ridge >= 2 or precipitation_probability >= 40:
                predicted_weather = "雨"
                snow_probability = 0.0
    
        # 修正：snow_probabilityも返す
        return snow_predicted, predicted_weather, snow_probability


           
def main():
    selected_area = input("都道府県を入力してください：")
    analysis = Analysis(selected_area)
    analysis.run()


class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("天気予報アプリ")
        
        # 都道府県選択用のラベルとコンボボックスを作成
        self.label = ttk.Label(root, text="都道府県を選択してください:")
        self.label.pack(pady=10)
        
        self.selected_area = tk.StringVar()
        self.combobox = ttk.Combobox(root, textvariable=self.selected_area)
        self.combobox['values'] = list(area_dic.keys())
        self.combobox.pack(pady=10)
        
        # 実行ボタンを作成
        self.button = ttk.Button(root, text="実行", command=self.run_analysis)
        self.button.pack(pady=10)
        
        # 結果表示用のテキストボックスを作成
        self.result_textbox = tk.Text(root, height=40, width=100)
        self.result_textbox.pack(pady=10)

    def run_analysis(self):
        selected_area = self.selected_area.get()
        analysis = Analysis(selected_area)
        result_text = analysis.run()
        
        # テキストボックスにコンソールの内容を表示
        self.result_textbox.delete(1.0, tk.END)  # 既存の内容をクリア
        self.result_textbox.insert(tk.END, result_text)

def main():
    root = tk.Tk()
    WeatherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

