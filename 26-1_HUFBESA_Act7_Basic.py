# 1. 도구 불러오기
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# 2. 데이터 다운받기 (학회원 직접 입력)
### 빈칸을 채워주세요! ###
csv_file_path = os.path.join(path, "mitbih_train.csv")
df = pd.read_csv(csv_file_path, header=None)

# 출력 메시지도 영문으로 통일하여 전문성을 높입니다.
print(f"Data Load Success! Patients: {df.shape[0]}, Features: {df.shape[1]}")

# 3. 정상 vs 비정상 신호 분리 및 파형 비교
normal_ecg = df[df[187] == 0.0].iloc[0, :-1].values
abnormal_ecg = df[df[187] == 1.0].iloc[0, :-1].values

plt.figure(figsize=(12, 5))
plt.plot(normal_ecg, label='Normal', color='blue')
plt.plot(abnormal_ecg, label='Abnormal', color='red', linestyle='--')
plt.title("Normal vs Abnormal ECG Signal Comparison")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# =============================================
# [증강 1] 클래스(진단 종류) 분포 확인
# =============================================
# 마지막 열(187번)에는 5가지 진단 클래스가 담겨 있습니다.
label_col = 187
label_names = {
    0.0: 'Normal',
    1.0: 'Myocardial Infarction',
    2.0: 'ST Change',
    3.0: 'Abnormal Beat',
    4.0: 'Other Abnormal'
}

# map()은 숫자 클래스를 우리가 읽기 쉬운 이름으로 바꿔줍니다.
df['label_name'] = df[label_col].map(label_names)
print("\n[Patient Count by Class]")
print(df['label_name'].value_counts())

plt.figure(figsize=(9, 4)) # 영어 텍스트가 길어질 수 있어 가로 길이를 7에서 9로 살짝 늘렸습니다.
sns.countplot(x='label_name', data=df,
              order=['Normal', 'Myocardial Infarction', 'ST Change', 'Abnormal Beat', 'Other Abnormal'],
              palette='Set2')
plt.title("ECG Data Distribution by Diagnosis Class")
plt.xlabel("Diagnosis Class")
plt.ylabel("Patient Count")
plt.show()

# [분석 포인트] 정상 환자 데이터가 압도적으로 많습니다.
# 이처럼 클래스 간 데이터 수가 크게 차이 나는 것을 '클래스 불균형(Class Imbalance)'이라 합니다.

# =============================================
# [증강 2] 클래스별 평균 파형 비교
# =============================================
# 같은 클래스 환자 100명의 파형을 평균내면 더 안정적인 '대표 파형'을 볼 수 있습니다.
plt.figure(figsize=(12, 6))

colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, (label_val, label_name) in enumerate(label_names.items()):
    # 해당 클래스 환자들만 골라서 마지막 열(정답 열) 제외하고 가져옵니다.
    subset = df[df[label_col] == label_val].iloc[:100, :-2].values
    # axis=0: 같은 시점(열)끼리 평균을 냅니다.
    mean_signal = subset.mean(axis=0)
    plt.plot(mean_signal, label=label_name, color=colors[i], alpha=0.8)

plt.title("Average ECG Waveform Comparison by Class")
plt.xlabel("Time (ms)")
plt.ylabel("Average Amplitude")
plt.legend()
plt.show()

# [분석 포인트] 정상 파형과 비정상 파형의 피크(R파) 높이와 주기가 어떻게 다른지 확인하세요.

# =============================================
# [증강 3] 신호의 기초 통계 특성 추출
# =============================================
print("\n[Signal Basic Statistics by Class (Based on Mean Waveform)]")
print(f"{'Class':<25} {'Mean':>8} {'Std Dev':>10} {'Max':>10} {'Min':>10}")
print("-" * 67)

for label_val, label_name in label_names.items():
    subset = df[df[label_col] == label_val].iloc[:100, :-2].values
    mean_signal = subset.mean(axis=0)
    # 영어 클래스명이 길어서 왼쪽 정렬 간격을 12에서 25로 늘렸습니다.
    print(f"{label_name:<25} "
          f"{mean_signal.mean():>8.4f} "
          f"{mean_signal.std():>10.4f} "
          f"{mean_signal.max():>10.4f} "
          f"{mean_signal.min():>10.4f}")

# [분석 포인트] 표준편차가 클수록 신호의 변동이 심하다는 의미입니다.