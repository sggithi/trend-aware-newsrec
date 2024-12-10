import pandas as pd
import re

# 데이터를 라인 단위로 직접 읽어 처리
data = []
with open("behaviors.tsv", "r") as file:
    for line in file:
        match = re.match(r"(\S+)\s+(\S+)\t(.+)", line.strip())
        if match:
            index = match.group(1)
            user_id = match.group(2)
            timestamp_and_items = match.group(3)

            timestamp_match = re.match(r"(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} (?:AM|PM))\s+(.+)", timestamp_and_items)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                items = timestamp_match.group(2)
            else:
                timestamp = None
                items = timestamp_and_items

            data.append([index, user_id, timestamp, items])

# 리스트를 데이터프레임으로 변환
df = pd.DataFrame(data, columns=["Index", "UserID", "timestamp", "items"])

# timestamp를 datetime 형식으로 변환
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# 1시간 단위로 그룹화하기 위해 새로운 열 'time_group' 추가
df['time_group'] = df['timestamp'].dt.floor('H')  # 시간대별로 그룹화 (1시간 단위)

# time_group과 UserID를 기준으로 정렬
df_sorted = df.sort_values(by=["time_group", "UserID"])

# 정렬된 데이터프레임을 TSV 파일로 저장
df_sorted.to_csv("sorted_output_by_1H.tsv", sep='\t', index=False)

# 그룹별로 묶어서 확인
grouped = df_sorted.groupby('time_group')

