import pandas as pd

# โหลดข้อมูล
file_path = 'export-pm25_eng-1d.xlsx'
dataset = pd.read_excel(file_path, engine="openpyxl")

# ลบแถวที่มีค่า NaN
dataset = dataset.dropna()

# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
numeric_cols = dataset.select_dtypes(include=['number']).columns

# คำนวณ IQR สำหรับทุกคอลัมน์
Q1 = dataset[numeric_cols].quantile(0.25)
Q3 = dataset[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# คำนวณขอบเขตของค่า Outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# กำจัดค่า Outlier
mask = ~((dataset[numeric_cols] < lower_bound) | (dataset[numeric_cols] > upper_bound)).any(axis=1)
cleaned_dataset = dataset.loc[mask]

# บันทึกข้อมูลที่ถูกล้างแล้วเป็นไฟล์ใหม่
output_path = 'eng_cleaned_data.xlsx'
cleaned_dataset.to_excel(output_path, index=False)

# รายงานผล
num_removed = len(dataset) - len(cleaned_dataset)
print(f"ข้อมูลหลังทำความสะอาดเหลือ {len(cleaned_dataset)} แถว จากเดิม {len(dataset)} แถว (ลบไป {num_removed} แถว)")


import pandas as pd

# โหลดข้อมูล
file_path = 'export-jsps012-1d.xlsx'
dataset = pd.read_excel(file_path, engine="openpyxl")

# ลบแถวที่มีค่า NaN
dataset = dataset.dropna()

# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
numeric_cols = dataset.select_dtypes(include=['number']).columns

# คำนวณ IQR สำหรับทุกคอลัมน์
Q1 = dataset[numeric_cols].quantile(0.25)
Q3 = dataset[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# คำนวณขอบเขตของค่า Outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# กำจัดค่า Outlier
mask = ~((dataset[numeric_cols] < lower_bound) | (dataset[numeric_cols] > upper_bound)).any(axis=1)
cleaned_dataset = dataset.loc[mask]

# บันทึกข้อมูลที่ถูกล้างแล้วเป็นไฟล์ใหม่
output_path = 'provinceHatyai_cleaned_data.xlsx'
cleaned_dataset.to_excel(output_path, index=False)

# รายงานผล
num_removed = len(dataset) - len(cleaned_dataset)
print(f"ข้อมูลหลังทำความสะอาดเหลือ {len(cleaned_dataset)} แถว จากเดิม {len(dataset)} แถว (ลบไป {num_removed} แถว)")
