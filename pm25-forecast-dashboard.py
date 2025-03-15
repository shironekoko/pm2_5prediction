import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# โหลดโมเดลที่เทรนไว้แล้ว
def load_model():
    try:
        with open('final_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None

# โหลดข้อมูลล่าสุด
def load_latest_data():
    try:
        data = pd.read_excel('cleaned_data.xlsx')
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        return data
    except Exception as e:
        print(f"ไม่สามารถโหลดข้อมูลได้: {e}")
        # สร้างข้อมูลตัวอย่างถ้าไม่สามารถโหลดได้
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='h')  # แก้ไข 'H' เป็น 'h'
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'pm_2_5': np.random.normal(50, 15, len(dates)),
            'humidity': np.random.normal(70, 10, len(dates)),
            'temperature': np.random.normal(30, 5, len(dates))
        })

        # เพิ่มคอลัมน์ที่จำเป็น
        sample_data['day'] = sample_data['timestamp'].dt.day
        sample_data['hour'] = sample_data['timestamp'].dt.hour
        
        # คำนวณ rolling_rate_3 ล่วงหน้า
        sample_data['pm_2_5_roll_rate_3'] = sample_data['pm_2_5'].rolling(window=3*24, min_periods=1).mean()
        
        return sample_data

# ทำการพยากรณ์ตามวิธีที่คุณระบุ
def make_forecast(model, dataset):
    from pycaret.regression import setup, predict_model
    
    # Ensure the dataset has the necessary columns for setup
    dataset = dataset.copy()
    dataset['day'] = dataset['timestamp'].dt.day
    dataset['hour'] = dataset['timestamp'].dt.hour
    if 'pm_2_5_roll_rate_3' not in dataset.columns:
        dataset['pm_2_5_roll_rate_3'] = dataset['pm_2_5'].rolling(window=3*24, min_periods=1).mean()
    
    # Run setup() with the dataset and target column
    setup(data=dataset, target='pm_2_5', silent=True, session_id=123)
    
    # สร้าง future_df สำหรับการทำนาย 7 วันข้างหน้า
    last_timestamp = dataset['timestamp'].max()
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=7, freq='D')

    # ใช้ค่าเฉลี่ยของ 30 วันล่าสุดสำหรับ humidity และ temperature
    recent_data = dataset.sort_values('timestamp').tail(30 * 24)  # 30 วันล่าสุด
    avg_humidity = recent_data['humidity'].mean()
    avg_temperature = recent_data['temperature'].mean()

    # ตรวจสอบว่ามีคอลัมน์ pm_2_5 หรือไม่
    if 'pm_2_5' not in dataset.columns:
        raise KeyError("คอลัมน์ 'pm_2_5' ไม่มีอยู่ใน dataset")

    # คำนวณ rolling_rate_3 ล่วงหน้า
    dataset['pm_2_5_roll_rate_3'] = dataset['pm_2_5'].rolling(window=3*24, min_periods=1).mean()
    rolling_rate_3 = dataset['pm_2_5_roll_rate_3']

    # ทำนายทีละวัน
    predictions = []
    current_data = dataset.copy()

    for i, date in enumerate(future_dates):
        new_day = pd.DataFrame({
            'timestamp': [date],
            'humidity': [avg_humidity],
            'temperature': [avg_temperature],
            'day': [date.day],
            'hour': [0],
        })

        # สร้าง lag features
        for lag in range(1, 8):
            if i >= lag:
                new_day[f'pm_2_5_lag_{lag}'] = [predictions[i - lag]]
            else:
                idx = -(lag - i) * 24
                if abs(idx) > len(current_data):
                    new_day[f'pm_2_5_lag_{lag}'] = [current_data['pm_2_5'].iloc[0]]  # ใช้ค่าตัวแรกของชุดข้อมูล
                else:
                    new_day[f'pm_2_5_lag_{lag}'] = [current_data['pm_2_5'].iloc[idx]]

        # ใช้ค่า rolling_rate_3
        if i == 0:
            new_day['pm_2_5_roll_rate_3'] = [rolling_rate_3.iloc[-1]]
        else:
            new_day['pm_2_5_roll_rate_3'] = [sum(predictions[-3:]) / len(predictions[-3:]) if len(predictions) >= 3 else predictions[-1]]

        try:
            # ทำนายค่า pm_2_5
            prediction = predict_model(model, data=new_day)
            predicted_value = prediction.iloc[0]['Label']  # ใช้ชื่อคอลัมน์ที่ถูกต้อง
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
            predicted_value = current_data['pm_2_5'].tail(24).mean()  # ใช้ค่าเฉลี่ยล่าสุดถ้าเกิดข้อผิดพลาด
            
        predictions.append(predicted_value)

        # เพิ่มข้อมูลที่ทำนายแล้วลงใน current_data
        new_row = new_day.copy()
        new_row['pm_2_5'] = predicted_value
        current_data = pd.concat([current_data, new_row], ignore_index=True)

    # สร้าง DataFrame สำหรับผลลัพธ์
    result_df = pd.DataFrame({
        'timestamp': future_dates,
        'predicted_pm_2_5': predictions,
        'humidity': [avg_humidity] * 7,
        'temperature': [avg_temperature] * 7
    })
    
    return result_df

# กำหนดช่วงคุณภาพอากาศตามเกณฑ์ AQI ของไทย
def get_aqi_category(pm25):
    if pm25 < 25:
        return "ดีมาก", "#a8e05f"
    elif pm25 < 37:
        return "ดี", "#87c13c"
    elif pm25 < 50:
        return "ปานกลาง", "#f8cd38"
    elif pm25 < 90:
        return "เริ่มมีผลกระทบต่อสุขภาพ", "#f89d3e"
    elif pm25 < 180:
        return "มีผลกระทบต่อสุขภาพ", "#e93f33"
    else:
        return "อันตราย", "#af2e24"

# สร้าง Dash app
app = dash.Dash(__name__, meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
])
server = app.server

# โหลดโมเดลและข้อมูล
model = load_model()
latest_data = load_latest_data()
forecast_results = make_forecast(model, latest_data)

# คำนวณค่าสถิติสำคัญ
current_pm25 = latest_data['pm_2_5'].iloc[-1]
avg_pm25_7days = latest_data['pm_2_5'].tail(7*24).mean()
max_forecast = forecast_results['predicted_pm_2_5'].max()
min_forecast = forecast_results['predicted_pm_2_5'].min()
avg_forecast = forecast_results['predicted_pm_2_5'].mean()

current_category, current_color = get_aqi_category(current_pm25)
forecast_category, forecast_color = get_aqi_category(avg_forecast)

# แปลงข้อมูลประวัติเป็นรายวัน
historical_daily = latest_data.copy()
historical_daily['date'] = latest_data['timestamp'].dt.date
historical_daily = historical_daily.groupby('date')['pm_2_5'].mean().reset_index()
historical_daily['date'] = pd.to_datetime(historical_daily['date'])
historical_daily = historical_daily.tail(30)  # แสดง 30 วันล่าสุด

# CSS สำหรับ Dashboard
external_stylesheets = []

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src='/assets/logo.png', style={'height': '60px'}, className='logo'),
        ], className='logo-container'),
        html.H1("ระบบพยากรณ์ค่า PM2.5 ล่วงหน้า 7 วัน", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '20px', 'marginBottom': '20px'})
    ], className='header'),
    
    html.Div([
        html.Div([
            html.Div([
                html.H4("ค่า PM2.5 ล่าสุด"),
                html.H2(f"{current_pm25:.1f} µg/m³", style={'color': current_color, 'fontWeight': 'bold'}),
                html.P(f"คุณภาพอากาศ: {current_category}", style={'color': current_color})
            ], className="stat-card"),
            html.Div([
                html.H4("พยากรณ์ PM2.5 เฉลี่ย 7 วันข้างหน้า"),
                html.H2(f"{avg_forecast:.1f} µg/m³", style={'color': forecast_color, 'fontWeight': 'bold'}),
                html.P(f"คุณภาพอากาศ: {forecast_category}", style={'color': forecast_color})
            ], className="stat-card"),
            html.Div([
                html.H4("ค่าพยากรณ์สูงสุด"),
                html.H2(f"{max_forecast:.1f} µg/m³"),
                html.P(f"วันที่: {forecast_results.loc[forecast_results['predicted_pm_2_5'].idxmax(), 'timestamp'].strftime('%d/%m/%Y')}")
            ], className="stat-card"),
            html.Div([
                html.H4("ค่าพยากรณ์ต่ำสุด"),
                html.H2(f"{min_forecast:.1f} µg/m³"),
                html.P(f"วันที่: {forecast_results.loc[forecast_results['predicted_pm_2_5'].idxmin(), 'timestamp'].strftime('%d/%m/%Y')}")
            ], className="stat-card"),
        ], className="stats-container"),
        
        html.Div([
            html.H3("การพยากรณ์ PM2.5 สำหรับ 7 วันข้างหน้า", style={'textAlign': 'center', 'marginTop': '20px'}),
            dcc.Graph(
                id='forecast-chart',
                figure={
                    'data': [
                        go.Bar(
                            x=forecast_results['timestamp'].dt.strftime('%d/%m/%Y'),
                            y=forecast_results['predicted_pm_2_5'],
                            marker={'color': [get_aqi_category(val)[1] for val in forecast_results['predicted_pm_2_5']]},
                            hovertemplate='วันที่: %{x}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
                        )
                    ],
                    'layout': go.Layout(
                        height=400,
                        margin={'l': 40, 'r': 20, 't': 20, 'b': 30},
                        xaxis={'title': 'วันที่'},
                        yaxis={'title': 'PM2.5 (µg/m³)'},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.03)'
                    )
                }
            )
        ], className="chart-container"),
        
        html.Div([
            html.H3("ประวัติค่า PM2.5 ย้อนหลัง 30 วัน", style={'textAlign': 'center', 'marginTop': '20px'}),
            dcc.Graph(
                id='history-chart',
                figure={
                    'data': [
                        go.Scatter(
                            x=historical_daily['date'],
                            y=historical_daily['pm_2_5'],
                            mode='lines+markers',
                            marker={'color': 'rgba(75, 192, 192, 0.8)'},
                            line={'width': 2, 'color': 'rgba(75, 192, 192, 0.8)'},
                            hovertemplate='วันที่: %{x|%d/%m/%Y}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
                        )
                    ],
                    'layout': go.Layout(
                        height=400,
                        margin={'l': 40, 'r': 20, 't': 20, 'b': 30},
                        xaxis={'title': 'วันที่'},
                        yaxis={'title': 'PM2.5 (µg/m³)'},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.03)'
                    )
                }
            )
        ], className="chart-container"),
        
        html.Div([
            html.H3("การเปรียบเทียบค่า PM2.5 จริงกับค่าพยากรณ์", style={'textAlign': 'center', 'marginTop': '20px'}),
            dcc.Graph(
                id='comparison-chart',
                figure={
                    'data': [
                        go.Scatter(
                            x=pd.concat([historical_daily['date'].tail(7), pd.Series(forecast_results['timestamp'].dt.date)]),
                            y=pd.concat([historical_daily['pm_2_5'].tail(7), pd.Series([None]*len(forecast_results))]),
                            mode='lines+markers',
                            name='ค่าจริง',
                            marker={'color': 'rgba(54, 162, 235, 0.8)'},
                            line={'width': 2, 'dash': 'dash'}
                        ),
                        go.Scatter(
                            x=forecast_results['timestamp'].dt.date,
                            y=forecast_results['predicted_pm_2_5'],
                            mode='lines+markers',
                            name='ค่าพยากรณ์',
                            marker={'color': 'rgba(255, 99, 132, 0.8)'},
                            line={'width': 2}
                        )
                    ],
                    'layout': go.Layout(
                        height=400,
                        margin={'l': 40, 'r': 20, 't': 20, 'b': 30},
                        xaxis={'title': 'วันที่'},
                        yaxis={'title': 'PM2.5 (µg/m³)'},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.03)',
                        legend={'orientation': 'h', 'y': 1.1}
                    )
                }
            )
        ], className="chart-container"),
    ], className="content-container"),
    
    html.Footer([
        html.P("© 2025 ระบบพยากรณ์คุณภาพอากาศ"),
        html.P("อัพเดทล่าสุด: " + datetime.now().strftime("%d/%m/%Y %H:%M"))
    ])
])