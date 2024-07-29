import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from finta import TA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Streamlit arayüzü
st.title('Borsa Analiz Aracı')

# Kullanıcıdan hisse senedi sembolünü alma
symbol_input = st.text_input("Hisse Senedi Sembolünü Girin:", "TUPRS")
symbol = symbol_input + ".IS"

# YFinance kullanarak veri çekme
data = yf.download(symbol, start='2022-01-01')

# Veriyi işleme
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Teknik analiz göstergeleri ekleme
data['SMA'] = TA.SMA(data, 30)
data['EMA'] = TA.EMA(data, 30)
data['RSI'] = TA.RSI(data)
macd = TA.MACD(data)
data['MACD'] = macd['MACD']
data['MACDSignal'] = macd['SIGNAL']

# Gelecekteki 30 günlük fiyatları tahmin etmek için özellikler oluşturma
data['Prediction'] = data['Close'].shift(-30)

# Eksik değerleri içeren satırları düşürme
data = data.dropna()

# Özellikleri ve hedef değişkeni belirleme
X = data[['Close', 'SMA', 'EMA', 'RSI', 'MACD', 'Volume']].values
y = data['Prediction'].values

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
X_future = data[['Close', 'SMA', 'EMA', 'RSI', 'MACD', 'Volume']].values[-30:]
future_predictions = model.predict(X_future)

# 24 saat, 48 saat ve 72 saat sonrası için kapanış fiyatı tahminleri
close_24h = future_predictions[-1]  # 24 saat sonra tahmin
close_48h = future_predictions[-2]  # 48 saat sonra tahmin
close_72h = future_predictions[-3]  # 72 saat sonra tahmin

# İlk iki grafiği yan yana yerleştirme
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Fiyat Grafiği
ax1.plot(data['Close'], label='Kapanış Fiyatı')
ax1.plot(data['SMA'], label='30-gün SMA')
ax1.plot(data['EMA'], label='30-Gün EMA')
ax1.set_title(f'{symbol_input} Fiyat Grafiği')
ax1.set_xlabel('Tarih')
ax1.set_ylabel('Fiyat')
ax1.legend()

# RSI Grafiği
ax2.plot(data['RSI'], label='RSI')
ax2.axhline(70, color='red', linestyle='--')
ax2.axhline(30, color='green', linestyle='--')
ax2.set_title(f'{symbol_input} RSI')
ax2.set_xlabel('Tarih')
ax2.set_ylabel('RSI Değeri')
ax2.legend()

st.pyplot(fig)

# Alt alta iki grafiği yan yana yerleştirme
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 7))

# Fiyat Tahmini Grafiği
ax3.plot(data.index, data['Close'], label='Close Price')
ax3.plot(data.index[-30:], future_predictions, label='Future Predictions', linestyle='--')
ax3.set_title(f'{symbol_input} Fiyat Tahmini')
ax3.set_xlabel('Tarih')
ax3.set_ylabel('Fiyat')
ax3.legend()

# MACD Grafiği
ax4.plot(data.index, data['MACD'], label='MACD')
ax4.plot(data.index, data['MACDSignal'], label='MACD Signal')
ax4.set_title(f'{symbol_input} MACD')
ax4.set_xlabel('Tarih')
ax4.set_ylabel('Değer')
ax4.legend()

st.pyplot(fig)

st.write('Son 30 gün RSI Değeri:', data['RSI'].iloc[-1])

# Tahmin edilen kapanış fiyatları
st.write(f'24 saat sonra tahmin edilen kapanış fiyatı: {close_24h}')
st.write(f'48 saat sonra tahmin edilen kapanış fiyatı: {close_48h}')
st.write(f'72 saat sonra tahmin edilen kapanış fiyatı: {close_72h}')
