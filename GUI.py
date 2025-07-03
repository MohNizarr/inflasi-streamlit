import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Set random seed
SEED = 42
#Pemodelan
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size, theta_size, horizon, n_neurons, n_layers, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        self.hidden = []
        for i in range(n_layers):
            self.hidden.append(tf.keras.layers.Dense(
                n_neurons,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED + i)
            ))
            if dropout_rate > 0:
                self.hidden.append(tf.keras.layers.Dropout(dropout_rate, seed=SEED + i))

        self.theta_layer = tf.keras.layers.Dense(
            theta_size,
            activation="linear",
            name="theta",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED + 100)
        )

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast  # only return forecast (change if your model expects backcast too)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'theta_size': self.theta_size,
            'horizon': self.horizon,
            'n_neurons': self.n_neurons,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Register custom object globally
get_custom_objects()['NBeatsBlock'] = NBeatsBlock

# Streamlit app title
st.title('📊 Aplikasi Prediksi Inflasi Jawa Timur dengan N-BEATS TPE')

# Upload Excel file
st.subheader("📥 Upload File Excel Untuk Dianalisis")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, parse_dates=['date'], index_col='date')

    st.subheader("📝 Tampilan Data")
    st.write(df)

    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

    st.subheader("📉 Visualisasi Plot Time Series Inflasi Jawa Timur")
    plt.figure(figsize=(10, 4))
    plt.plot(df, label='Inflasi')
    plt.title('Inflasi Jawa Timur Tahun 2005-2024')
    plt.xlabel('Tanggal')
    plt.ylabel('Inflasi')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # Splitting and scaling
    split_size = int(len(df) * 0.8)
    train_data = df[:split_size]
    test_data = df[split_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    train_scaled_df = pd.DataFrame(train_scaled, columns=["inflasi"], index=train_data.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=["inflasi"], index=test_data.index)

    # Sliding window function
    def create_sliding_window_df(df_scaled, window_size, col_name='inflasi'):
        X, y, dates = [], [], []
        for start in range(len(df_scaled) - window_size):
            end = start + window_size
            X.append(df_scaled.iloc[start:end][col_name].values)
            y.append(df_scaled.iloc[end][col_name])
            dates.append(df_scaled.index[end])
        X_df = pd.DataFrame(np.array(X), columns=[f"{col_name}_{i+1}" for i in range(window_size)], index=dates)
        y_df = pd.Series(y, index=dates, name="target")
        return X_df, y_df

    # Windowing
    WINDOW_SIZE = 12
    full_scaled_df = pd.concat([train_scaled_df, test_scaled_df])
    X_full, y_full = create_sliding_window_df(full_scaled_df, window_size=WINDOW_SIZE)
    test_start_index = len(train_scaled_df) - WINDOW_SIZE

    X_train = X_full[:test_start_index]
    y_train = y_full[:test_start_index]
    X_test = X_full[test_start_index:]
    y_test = y_full[test_start_index:]

    # Dataset preparation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(32).prefetch(1)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values)).batch(32).prefetch(1)

    # Load model
    st.subheader("🔍 Proses Membaca Model N-BEATS")
    try:
        model = load_model('model_nbeats.h5', custom_objects={'NBeatsBlock': NBeatsBlock})
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    
    
    st.subheader("🔍 Proses Membaca Model N-BEATS TPE")
    try:
        final_model = load_model('final_model_fix2.h5', custom_objects={'NBeatsBlock': NBeatsBlock})
        st.success("✅ Model Berhasil Dimuat")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

    # Predictions
    st.subheader("📊 Model Prediksi")
    train_pred = final_model.predict(train_dataset)
    test_pred = final_model.predict(test_dataset)
    train_pred2 = model.predict(train_dataset)
    test_pred2 = model.predict(test_dataset)

    train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
    test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))
    train_pred2 = scaler.inverse_transform(train_pred2.reshape(-1, 1))
    test_pred2 = scaler.inverse_transform(test_pred2.reshape(-1, 1))
    y_train_true = scaler.inverse_transform(y_train.values.reshape(-1, 1))
    y_test_true = scaler.inverse_transform(y_test.values.reshape(-1, 1))
    # Hitung gabungan All
    # Gabungkan indeks y_train dan y_test
    y_all_index = pd.concat([y_train, y_test]).index

    # Jadikan hasil invers transform sebagai Series dengan index waktu
    y_all_true = pd.Series(np.vstack([y_train_true, y_test_true]).flatten(), index=y_all_index)
    y_all_pred = pd.Series(np.vstack([train_pred, test_pred]).flatten(), index=y_all_index)
    y_all_pred2 = pd.Series(np.vstack([train_pred2, test_pred2]).flatten(), index=y_all_index)

    # Plot
    # Subheader
    st.subheader("📊 Perbandingan Prediksi Model N-BEATS dan N-BEATS TPE")

    # Buat dua kolom berdampingan
    col1, col2 = st.columns(2)

    # Kolom pertama untuk N-BEATS TPE
    with col1:
        st.markdown("**Prediksi Data Test Model N-BEATS**")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(y_test.index, y_test_true, label='Aktual', color='blue')
        ax1.plot(y_test.index, test_pred2, label='Prediksi', color='orange')
        ax1.set_title('N-BEATS')
        ax1.legend()
        st.pyplot(fig1)

    # Kolom kedua untuk N-BEATS
    with col2:
        st.markdown("**Prediksi Data Test Model N-BEATS TPE**")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(y_test.index, y_test_true, label='Aktual', color='blue')
        ax2.plot(y_test.index, test_pred, label='Prediksi', color='red')
        ax2.set_title('N-BEATS TPE')
        ax2.legend()
        st.pyplot(fig2)

    st.subheader("📊 Perbandingan Prediksi Model Pada Seluruh Data")

    # Buat dua kolom berdampingan
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Prediksi Seluruh Data Model N-BEATS**")
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        ax1.plot(y_all_true.index, y_all_true, label='Aktual', color='blue')
        ax1.plot(y_all_pred2.index, y_all_pred2, label='Prediksi', color='red')
        ax1.set_title('N-BEATS')
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.markdown("**Prediksi Seluruh Data Model N-BEATS TPE**")
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.plot(y_all_true.index, y_all_true, label='Aktual', color='blue')
        ax2.plot(y_all_pred.index, y_all_pred, label='Prediksi', color='red')
        ax2.set_title('N-BEATS TPE')
        ax2.legend()
        st.pyplot(fig2)

    st.subheader("📝 Prediksi Data Baru")
    # Future prediction
    last_input = X_test.iloc[-1].values.reshape(1, -1)
    future_predictions = []

    for _ in range(12):
        next_pred = final_model.predict(last_input)[0]
        future_predictions.append(next_pred)
        last_input = np.append(last_input[:, 1:], next_pred).reshape(1, -1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted'])

    st.write(future_df)

    plt.figure(figsize=(10, 4))
    plt.plot(y_all_true.index, y_all_true, label='Aktual', color='blue')
    plt.plot(y_all_pred.index, y_all_pred, label='Prediksi', color='red')
    plt.plot(future_df.index, future_df['Predicted'], label='Prediksi Data Baru', color='purple')
    plt.title('Prediksi Inflasi Jawa Timur dengan N-BEATS TPE')
    plt.legend()
    st.pyplot(plt)
    
    # ========== Subheader Evaluasi ==========
    st.subheader("📝 Evaluasi Model")

    # Evaluasi N-BEATS
    eval_df2 = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'MSE'],
        'Train': [
            mean_absolute_error(y_train_true, train_pred2),
            np.sqrt(mean_squared_error(y_train_true, train_pred2)),
            mean_absolute_percentage_error(y_train_true, train_pred2) * 100,
            mean_squared_error(y_train_true, train_pred2)
        ],
        'Test': [
            mean_absolute_error(y_test_true, test_pred2),
            np.sqrt(mean_squared_error(y_test_true, test_pred2)),
            mean_absolute_percentage_error(y_test_true, test_pred2) * 100,
            mean_squared_error(y_test_true, test_pred2)
        ]
    })
    eval_df2['All'] = [
        mean_absolute_error(y_all_true, y_all_pred2),
        np.sqrt(mean_squared_error(y_all_true, y_all_pred2)),
        mean_absolute_percentage_error(y_all_true, y_all_pred2) * 100,
        mean_squared_error(y_all_true, y_all_pred2)
    ]
    st.write("Evaluasi Model N-BEATS")
    st.write(eval_df2)

    # Evaluasi N-BEATS TPE
    eval_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE', 'MSE'],
        'Train': [
            mean_absolute_error(y_train_true, train_pred),
            np.sqrt(mean_squared_error(y_train_true, train_pred)),
            mean_absolute_percentage_error(y_train_true, train_pred) * 100,
            mean_squared_error(y_train_true, train_pred)
        ],
        'Test': [
            mean_absolute_error(y_test_true, test_pred),
            np.sqrt(mean_squared_error(y_test_true, test_pred)),
            mean_absolute_percentage_error(y_test_true, test_pred) * 100,
            mean_squared_error(y_test_true, test_pred)
        ]
    })
    eval_df['All'] = [
        mean_absolute_error(y_all_true, y_all_pred),
        np.sqrt(mean_squared_error(y_all_true, y_all_pred)),
        mean_absolute_percentage_error(y_all_true, y_all_pred) * 100,
        mean_squared_error(y_all_true, y_all_pred)
    ]
    st.write("Evaluasi Model N-BEATS TPE")
    st.write(eval_df)

    # ========== Plot Perbandingan Evaluasi ==========
    st.subheader("📈 Visualisasi Evaluasi Model")

    metrics = ['MAE', 'RMSE', 'MAPE', 'MSE']
    x = np.arange(len(metrics))
    width = 0.35

    nbeats_train = eval_df2['Train'].values
    nbeats_tpe_train = eval_df['Train'].values
    nbeats_test = eval_df2['Test'].values
    nbeats_tpe_test = eval_df['Test'].values
    nbeats_all = eval_df2['All'].values
    nbeats_tpe_all = eval_df['All'].values

    def plot_metrics_streamlit(nbeats_vals, nbeats_tpe_vals, judul):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, nbeats_vals, width, label='N-BEATS', color='skyblue')
        bars2 = ax.bar(x + width/2, nbeats_tpe_vals, width, label='N-BEATS TPE', color='salmon')

        for i, bar in enumerate(bars1):
            height = bar.get_height()
            label = f'{height:.2f}%' if metrics[i] == 'MAPE' else f'{height:.2f}'
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            label = f'{height:.2f}%' if metrics[i] == 'MAPE' else f'{height:.2f}'
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Nilai Metrik')
        ax.set_title(f'Perbandingan Evaluasi Model ({judul})')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

    # Plot tiga grafik
    plot_metrics_streamlit(nbeats_train, nbeats_tpe_train, 'Train Metrics')
    plot_metrics_streamlit(nbeats_test, nbeats_tpe_test, 'Test Metrics')
    plot_metrics_streamlit(nbeats_all, nbeats_tpe_all, 'Overall Metrics')

