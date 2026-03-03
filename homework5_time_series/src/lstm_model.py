"""
LSTM时间序列预测模型
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTMModel:
    """
    LSTM模型用于时间序列预测
    
    支持:
    - 多步预测
    - 滚动预测
    """
    
    def __init__(self, sequence_length: int = 60, 
                 n_features: int = 1,
                 lstm_units: int = 50,
                 dropout: float = 0.2,
                 dense_units: int = 1):
        """
        Args:
            sequence_length: 输入序列长度 (滑动窗口大小)
            n_features: 特征数量
            lstm_units: LSTM隐藏层单元数
            dropout: Dropout比例
            dense_units: 输出层单元数
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        
        self.model = None
        self.scaler = MinMaxScaler()
        
        self._build_model()
    
    def _build_model(self):
        """构建LSTM模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
            
            self.model = Sequential([
                LSTM(self.lstm_units, return_sequences=True,
                     input_shape=(self.sequence_length, self.n_features)),
                Dropout(self.dropout),
                LSTM(self.lstm_units, return_sequences=False),
                Dropout(self.dropout),
                Dense(self.dense_units)
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            
        except ImportError:
            print("TensorFlow未安装")
            self.model = None
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口序列
        
        Args:
            data: 原始时间序列
        
        Returns:
            (X, y) - 输入和目标
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, series: pd.Series, train_ratio: float = 0.8) -> dict:
        """
        准备训练和测试数据
        
        Returns:
            包含训练/测试数据和scaler的字典
        """
        values = series.values.reshape(-1, 1)
        
        scaled_data = self.scaler.fit_transform(values)
        
        X, y = self.create_sequences(scaled_data)
        
        train_size = int(len(X) * train_ratio)
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'raw_test': series.values[train_size + self.sequence_length:]
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.1,
            early_stopping: bool = True) -> 'LSTMModel':
        """
        训练模型
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        callbacks = []
        if early_stopping:
            from tensorflow.keras.callbacks import EarlyStopping
            callbacks.append(EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            ))
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_future(self, last_sequence: np.ndarray, 
                       steps: int) -> np.ndarray:
        """
        预测未来steps个时间步
        
        Args:
            last_sequence: 最后一段已知序列
            steps: 预测步数
        
        Returns:
            未来预测值
        """
        predictions = []
        current_seq = last_sequence.copy()
        
        for _ in range(steps):
            pred = self.model.predict(current_seq.reshape(1, -1, 1))
            predictions.append(pred[0, 0])
            
            # 滚动预测
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred
        
        return np.array(predictions)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反归一化"""
        return self.scaler.inverse_transform(data)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """评估模型性能"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape if not np.isnan(mape) else None
        }


class MultiStepLSTM:
    """多步预测LSTM"""
    
    def __init__(self, sequence_length: int = 60, 
                 forecast_horizon: int = 5,
                 lstm_units: int = 64):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.scaler = MinMaxScaler()
        self.model = None
        
        self._build_model()
    
    def _build_model(self):
        """构建直接多步预测模型"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        self.model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.sequence_length, 1)),
            Dense(self.forecast_horizon)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
    
    def create_multi_step_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建多步预测的训练数据"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        
        return np.array(X), np.array(y)
