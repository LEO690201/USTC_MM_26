"""
ARIMA时间序列模型
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """时间序列分析工具类"""
    
    @staticmethod
    def adf_test(series: pd.Series, significance: float = 0.05) -> Dict:
        """
        ADF检验 (Augmented Dickey-Fuller Test)
        
        H0: 序列存在单位根 (非平稳)
        H1: 序列是平稳的
        
        Returns:
            包含检验统计量、p值、是否平稳等信息的字典
        """
        result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'lags_used': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < significance
        }
    
    @staticmethod
    def kpss_test(series: pd.Series) -> Dict:
        """KPSS检验 (另一个平稳性检验)"""
        from statsmodels.tsa.stattools import kpss
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05
        }
    
    @staticmethod
    def difference(series: pd.Series, order: int = 1) -> pd.Series:
        """差分使序列平稳"""
        diff_series = series.copy()
        for _ in range(order):
            diff_series = diff_series.diff()
        return diff_series.dropna()
    
    @staticmethod
    def find_arima_params(series: pd.Series, max_p: int = 5, 
                         max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        使用AIC/BIC自动搜索最优ARIMA参数
        
        Returns:
            (p, d, q)
        """
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        print(f"最优参数: {best_params}, AIC: {best_aic:.2f}")
        return best_params


class ARIMAModel:
    """ARIMA模型"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Args:
            order: (p, d, q) 参数
        """
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, series: pd.Series) -> 'ARIMAModel':
        """拟合ARIMA模型"""
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测未来值
        
        Returns:
            (预测值, 置信区间下限, 置信区间上限)
        """
        if self.fitted_model is None:
            raise ValueError("模型未训练")
        
        forecast = self.fitted_model.get_forecast(steps=steps)
        
        return forecast.predicted_mean.values, \
               forecast.conf_int()['lower'].values, \
               forecast.conf_int()['upper'].values
    
    def get_summary(self) -> str:
        """获取模型摘要"""
        if self.fitted_model is None:
            raise ValueError("模型未训练")
        return self.fitted_model.summary().as_text()


class AutoARIMA:
    """自动ARIMA模型 (使用pmdarima)"""
    
    def __init__(self, seasonal: bool = False, 
                 seasonal_period: int = 12,
                 information_criterion: str = 'aic'):
        """
        Args:
            seasonal: 是否考虑季节性
            seasonal_period: 季节周期
            information_criterion: 'aic' 或 'bic'
        """
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.ic = information_criterion
        self.model = None
    
    def fit(self, series: pd.Series) -> 'AutoARIMA':
        """自动拟合ARIMA"""
        try:
            from pmdarima import auto_arima
            
            self.model = auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.seasonal_period,
                information_criterion=self.ic,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            print(f"最优模型: {self.model.order()} (seasonal: {self.model.seasonal_order})")
        except ImportError:
            print("pmdarima未安装, 使用手动搜索")
            analyzer = TimeSeriesAnalyzer()
            order = analyzer.find_arima_params(series)
            self.model = ARIMAModel(order)
            self.model.fit(series)
        
        return self
    
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if hasattr(self.model, 'predict'):
            # pmdarima模型
            forecast = self.model.predict(n_periods=steps)
            conf_int = self.model.predict(n_periods=steps, return_conf_int=True)[1]
            
            return forecast, conf_int[:, 0], conf_int[:, 1]
        else:
            return self.model.predict(steps)


def rolling_forecast(series: pd.Series, train_size: int, 
                     steps: int, order: Tuple[int, int, int]) -> np.ndarray:
    """
    滚动预测 (Rolling Forecast)
    
    每次用真实值更新模型重新预测
    """
    predictions = []
    
    for i in range(train_size, len(series) - steps + 1):
        train = series[:i]
        model = ARIMAModel(order)
        model.fit(train)
        
        pred = model.predict(steps)[0][0]
        predictions.append(pred)
    
    return np.array(predictions)
