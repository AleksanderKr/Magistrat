from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)


class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        if data_source == "alpaca":
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                API_BASE_URL = kwargs.get("API_BASE_URL")
                self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
                print("Alpaca successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "wrds":
            self.processor = Wrds()

        elif data_source == "yahoofinance":
            self.processor = YahooFinance()

        else:
            raise ValueError("Data source input is NOT supported yet.")

        # Initialize variable in case it is using cache and does not use download_data() method
        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)
        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)
        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)
        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)
        return df

    def add_vixor(self, df) -> pd.DataFrame:
        df = self.processor.add_vixor(df)
        return df

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0
        return price_array, tech_array, turbulence_array

    def add_covariance_matrix(self, df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
        data = df.copy()

        if "date" not in data.columns:
            if "timestamp" not in data.columns:
                raise ValueError("add_covariance_matrix: requires 'timestamp' or 'date' column.")
            data["date"] = pd.to_datetime(data["timestamp"]).dt.date

        price_pivot = (
            data.pivot(index="date", columns="tic", values="close")
            .sort_index()
        )
        returns = price_pivot.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        dates = returns.index.to_list()
        cov_list = []
        n_assets = len(price_pivot.columns)

        for i in range(len(dates)):
            start = max(0, i - lookback + 1)
            window = returns.iloc[start:i+1]
            if window.shape[0] < 2:
                cov = np.eye(n_assets, dtype=float)
            else:
                cov = np.cov(window.values.T)
                if np.isnan(cov).any() or np.isinf(cov).any():
                    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov_list.append(cov)

        cov_df = pd.DataFrame({"date": dates, "cov_list": cov_list})
        out = data.merge(cov_df, on="date", how="left").sort_values(["timestamp", "tic"])

        out["day"] = pd.factorize(out["date"])[0]  # 0,1,2,...
        out = out.set_index("day")

        return out


