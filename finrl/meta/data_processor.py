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
        """
        Attach a rolling cross-sectional covariance matrix per time step.

        Works for both daily and intraday data:
        - prefers 'timestamp' if present, otherwise uses 'date'
        - deduplicates (time, tic) pairs and uses pivot_table with aggfunc='last'
        - computes returns by pct_change over time steps
        - builds a dict time -> covariance(matrix) over a rolling window of length 'lookback'
        - maps 'cov_list' back to the long dataframe
        - creates 'date' (calendar date), 'day' (factorized), and 'step' (time-step id)
        - sets index to 'step' so env can iterate 0..N-1 through time steps
        """
        data = df.copy()

        # choose the time column: intraday uses 'timestamp' (datetime), daily may only have 'date'
        time_col = "timestamp" if "timestamp" in data.columns else "date"
        if time_col not in data.columns:
            raise ValueError("add_covariance_matrix: requires 'timestamp' or 'date' column.")

        # ensure datetime dtype for time column
        data[time_col] = pd.to_datetime(data[time_col])

        # create a pure calendar date column for compatibility with env (date_memory etc.)
        data["date"] = data[time_col].dt.date

        # order and deduplicate (time, tic); keep the last seen price per time step
        data = (
            data.sort_values([time_col, "tic"])
            .drop_duplicates([time_col, "tic"], keep="last")
            .reset_index(drop=True)
        )

        # wide price matrix over time steps
        wide = (
            data.pivot_table(index=time_col, columns="tic", values="close", aggfunc="last")
            .sort_index()
        )

        # forward-fill missing prices across time (e.g., listing gaps or illiquid tics)
        wide = wide.ffill()

        # returns per step; use pct_change so env logic based on relative moves stays consistent
        rets = wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # build rolling covariance dictionary time -> cov matrix (assets x assets)
        times = rets.index.to_list()
        n_assets = wide.shape[1]
        cov_map = {}

        for i, t in enumerate(times):
            start = max(0, i - lookback + 1)
            window = rets.iloc[start: i + 1]
            if window.shape[0] < 2:
                cov = np.eye(n_assets, dtype=float)
            else:
                cov = np.cov(window.values.T)
                if not np.isfinite(cov).all():
                    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov_map[t] = cov

        # attach cov_list back to the long dataframe via full time column
        data["cov_list"] = data[time_col].map(cov_map)

        # build ids:
        # - 'day' = factorized calendar date (0,1,2,...)
        # - 'step' = factorized full timestamp/time (0..N-1), used as index for stepping
        data["day"] = pd.factorize(data["date"])[0].astype(int)
        unique_times = pd.Index(times)
        step_id_map = pd.Series(index=unique_times, data=np.arange(len(unique_times), dtype=int))
        data["step"] = data[time_col].map(step_id_map).astype(int)

        # final sort and index = 'step' so env.loc[self.day, :] fetches all tics for that time step
        out = data.sort_values([time_col, "tic"]).set_index("step")

        return out


