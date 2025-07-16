"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

import datetime
import time
from datetime import date
from datetime import timedelta
from sqlite3 import Timestamp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from stockstats import StockDataFrame as Sdf
from webdriver_manager.chrome import ChromeDriverManager

from finrl.config import DATA_SAVE_DIR


### Added by aymeric75 for scrap_data function


class YahooFinanceProcessor:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self):
        pass

    """
    Param
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers
    Example
    -------
    input:
    ticker_list = config_tickers.DOW_30_TICKER
    start_date = '2009-01-01'
    end_date = '2021-10-31'
    time_interval == "1D"

    output:
        date	    tic	    open	    high	    low	        close	    volume
    0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
    1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
    2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
    3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
    ...
    """

    @staticmethod
    def pad_prices(
            df: pd.DataFrame,
            price_col: str = "close",
            date_col: str = "timestamp",
            tic_col: str = "tic",
            calendar: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Uzupełnia brakujące kombinacje (data, ticker) bez dodawania dni,
        w które żadna spółka nie była notowana.

        Jeśli podasz argument `calendar`, zostanie użyty zamiast unii dat.
        """

        df = df.copy()

        # 1. porządkuj oś czasu
        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)

        # 2. pełny indeks dat: unia istniejących lub podany kalendarz
        if calendar is None:
            full_index = pd.DatetimeIndex(sorted(df[date_col].unique()))
        else:
            full_index = pd.DatetimeIndex(pd.to_datetime(calendar))

        # 3. macierz data × ticker
        mat = (df.pivot(index=date_col,
                        columns=tic_col,
                        values=price_col)
               .reindex(full_index))

        # 4. forward-fill, a brak wiodący – backward-fill
        mat = mat.ffill().bfill()

        # 5. z powrotem do long-format
        padded = (mat
                  .reset_index(names=date_col)
                  .melt(id_vars=date_col,
                        var_name=tic_col,
                        value_name=price_col)
                  .sort_values([date_col, tic_col])
                  .reset_index(drop=True))

        # 6. scal pozostałe kolumny; tam gdzie powstały nowe wiersze,
        #    wolumen = 0 (albo inna neutralna wartość)
        other_cols = [c for c in df.columns
                      if c not in (date_col, tic_col, price_col)]

        out = (padded
               .merge(df.drop(columns=price_col),
                      on=[date_col, tic_col],
                      how="left"))

        for col in other_cols:
            if out[col].dtype.kind in "fi":  # liczby
                out[col] = out[col].fillna(0.0)
            else:  # kategorie / obiekty
                out[col] = out[col].fillna(method="ffill").fillna(method="bfill")

        return out

    ######## ADDED BY aymeric75 ###################
    def date_to_unix(self, date_str) -> int:
        """Convert a date string in yyyy-mm-dd format to Unix timestamp."""
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_stock_data(self, stock_name, period1, period2) -> pd.DataFrame:
        # Base URL
        url = f"https://finance.yahoo.com/quote/{stock_name}/history/?period1={period1}&period2={period2}&filter=history"

        # Selenium WebDriver Setup
        options = Options()
        options.add_argument("--headless")  # Headless for performance
        options.add_argument("--disable-gpu")  # Disable GPU for compatibility
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        # Navigate to the URL
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)  # Wait for redirection and page load

        # Handle potential popup
        try:
            RejectAll = driver.find_element(
                By.XPATH, '//button[@class="btn secondary reject-all"]'
            )
            action = ActionChains(driver)
            action.click(on_element=RejectAll)
            action.perform()
            time.sleep(5)

        except Exception as e:
            print("Popup not found or handled:", e)

        # Parse the page for the table
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if not table:
            raise Exception("No table found after handling redirection and popup.")

        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        headers[4] = "Close"
        headers[5] = "Adj Close"
        headers = ["date", "open", "high", "low", "close", "adjcp", "volume"]
        # , 'tic', 'day'

        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header row
            cells = [td.text.strip() for td in tr.find_all("td")]
            if len(cells) == len(headers):  # Only add rows with correct column count
                rows.append(cells)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Convert columns to appropriate data types
        def safe_convert(value, dtype):
            try:
                return dtype(value.replace(",", ""))
            except ValueError:
                return value

        df["open"] = df["open"].apply(lambda x: safe_convert(x, float))
        df["high"] = df["high"].apply(lambda x: safe_convert(x, float))
        df["low"] = df["low"].apply(lambda x: safe_convert(x, float))
        df["close"] = df["close"].apply(lambda x: safe_convert(x, float))
        df["adjcp"] = df["adjcp"].apply(lambda x: safe_convert(x, float))
        df["volume"] = df["volume"].apply(lambda x: safe_convert(x, int))

        # Add 'tic' column
        df["tic"] = stock_name

        # Add 'day' column
        start_date = datetime.datetime.fromtimestamp(period1)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - start_date).dt.days
        df = df[df["day"] >= 0]  # Exclude rows with days before the start date

        # Reverse the DataFrame rows
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    def scrap_data(self, stock_names, start_date, end_date) -> pd.DataFrame:
        """Fetch and combine stock data for multiple stock names."""
        period1 = self.date_to_unix(start_date)
        period2 = self.date_to_unix(end_date)

        all_dataframes = []
        total_stocks = len(stock_names)

        for i, stock_name in enumerate(stock_names):
            try:
                print(
                    f"Processing {stock_name} ({i + 1}/{total_stocks})... {(i + 1) / total_stocks * 100:.2f}% complete."
                )
                df = self.fetch_stock_data(stock_name, period1, period2)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error fetching data for {stock_name}: {e}")

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(by=["day", "tick"]).reset_index(drop=True)

        return combined_df

    ######## END ADDED BY aymeric75 ###################

    def convert_interval(self, time_interval: str) -> str:
        # Convert FinRL 'standardised' time periods to Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        yahoo_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]
        if time_interval in yahoo_intervals:
            return time_interval
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    from pathlib import Path

    def download_data(
            self,
            ticker_list: list[str],
            start_date: str,
            end_date: str,
            time_interval: str,
            proxy: str | dict = None,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        # remember arguments for later functions
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # ---------- NEW: common cache folder -------------------------------
        cache_dir = Path(DATA_SAVE_DIR)  # DATA_SAVE_DIR is already created in main.py
        cache_dir.mkdir(parents=True, exist_ok=True)
        # -------------------------------------------------------------------

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        intraday = self.time_interval.endswith("m") or self.time_interval in {"1h"}
        chunk = timedelta(days=7 if intraday else (end_ts - start_ts).days + 1)

        data_df = pd.DataFrame()
        for tic in ticker_list:

            # ---------- NEW: filename and loading -------------------------
            fname = f"{tic}_{start_date}_{end_date}_{time_interval}.parquet"
            fpath = cache_dir / fname
            if fpath.exists():
                tmp_df = pd.read_parquet(fpath)
                data_df = pd.concat([data_df, tmp_df])
                continue
            # --------------------------------------------------------------

            current = start_ts
            pieces = []
            while current <= end_ts:
                tmp_df = yf.download(
                    tic,
                    start=current,
                    end=min(current + chunk, end_ts + timedelta(days=1)),
                    interval=self.time_interval,
                    proxy=proxy,
                    progress=False,
                    auto_adjust=True,
                    repair=True,
                )
                if tmp_df.empty:
                    current += chunk
                    continue
                if tmp_df.columns.nlevels != 1:
                    tmp_df.columns = tmp_df.columns.droplevel(1)
                tmp_df["tic"] = tic
                pieces.append(tmp_df)
                current += chunk

            if not pieces:  # no data fetched (IPO later than start, etc.)
                continue
            tmp_df = pd.concat(pieces)
            # ---------- NEW: save to cache --------------------------------
            tmp_df.to_parquet(fpath, compression="snappy")
            # --------------------------------------------------------------
            data_df = pd.concat([data_df, tmp_df])

        # still return the unified dataframe expected by downstream code
        if data_df.empty:
            raise ValueError("No price data downloaded or found in cache.")

        data_df = (
            data_df.reset_index()
            .drop(columns=["Adj Close"], errors="ignore")
            .rename(
                columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
        )
        # ensure column order
        data_df = data_df[["timestamp", "close", "high", "low", "open", "volume", "tic"]]
        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        tic_list = np.unique(df.tic.values)
        NY = "America/New_York"

        trading_days = self.get_trading_days(start=self.start, end=self.end)
        # produce full timestamp index
        if self.time_interval == "1d":
            times = pd.to_datetime(trading_days)
        elif self.time_interval == "1m":
            times = []
            for day in trading_days:
                #                NY = "America/New_York"
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                for i in range(390):  # 390 minutes in trading day
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )

        # create a new dataframe with full timestamp series
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times, dtype=float
            )
            tic_df = df[
                df.tic == tic
            ]  # extract just the rows from downloaded data relating to this tic
            for i in range(tic_df.shape[0]):  # fill empty DataFrame using original data
                tmp_timestamp = tic_df.iloc[i]["timestamp"]
                tmp_timestamp = pd.to_datetime(tmp_timestamp).tz_localize(None)

                row = tic_df.iloc[i][["open", "high", "low", "close", "volume"]]
                if row.isna().all():
                    continue
                tmp_df.loc[tmp_timestamp] = row

            # print("(9) tmp_df\n", tmp_df.to_string()) # print ALL dataframe to check for missing rows from download

            # if close on start date is NaN, fill data with first valid close
            # and set volume to 0.
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print("NaN data on start date, fill using first valid data.")
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_close = tmp_df.iloc[i]["close"]
                        tmp_df.iloc[0] = [
                            first_valid_close,
                            first_valid_close,
                            first_valid_close,
                            first_valid_close,
                            0.0,
                        ]
                        break

            # if the close price of the first row is still NaN (All the prices are NaN in this case)
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "Missing data for ticker: ",
                    tic,
                    " . The prices are all NaN. Fill with 0.",
                )
                tmp_df.iloc[0] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

            # fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
                    # print(tmp_df.iloc[i], " Filled NaN data with previous close and set volume to 0. ticker: ", tic)

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        #            print(("Data clean for ") + tic + (" is finished."))

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        #        print("Data clean all finished!")

        return new_df

    def add_technical_indicator(self, data: pd.DataFrame, tech_indicator_list: list[str]) -> pd.DataFrame:
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        # ─────────────────────────────────────────────────────────
        # Fill missing values smartly after adding technical indicators
        # ─────────────────────────────────────────────────────────

        for col in tech_indicator_list:
            if col in df.columns:
                if col in ["macd", "cci_30", "dx_30"]:
                    df[col] = df[col].fillna(0.0)  # momentum indicators neutral
                elif col in ["rsi_30"]:
                    df[col] = df[col].fillna(50.0)  # RSI neutral
                elif col in ["boll_ub", "boll_lb"]:
                    df[col] = df[col].fillna(df["close"])  # fallback to close price
                else:
                    df[col] = df[col].fillna(0.0)  # default safe fallback

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        df = df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)

        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        vix_df = self.download_data(["^VIX"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        #print("cleaned_vix\n", cleaned_vix)
        vix = cleaned_vix[["timestamp", "close"]]
        #print('cleaned_vix[["timestamp", "close"]\n', vix)
        vix = vix.rename(columns={"close": "VIXY"})
        #print('vix.rename(columns={"close": "VIXY"}\n', vix)

        df = data.copy()
        #print("df\n", df)
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = self.pad_prices(df.copy())
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        print("Successfully transformed into array")

        price_array = np.nan_to_num(price_array)
        tech_array = np.nan_to_num(tech_array)
        turbulence_array = np.nan_to_num(turbulence_array)

        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start: str, end: str) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
        return trading_days

    # ****** NB: YAHOO FINANCE DATA MAY BE IN REAL-TIME OR DELAYED BY 15 MINUTES OR MORE, DEPENDING ON THE EXCHANGE ******
    def fetch_latest_data(
        self,
        ticker_list: list[str],
        time_interval: str,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(
            minutes=limit + 1
        )  # get the last rows up to limit

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )  # use start and end datetime to simulate the limit parameter
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(
            columns=["Adj Close"]
        )  # Alpaca data does not have 'Adj Close'

        data_df.columns = [  # convert to Alpaca column names lowercase
            "timestamp",
            "close",
            "high",
            "low",
            "open",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times, dtype=float
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb
