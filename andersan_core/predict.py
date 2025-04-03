import datetime
from datetime import timedelta
from logging import getLogger
import pytz

import numpy as np
import pandas as pd
import json
import keras

from andersan import openmeteo, airmonitor
from icecream import ic


class InvalidForecastingRangeError(Exception):
    """予報可能な時間範囲を越える場合に発生する例外"""

    pass


def _prepare_data_for_nn(
    pref_name: str,
    isodatehour: str,
    zoom: int = 12,
    lookback_length: int = 24,
    forecast_length: int = 8,
    cols_lookbacks=("NMHC", "OX", "NOX", "TEMP", "WX", "WY"),
    cols_forecasts=(
        "temperature_2m",
        "cloud_cover",
        "pressure_msl",
        "shortwave_radiation",
        "wind_speed_10m",
    ),
    openweathermap=False,
) -> dict:
    """NNに入力する前のデータ準備を行う。

    Args:
        pref_name (str): 県名(半角ローマ字)
        isodatehour (str): 目的の日時
        zoom (int): 地理院タイルのレベル。12を想定。
        lookback_length (int, optional): 24を指定すると23時間前〜現在の大気測定値を利用. Defaults to 24.
        forecast_length (int, optional): 8を指定すると1時間先〜8時間先までの気象予報情報を利用_description_. Defaults to 8.
        cols_lookbacks (tuple, optional): 大気測定項目. Defaults to ("NMHC", "OX", "NOX", "TEMP", "WX", "WY").
        cols_forecasts (tuple, optional): 気象予報項目. Defaults to ( "temperature_2m", "cloud_cover", "pressure_msl", "shortwave_radiation", "wind_speed_10m", ).
        openweathermap (bool): OWMから予報値を入手する。いくつか条件を満たす場合に限り利用可能。

    Returns:
        dict: NNへの入力データX0, X2, X3とtimeoriginを含む辞書
    """
    logger = getLogger()

    # 現在時刻の予報で、日射量が必要ない場合に限りOpenWeathermapを指名できる。
    if openweathermap:
        if isodatehour == "now" and "shortwave_radiation" not in cols_forecasts:
            logger.info("OpenWeathermap is selected.")
        else:
            logger.info(
                "OpenWeathermap is requested but is not available for the specified condition."
            )
            openweathermap = False
        logger.info(
            "Anyway, OpenWeathermap is still not available for technical reasons."
        )

    # 時刻がnowになっている場合は日時に変換する。
    if isodatehour == "now":
        now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        now = now.replace(minute=0, second=0, microsecond=0)
        isodatehour = now.isoformat()

    # lookback値の読みこみ
    air_table = pd.DataFrame()
    timeorigin = datetime.datetime.fromisoformat(isodatehour)
    missing_newest = False
    for delta in range(-lookback_length + 1, 1):
        dt = timeorigin + timedelta(hours=delta)
        table = airmonitor.tiles(pref_name, dt.isoformat(), zoom)
        if table is None:
            ic(f"TimeDelta={delta}")
        if delta == 0 and table is None:
            # 最新データを取得しそこねた
            # 代わりに、24時間前のデータを取得
            dt = timeorigin + timedelta(hours=-lookback_length)
            table = airmonitor.tiles(pref_name, dt.isoformat(), zoom)
            # air_tableの先頭にくっつける
            air_table = pd.concat([table, air_table], axis=0)
            missing_newest = True
        else:
            air_table = pd.concat([air_table, table], axis=0)

    if missing_newest:
        # 現在時刻を1時間ずらす。
        timeorigin = timeorigin + timedelta(hours=-1)
        ic(f"Newest observed data are missing.")

    # 念のため、ほかの時刻表現を消しておく
    del isodatehour

    # forecast値の読みこみ
    timebegin = timeorigin + timedelta(hours=1)
    all_forecast_dataframe = openmeteo.tiles(
        pref_name,
        datehour=timebegin.strftime("%Y-%m-%dT%H"),
        hours=forecast_length,
        zoom=zoom,
    )
    tiles = np.unique(all_forecast_dataframe[["X", "Y"]].to_numpy(), axis=0)

    X0 = np.zeros([len(tiles), lookback_length, len(cols_lookbacks)])
    X2 = np.zeros([len(tiles), forecast_length, len(cols_forecasts)])
    X3 = np.zeros([len(tiles), forecast_length], dtype=int)

    # ここでエラーが出る、ということは、openmeteoではなく過去データの問題。なぜ?
    for j, (tileX, tileY) in enumerate(tiles):
        for i, item in enumerate(cols_lookbacks):
            X0[j, :, i] = air_table[(air_table.X == tileX) & (air_table.Y == tileY)][
                item
            ].to_numpy()
        # print(X0)

        for i, item in enumerate(cols_forecasts):
            X2[j, :, i] = all_forecast_dataframe[
                (all_forecast_dataframe.X == tileX)
                & (all_forecast_dataframe.Y == tileY)
            ][item]

        X3[j, :] = all_forecast_dataframe[
            (all_forecast_dataframe.X == tileX) & (all_forecast_dataframe.Y == tileY)
        ]["weather_code"]

    logger.info(X0.shape)
    logger.info(X2.shape)
    logger.info(X3.shape)

    return {
        "Input_lookbacks": X0,
        "Input_forecasts": X2,
        "Input_weathercodes": X3,
        "timeorigin": timeorigin,
    }


def _standardize_data(X: dict, stdfilename: str) -> dict:
    """データを標準化する。

    Args:
        X (dict): 標準化前のデータ
        stdfilename (str): 標準化係数ファイル名

    Returns:
        dict: 標準化後のデータ
    """
    logger = getLogger()
    logger.info(f"Standardization with {stdfilename}")

    with open(stdfilename) as f:
        specs = json.load(f)

    for label in specs:
        for icol in range(X[label].shape[-1]):
            average = specs[label][icol]["average"]
            std = specs[label][icol]["std"]
            X[label][:, :, icol] = (X[label][:, :, icol] - average) / std

    return X


def _prepare_data_for_nn_single_tile(
    x: int,
    y: int,
    isodatehour: str,
    zoom: int = 12,
    lookback_length: int = 24,
    forecast_length: int = 8,
    cols_lookbacks=("NMHC", "OX", "NOX", "TEMP", "WX", "WY"),
    cols_forecasts=(
        "temperature_2m",
        "cloud_cover",
        "pressure_msl",
        "shortwave_radiation",
        "wind_speed_10m",
    ),
    openweathermap=False,
) -> dict:
    """NNの入力データXを、単一のタイルに対して構築する。標準化前のデータ準備を行う。

    Args:
        x (int): 地理院タイルのX座標
        y (int): 地理院タイルのY座標
        isodatehour (str): 目的の日時
        zoom (int): 地理院タイルのレベル。12を想定。
        lookback_length (int, optional): 24を指定すると23時間前〜現在の大気測定値を利用. Defaults to 24.
        forecast_length (int, optional): 8を指定すると1時間先〜8時間先までの気象予報情報を利用_description_. Defaults to 8.
        cols_lookbacks (tuple, optional): 大気測定項目. Defaults to ("NMHC", "OX", "NOX", "TEMP", "WX", "WY").
        cols_forecasts (tuple, optional): 気象予報項目. Defaults to ( "temperature_2m", "cloud_cover", "pressure_msl", "shortwave_radiation", "wind_speed_10m", ).
        openweathermap (bool): OWMから予報値を入手する。いくつか条件を満たす場合に限り利用可能。

    Returns:
        dict: NNへの入力データX0, X2, X3とtimeoriginを含む辞書
    """
    logger = getLogger()

    # 現在時刻の予報で、日射量が必要ない場合に限りOpenWeathermapを指名できる。
    if openweathermap:
        if isodatehour == "now" and "shortwave_radiation" not in cols_forecasts:
            logger.info("OpenWeathermap is selected.")
        else:
            logger.info(
                "OpenWeathermap is requested but is not available for the specified condition."
            )
            openweathermap = False
        logger.info(
            "Anyway, OpenWeathermap is still not available for technical reasons."
        )

    # 時刻がnowになっている場合は日時に変換する。
    if isodatehour == "now":
        now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        now = now.replace(minute=0, second=0, microsecond=0)
        isodatehour = now.isoformat()

    # lookback値の読みこみ
    air_table = pd.DataFrame()
    timeorigin = datetime.datetime.fromisoformat(isodatehour)
    missing_newest = False
    for delta in range(-lookback_length + 1, 1):
        dt = timeorigin + timedelta(hours=delta)
        table = airmonitor.tiles("kanagawa", dt.isoformat(), zoom)
        if table is None:
            ic(f"TimeDelta={delta}")
        if delta == 0 and table is None:
            # 最新データを取得しそこねた
            # 代わりに、24時間前のデータを取得
            dt = timeorigin + timedelta(hours=-lookback_length)
            table = airmonitor.tiles("kanagawa", dt.isoformat(), zoom)
            # air_tableの先頭にくっつける
            air_table = pd.concat([table, air_table], axis=0)
            missing_newest = True
        else:
            air_table = pd.concat([air_table, table], axis=0)

    if missing_newest:
        # 現在時刻を1時間ずらす。
        timeorigin = timeorigin + timedelta(hours=-1)
        ic(f"Newest observed data are missing.")

    # 念のため、ほかの時刻表現を消しておく
    del isodatehour

    # forecast値の読みこみ
    timebegin = timeorigin + timedelta(hours=1)
    all_forecast_dataframe = openmeteo.tiles(
        "kanagawa",
        datehour=timebegin.strftime("%Y-%m-%dT%H"),
        hours=forecast_length,
        zoom=zoom,
    )

    # 単一タイルに絞り込む
    air_table = air_table[(air_table.X == x) & (air_table.Y == y)]
    all_forecast_dataframe = all_forecast_dataframe[
        (all_forecast_dataframe.X == x) & (all_forecast_dataframe.Y == y)
    ]

    X0 = np.zeros([1, lookback_length, len(cols_lookbacks)])
    X2 = np.zeros([1, forecast_length, len(cols_forecasts)])
    X3 = np.zeros([1, forecast_length], dtype=int)

    for i, item in enumerate(cols_lookbacks):
        X0[0, :, i] = air_table[item].to_numpy()

    for i, item in enumerate(cols_forecasts):
        X2[0, :, i] = all_forecast_dataframe[item].to_numpy()

    X3[0, :] = all_forecast_dataframe["weather_code"].to_numpy()

    logger.info(X0.shape)
    logger.info(X2.shape)
    logger.info(X3.shape)

    return {
        "Input_lookbacks": X0,
        "Input_forecasts": X2,
        "Input_weathercodes": X3,
        "timeorigin": timeorigin,
    }


def X1_openmeteo(
    x: int,
    y: int,
    isodatehour: str,
    zoom: int = 12,
    lookback_length: int = 24,
    forecast_length: int = 8,
    cols_lookbacks=("NMHC", "OX", "NOX", "TEMP", "WX", "WY"),
    cols_forecasts=(
        "temperature_2m",
        "cloud_cover",
        "pressure_msl",
        "shortwave_radiation",
        "wind_speed_10m",
    ),
    stdfilename="standards.json",
    openweathermap=False,
) -> dict:
    """NNの入力データXを、単一のタイルに対して構築する。

    Args:
        x (int): 地理院タイルのX座標
        y (int): 地理院タイルのY座標
        isodatehour (str): 目的の日時
        zoom (int): 地理院タイルのレベル。12を想定。
        lookback_length (int, optional): 24を指定すると23時間前〜現在の大気測定値を利用. Defaults to 24.
        forecast_length (int, optional): 8を指定すると1時間先〜8時間先までの気象予報情報を利用_description_. Defaults to 8.
        cols_lookbacks (tuple, optional): 大気測定項目. Defaults to ("NMHC", "OX", "NOX", "TEMP", "WX", "WY").
        cols_forecasts (tuple, optional): 気象予報項目. Defaults to ( "temperature_2m", "cloud_cover", "pressure_msl", "shortwave_radiation", "wind_speed_10m", ).
        stdfilename (str, optional): 各項目を標準化するための係数の情報のとりこみ. Defaults to "standards.json".
        openweathermap (bool): OWMから予報値を入手する。いくつか条件を満たす場合に限り利用可能。

    Returns:
        dict: 標準化されたNNへの入力データX0, X2, X3とtimeoriginを含む辞書
    """
    X = _prepare_data_for_nn_single_tile(
        x,
        y,
        isodatehour,
        zoom,
        lookback_length,
        forecast_length,
        cols_lookbacks,
        cols_forecasts,
        openweathermap,
    )
    X = _standardize_data(X, stdfilename)
    return X


def X_openmeteo(
    pref_name: str,
    isodatehour: str,
    zoom: int = 12,
    lookback_length: int = 24,
    forecast_length: int = 8,
    cols_lookbacks=("NMHC", "OX", "NOX", "TEMP", "WX", "WY"),
    cols_forecasts=(
        "temperature_2m",
        "cloud_cover",
        "pressure_msl",
        "shortwave_radiation",
        "wind_speed_10m",
    ),
    stdfilename="standards.json",
    openweathermap=False,
) -> dict:
    """NNの入力データXを構築する。

    Args:
        pref_name (str): 県名(半角ローマ字)
        isodatehour (str): 目的の日時
        zoom (int): 地理院タイルのレベル。12を想定。
        lookback_length (int, optional): 24を指定すると23時間前〜現在の大気測定値を利用. Defaults to 24.
        forecast_length (int, optional): 8を指定すると1時間先〜8時間先までの気象予報情報を利用_description_. Defaults to 8.
        cols_lookbacks (tuple, optional): 大気測定項目. Defaults to ("NMHC", "OX", "NOX", "TEMP", "WX", "WY").
        cols_forecasts (tuple, optional): 気象予報項目. Defaults to ( "temperature_2m", "cloud_cover", "pressure_msl", "shortwave_radiation", "wind_speed_10m", ).
        stdfilename (str, optional): 各項目を標準化するための係数の情報のとりこみ. Defaults to "standards.json".
        openweathermap (bool): OWMから予報値を入手する。いくつか条件を満たす場合に限り利用可能。

    Returns:
        dict: 標準化されたNNへの入力データX0, X2, X3とtimeoriginを含む辞書
    """
    X = _prepare_data_for_nn(
        pref_name,
        isodatehour,
        zoom,
        lookback_length,
        forecast_length,
        cols_lookbacks,
        cols_forecasts,
        openweathermap,
    )
    X = _standardize_data(X, stdfilename)
    return X


def predict_ox(
    prefecture,
    isodate,
    model="andersan0_2",
    zoom=12,
    lookback_hours=24,
    forecast_hours=24,
    datatype="/AIR/andersan-train/datatype4",
):
    stdfilename = datatype + "/standards.json"
    with open(datatype + "/columns.json") as f:
        col_names = json.load(f)

    # NNに食わせるデータの生成
    X = X_openmeteo(
        prefecture,
        isodate,
        zoom,
        lookback_length=lookback_hours,
        forecast_length=forecast_hours,
        cols_forecasts=col_names["Input_forecasts"],
        stdfilename=stdfilename,
    )

    timeorigin = X["timeorigin"]
    del X["timeorigin"]
    del isodate

    # モデルの準備
    model = keras.models.load_model(f"{model}.py.best.keras")

    # 予測
    pred = model.predict(X)

    # andersan0_1はOX値の二乗を予測するので、ここで平方根をとって戻す。
    # 二乗を予測するのは、OXが大きい時の精度を高めるため。
    pred = pred**0.5

    # タイルと時刻の情報を得る
    table = airmonitor.tiles("kanagawa", timeorigin.isoformat(), zoom)

    table = table.drop(columns=col_names["Input_lookbacks"])
    for i in range(forecast_hours):
        table[f"+{i+1}"] = pred[:, i]

    # NaNを-1にする。(jsonの制約のため)
    return table.fillna(-1)


predict_ox_v0 = lambda prefecture, isodate: predict_ox(
    prefecture,
    isodate,
    model="/AIR/andersan-train/andersan0_1",
    zoom=12,
    lookback_hours=24,
    forecast_hours=8,
    datatype="/AIR/andersan-train/datatype3",
)
predict_ox_v0a = lambda prefecture, isodate: predict_ox(
    prefecture,
    isodate,
    model="/AIR/andersan-train/andersan0_1_1",
    zoom=12,
    lookback_hours=24,
    forecast_hours=24,
    datatype="/AIR/andersan-train/datatype5",
)
predict_ox_v1 = lambda prefecture, isodate: predict_ox(
    prefecture,
    isodate,
    model="/AIR/andersan-train/andersan0_2",
    zoom=12,
    lookback_hours=24,
    forecast_hours=24,
    datatype="/AIR/andersan-train/datatype4",
)
predict_ox_v1a = lambda prefecture, isodate: predict_ox(
    prefecture,
    isodate,
    model="/AIR/andersan-train/andersan0_2_1",
    zoom=12,
    lookback_hours=24,
    forecast_hours=24,
    datatype="/AIR/andersan-train/datatype4",
)


def test():
    basicConfig(level=DEBUG)
    logger = getLogger()
    # logger.info(predict_ox_v0("kanagawa", "2025-02-20T09:00+09:00"))
    logger.info(predict_ox_v0("kanagawa", "2015-08-19T09:00+09:00"))
    # logger.info(predict_ox_v0("kanagawa", "now"))


if __name__ == "__main__":
    import os

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test()
