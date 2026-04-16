import datetime
from datetime import timedelta
from logging import getLogger
import time
from threading import Event, Lock
import pytz

import numpy as np
import pandas as pd
import json
import keras

from andersan import openmeteo, airmonitor, tile, prefecture_ranges
from icecream import ic

from andersan_core.periodic_features import periodic_features_numpy


class InvalidForecastingRangeError(Exception):
    """予報可能な時間範囲を越える場合に発生する例外"""

    pass


# 入力データが概ね1時間ごとに更新される前提で、同じ (県, 時刻, モデル条件) の再計算を避ける。
_PREDICT_OX_TTL_SEC = 3600
_PREDICT_OX_CACHE_MAX = 64
_PREDICT_OX_CACHE: dict = {}
_PREDICT_OX_CACHE_LOCK = Lock()
# 同一キーで計算中のリクエストを合流させる（キャッシュ投入前のサンダリング・ハード防止）。
# 重要: 合流とキャッシュはいずれも「このプロセス内」のみ有効。
# uvicorn の --workers 2 以上や gunicorn の複数ワーカーではプロセスが分かれるため、
# 同じリクエストが別ワーカーに振られると in-flight もキャッシュも共有されず、
# 重複計算が再び起き得る。プロセス横断で抑えたい場合は Redis 等の分散ロックや
# 外部ジョブキュー、もしくはワーカー数 1 にする運用を検討すること。
_PREDICT_OX_INFLIGHT: dict = {}


class _PredictOxInflight:
    __slots__ = ("event", "result", "error")

    def __init__(self):
        self.event = Event()
        self.result = None
        self.error = None


def _cache_key_isodate(isodate: str) -> str:
    if isodate == "now":
        now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        now = now.replace(minute=0, second=0, microsecond=0)
        return now.isoformat()
    dt = datetime.datetime.fromisoformat(isodate)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.timezone("Asia/Tokyo"))
    else:
        dt = dt.astimezone(pytz.timezone("Asia/Tokyo"))
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat()


def _predict_ox_cache_prune_expired(now: float) -> None:
    stale = [
        k
        for k, (deadline, _) in _PREDICT_OX_CACHE.items()
        if now >= deadline
    ]
    for k in stale:
        del _PREDICT_OX_CACHE[k]
    while len(_PREDICT_OX_CACHE) > _PREDICT_OX_CACHE_MAX:
        _PREDICT_OX_CACHE.pop(next(iter(_PREDICT_OX_CACHE)))


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
    tiles_max_retries: int = 3,
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
        tiles_max_retries (int): airmonitor.tiles（APW 取得）の HTTP 試行回数。1 で再試行なし。

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
    # 取得は新しい時刻から行い、保存は古い時刻順に並べる。
    # これにより最新データ欠損の検知を早める。
    air_tables_by_delta = []
    timeorigin = datetime.datetime.fromisoformat(isodatehour)
    missing_newest = False
    for delta in range(0, -lookback_length, -1):
        dt = timeorigin + timedelta(hours=delta)
        table = airmonitor.tiles(
            pref_name, dt.isoformat(), zoom, max_retries=tiles_max_retries
        )
        # print(f"prefecture: {pref_name}, isodate: {dt.isoformat()}")
        if table is None:
            ic(f"Missing data at TimeDelta={delta}={dt.isoformat()}")
        if delta == 0 and table is None:
            # 最新データを取得しそこねた
            # 代わりに、24時間前のデータを取得
            dt = timeorigin + timedelta(hours=-lookback_length)
            table = airmonitor.tiles(
                pref_name, dt.isoformat(), zoom, max_retries=tiles_max_retries
            )
            if table is None:
                raise RuntimeError(
                    f"Missing fallback observed data at {dt.isoformat()} (delta={-lookback_length})"
                )
            # 時系列順へ復元するため、deltaをキーとして保持
            air_tables_by_delta.append((-lookback_length, table))
            missing_newest = True
        else:
            if table is None:
                raise RuntimeError(
                    f"Missing observed data at {dt.isoformat()} (delta={delta})"
                )
            air_tables_by_delta.append((delta, table))

    # 保存順（古い→新しい）は従来どおり
    air_table = pd.concat(
        [tbl for _, tbl in sorted(air_tables_by_delta, key=lambda x: x[0])], axis=0
    )

    if missing_newest:
        # 現在時刻を1時間ずらす。
        timeorigin = timeorigin + timedelta(hours=-1)
        ic(f"Newest observed data are missing.")
        ic(air_table)

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
    # これが、タイルの順番を変えてしまっている。
    # ただ学習をする場合には順番はどうでもいいのだけど、可視化する時に、ほかのデータの並びと違うと困る。
    # tiles = np.unique(all_forecast_dataframe[["X", "Y"]].to_numpy(), axis=0)
    tiles, _ = tile.tiles(12, np.array(prefecture_ranges[pref_name]))

    X0 = np.zeros([len(tiles), lookback_length, len(cols_lookbacks)])
    X2 = np.zeros([len(tiles), forecast_length, len(cols_forecasts)])
    X3 = np.zeros([len(tiles), forecast_length], dtype=int)

    # ここで shape エラーが起きやすいので、どこで欠損したかを明示的に検出する。
    for j, (tileX, tileY) in enumerate(tiles):
        air_rows = air_table[(air_table.X == tileX) & (air_table.Y == tileY)]
        fc_rows = all_forecast_dataframe[
            (all_forecast_dataframe.X == tileX) & (all_forecast_dataframe.Y == tileY)
        ]
        if len(air_rows) != lookback_length:
            raise RuntimeError(
                "Failed to build lookback tensor from observed data: "
                f"tile=({tileX},{tileY}), expected_rows={lookback_length}, "
                f"actual_rows={len(air_rows)}, timeorigin={timeorigin.isoformat()}, "
                f"air_table_rows={len(air_table)}"
            )
        if len(fc_rows) != forecast_length:
            raise RuntimeError(
                "Failed to build forecast tensor from forecast data: "
                f"tile=({tileX},{tileY}), expected_rows={forecast_length}, "
                f"actual_rows={len(fc_rows)}, timebegin={timebegin.isoformat()}, "
                f"forecast_table_rows={len(all_forecast_dataframe)}"
            )

        for i, item in enumerate(cols_lookbacks):
            values = air_rows[item].to_numpy()
            if values.shape[0] != lookback_length:
                raise RuntimeError(
                    "Observed series length mismatch: "
                    f"item={item}, tile=({tileX},{tileY}), "
                    f"expected={lookback_length}, actual={values.shape[0]}"
                )
            X0[j, :, i] = values
            # if tileX == 3633 and tileY == 1617 and item == "OX":
            #     ic(X0[j, :, i], tileX, tileY)
        # print(X0)

        for i, item in enumerate(cols_forecasts):
            values = fc_rows[item].to_numpy()
            if values.shape[0] != forecast_length:
                raise RuntimeError(
                    "Forecast series length mismatch: "
                    f"item={item}, tile=({tileX},{tileY}), "
                    f"expected={forecast_length}, actual={values.shape[0]}"
                )
            X2[j, :, i] = values

        wc = fc_rows["weather_code"].to_numpy()
        if wc.shape[0] != forecast_length:
            raise RuntimeError(
                "Forecast weather_code length mismatch: "
                f"tile=({tileX},{tileY}), expected={forecast_length}, actual={wc.shape[0]}"
            )
        X3[j, :] = wc

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
        # standards.json に y2 など「入力テンソルに無い」統計だけが含まれることがある。
        if label not in X:
            continue
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
    tiles_max_retries: int = 3,
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
        tiles_max_retries (int): airmonitor.tiles（APW 取得）の HTTP 試行回数。

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
    # 取得は新しい時刻から行い、保存は古い時刻順に並べる。
    air_tables_by_delta = []
    timeorigin = datetime.datetime.fromisoformat(isodatehour)
    missing_newest = False
    for delta in range(0, -lookback_length, -1):
        dt = timeorigin + timedelta(hours=delta)
        table = airmonitor.tiles(
            "kanagawa", dt.isoformat(), zoom, max_retries=tiles_max_retries
        )
        if table is None:
            ic(f"TimeDelta={delta}")
        if delta == 0 and table is None:
            # 最新データを取得しそこねた
            # 代わりに、24時間前のデータを取得
            dt = timeorigin + timedelta(hours=-lookback_length)
            table = airmonitor.tiles(
                "kanagawa", dt.isoformat(), zoom, max_retries=tiles_max_retries
            )
            if table is None:
                raise RuntimeError(
                    f"Missing fallback observed data at {dt.isoformat()} (delta={-lookback_length})"
                )
            air_tables_by_delta.append((-lookback_length, table))
            missing_newest = True
        else:
            if table is None:
                raise RuntimeError(
                    f"Missing observed data at {dt.isoformat()} (delta={delta})"
                )
            air_tables_by_delta.append((delta, table))

    # 保存順（古い→新しい）は従来どおり
    air_table = pd.concat(
        [tbl for _, tbl in sorted(air_tables_by_delta, key=lambda x: x[0])], axis=0
    )

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
        values = air_table[item].to_numpy()
        if values.shape[0] != lookback_length:
            raise RuntimeError(
                "Single-tile observed series length mismatch: "
                f"item={item}, tile=({x},{y}), "
                f"expected={lookback_length}, actual={values.shape[0]}, "
                f"timeorigin={timeorigin.isoformat()}"
            )
        X0[0, :, i] = values

    for i, item in enumerate(cols_forecasts):
        values = all_forecast_dataframe[item].to_numpy()
        if values.shape[0] != forecast_length:
            raise RuntimeError(
                "Single-tile forecast series length mismatch: "
                f"item={item}, tile=({x},{y}), "
                f"expected={forecast_length}, actual={values.shape[0]}, "
                f"timebegin={timebegin.isoformat()}"
            )
        X2[0, :, i] = values

    wc = all_forecast_dataframe["weather_code"].to_numpy()
    if wc.shape[0] != forecast_length:
        raise RuntimeError(
            "Single-tile weather_code length mismatch: "
            f"tile=({x},{y}), expected={forecast_length}, actual={wc.shape[0]}"
        )
    X3[0, :] = wc

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
    tiles_max_retries: int = 3,
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
        tiles_max_retries (int): airmonitor.tiles（APW 取得）の HTTP 試行回数。

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
        tiles_max_retries,
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
    tiles_max_retries: int = 3,
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
        tiles_max_retries (int): airmonitor.tiles（APW 取得）の HTTP 試行回数。

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
        tiles_max_retries,
    )
    X = _standardize_data(X, stdfilename)
    return X


def _predict_ox_compute(
    prefecture,
    isodate,
    model,
    zoom,
    lookback_hours,
    forecast_hours,
    datatype,
    *,
    checkpoint_path=None,
    include_periodic=False,
    tiles_max_retries: int = 3,
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
        tiles_max_retries=tiles_max_retries,
    )
    timeorigin = X["timeorigin"]
    del X["timeorigin"]

    if include_periodic:
        u = np.array([int(timeorigin.timestamp())], dtype=np.int64)
        pv = periodic_features_numpy(u)[0]
        n = X["Input_lookbacks"].shape[0]
        X["Input_periodic"] = np.broadcast_to(pv, (n, 6)).copy()

    weights_path = checkpoint_path if checkpoint_path else f"{model}.py.best.keras"
    keras_model = keras.models.load_model(weights_path)

    # 予測
    pred = keras_model.predict(X)

    # andersan0_1はOX値の二乗を予測するので、ここで平方根をとって戻す。
    # 二乗を予測するのは、OXが大きい時の精度を高めるため。
    pred = pred**0.5

    # タイルと時刻の情報を得る
    table = airmonitor.tiles(
        "kanagawa", timeorigin.isoformat(), zoom, max_retries=tiles_max_retries
    )

    table = table.drop(columns=col_names["Input_lookbacks"])
    for i in range(forecast_hours):
        table[f"+{i+1}"] = pred[:, i]

    # NaNを-1にする。(jsonの制約のため)
    return table.fillna(-1)


def predict_ox(
    prefecture,
    isodate,
    model="andersan0_2",
    zoom=12,
    lookback_hours=24,
    forecast_hours=24,
    datatype="/AIR/andersan-train/datatype4",
    *,
    checkpoint_path=None,
    include_periodic=False,
    tiles_max_retries: int = 3,
):
    key = (
        prefecture,
        _cache_key_isodate(isodate),
        model,
        zoom,
        lookback_hours,
        forecast_hours,
        datatype,
        checkpoint_path,
        include_periodic,
        tiles_max_retries,
    )
    now = time.monotonic()
    with _PREDICT_OX_CACHE_LOCK:
        entry = _PREDICT_OX_CACHE.get(key)
        if entry is not None:
            deadline, cached_df = entry
            if now < deadline:
                return cached_df.copy()
            del _PREDICT_OX_CACHE[key]

        if key in _PREDICT_OX_INFLIGHT:
            state = _PREDICT_OX_INFLIGHT[key]
            leader = False
        else:
            state = _PredictOxInflight()
            _PREDICT_OX_INFLIGHT[key] = state
            leader = True

    if not leader:
        state.event.wait()
        if state.error is not None:
            raise state.error
        return state.result.copy()

    try:
        result = _predict_ox_compute(
            prefecture,
            isodate,
            model,
            zoom,
            lookback_hours,
            forecast_hours,
            datatype,
            checkpoint_path=checkpoint_path,
            include_periodic=include_periodic,
            tiles_max_retries=tiles_max_retries,
        )
    except BaseException as e:
        state.error = e
        raise
    else:
        state.result = result
        now_done = time.monotonic()
        with _PREDICT_OX_CACHE_LOCK:
            _PREDICT_OX_CACHE[key] = (now_done + _PREDICT_OX_TTL_SEC, result)
            if len(_PREDICT_OX_CACHE) > _PREDICT_OX_CACHE_MAX:
                _predict_ox_cache_prune_expired(now_done)
        return result.copy()
    finally:
        with _PREDICT_OX_CACHE_LOCK:
            _PREDICT_OX_INFLIGHT.pop(key, None)
        state.event.set()


def predict_ox_v0(prefecture, isodate, *, tiles_max_retries: int = 3):
    return predict_ox(
        prefecture,
        isodate,
        model="/AIR/andersan-train/andersan0_1",
        zoom=12,
        lookback_hours=24,
        forecast_hours=8,
        datatype="/AIR/andersan-train/datatype3",
        tiles_max_retries=tiles_max_retries,
    )


def predict_ox_v0a(prefecture, isodate, *, tiles_max_retries: int = 3):
    return predict_ox(
        prefecture,
        isodate,
        model="/AIR/andersan-train/andersan0_1_1",
        zoom=12,
        lookback_hours=24,
        forecast_hours=24,
        datatype="/AIR/andersan-train/datatype5",
        tiles_max_retries=tiles_max_retries,
    )


def predict_ox_v1(prefecture, isodate, *, tiles_max_retries: int = 3):
    return predict_ox(
        prefecture,
        isodate,
        model="/AIR/andersan-train/andersan0_2",
        zoom=12,
        lookback_hours=24,
        forecast_hours=24,
        datatype="/AIR/andersan-train/datatype4",
        tiles_max_retries=tiles_max_retries,
    )


def predict_ox_v1a(prefecture, isodate, *, tiles_max_retries: int = 3):
    return predict_ox(
        prefecture,
        isodate,
        model="/AIR/andersan-train/andersan0_2_1",
        zoom=12,
        lookback_hours=24,
        forecast_hours=24,
        datatype="/AIR/andersan-train/datatype4",
        tiles_max_retries=tiles_max_retries,
    )


def predict_ox_a1(prefecture, isodate, *, tiles_max_retries: int = 3):
    return predict_ox(
        prefecture,
        isodate,
        model="/AIR/andersan-train/andersan1",
        zoom=12,
        lookback_hours=24,
        forecast_hours=24,
        datatype="/AIR/andersan-train/datatype5idw",
        checkpoint_path="/AIR/andersan-train/andersan1.best.keras",
        include_periodic=True,
        tiles_max_retries=tiles_max_retries,
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
