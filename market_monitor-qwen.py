# market_monitor.py
# 市场环境监控仪表盘 - GitHub Actions适配版 v5.1
# 核心升级：引入 ADX, RSI, ATR, OBV, 线性回归等高级量化因子
# 修复：IndentationError in identify_bottom_fractal

import subprocess
import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import warnings
import socket
import requests
import pickle
import glob
from pathlib import Path

# ================= 核心配置 =================
socket.setdefaulttimeout(15)
requests.adapters.DEFAULT_RETRIES = 2
requests.packages.urllib3.disable_warnings()
warnings.filterwarnings('ignore')

# GitHub Actions中使用简洁日志格式
is_github = os.environ.get('GITHUB_ACTIONS') == 'true'
if is_github:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)

# GitHub Actions中跳过交互式安装，但需确保关键库存在
def ensure_packages():
    required = ["akshare", "pandas", "numpy", "openpyxl", "yfinance", "scipy"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            if is_github:
                logger.error(f"Missing package: {pkg}. Please add to requirements.txt or install step.")
                # In GH Actions, we might want to fail fast or try install
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(f"Installed {pkg} on the fly.")
                except Exception as e:
                    logger.error(f"Failed to install {pkg}: {e}")                    sys.exit(1)
            else:
                logger.info(f"Installing missing package: {pkg}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

ensure_packages()

import akshare as ak
import yfinance as yf
try:
    from scipy import stats
except ImportError:
    logger.warning("Scipy not available, some advanced trend features will be disabled.")
    stats = None

# 北京时间
utc_now = datetime.now(timezone.utc)
beijing_now = utc_now.astimezone(timezone(timedelta(hours=8)))
TODAY = beijing_now.strftime("%Y-%m-%d")

# 监控标的配置
monitor_stocks = {
    "中信证券": {"code": "600030", "role": "压舱石_望远镜", "weight": 0.55},
    "赤峰黄金": {"code": "600988", "role": "避险矛_显微镜", "weight": 0.30},
    "西藏矿业": {"code": "000762", "role": "弹性牌_信号灯", "weight": 0.15}
}

FACTOR_COLS = ["trend_score", "momentum_score", "volatility_score", "volume_score",
               "rel_strength_score", "gold_beta_score", "reversal_score", "breakout_score", "vol_compress_score"]

# ================= 缓存管理 =================
def get_cache_dir():
    """获取缓存目录"""
    cache_dir = Path(os.environ.get('CACHE_DIR', 'cache'))
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def load_cache(code, data_type="stock", max_age_hours=4):
    """加载缓存数据"""
    cache_file = get_cache_dir() / f"{data_type}_{code}_{TODAY}.pkl"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                    logger.info(f"使用缓存: {data_type}_{code} (已缓存{age_hours:.1f}小时)")
                    return df
            except Exception as e:
                logger.debug(f"缓存读取失败: {e}, 删除旧缓存")                cache_file.unlink(missing_ok=True)
    return None

def save_cache(df, code, data_type="stock"):
    """保存缓存数据"""
    try:
        cache_file = get_cache_dir() / f"{data_type}_{code}_{TODAY}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.debug(f"缓存保存失败: {e}")

# ================= 核心工具函数 =================
def validate_data_freshness(df, max_delay_days=3):
    """验证数据新鲜度"""
    try:
        last_date = pd.to_datetime(df["date"].iloc[-1])
        delta = (datetime.now() - last_date).days
        return delta <= max_delay_days, delta
    except:
        return False, -1

def safe_ak_call(func, *args, **kwargs):
    """安全的API调用包装器"""
    func_name = getattr(func, '__name__', str(func))
    logger.info(f"  [请求] {func_name}...")
    try:
        res = func(*args, **kwargs)
        if res is not None and not (isinstance(res, pd.DataFrame) and res.empty):
            logger.info(f"  [成功] {func_name}")
            return res
        else:
            logger.warning(f"  [警告] {func_name} 返回空")
            return None
    except socket.timeout:
        logger.error(f"  [超时] {func_name} (15s)")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"  [网络] {func_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"  [失败] {func_name}: {e}")
        return None

def normalize_turnover(x, source="default"):
    """成交额单位自适应归一化"""
    try:
        val = float(str(x).replace(',', ''))
    except:
        return None    if pd.isna(val) or val <= 0:
        return None
    if val > 1e11:
        return val / 1e8
    elif val > 1e7:
        return val / 1e4
    elif val > 1e3:
        return val
    return val

def calc_ic_weights(df, factor_cols, target_col="ret", lookback=60):
    """IC权重计算（Rank IC）"""
    ic_dict = {}
    recent = df.tail(lookback)
    for col in factor_cols:
        if col not in df.columns:
            continue
        valid = recent[[col, target_col]].dropna()
        if len(valid) < 20:
            ic_dict[col] = 0
            continue
        # 使用 Spearman 秩相关系数
        ic = valid[col].rank().corr(valid[target_col].shift(-1).rank(), method='spearman')
        ic_dict[col] = ic if not np.isnan(ic) else 0
    
    total = sum(abs(v) for v in ic_dict.values()) + 1e-6
    weights = {k: abs(v)/total for k, v in ic_dict.items()}
    return weights

def identify_bottom_fractal(df):
    """识别底分型形态"""
    df = df.copy()
    required = ["open", "high", "low", "close", "volume"]
    if len(df) < 5 or not all(c in df.columns for c in required):
        return False, None, {"底分型形态": "否(数据不足)", "止损参考价": None, "今日量比": 1.0}
    
    recent = df.tail(5)
    p2, p1, p0 = recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]
    
    # 处理包含关系
    if p1['high'] <= p2['high'] and p1['low'] >= p2['low'] and len(df) >= 6:
        p2, p1, p0 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
        
    # 趋势过滤：如果MA20在MA60之上，通常不视为底部反转信号，除非是回调企稳
    if "ma20" in df.columns and "ma60" in df.columns and len(df) >= 5:
        if df["ma20"].iloc[-3] >= df["ma60"].iloc[-3]:
             # 多头趋势中的底分型可能只是回调，这里简化处理，仍允许检测但降低置信度
             pass 

    is_bottom = (p1['low'] < p2['low'] and p1['low'] < p0['low'] and                  p1['high'] < p2['high'] and p1['high'] < p0['high'])
    is_confirmed = p0['close'] > p1['high']
    
    vol_min10 = df['volume'].rolling(10).min().iloc[-1]
    vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
    is_strict = p1['volume'] <= vol_min10 * 1.2
    is_normal = p1['volume'] < vol_ma5 * 0.8
    
    vol_ratio = df['volume'].iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1.0
    
    # --- 修复缩进开始 ---
    p0_body = abs(p0['close'] - p0['open'])
    p0_range = max(p0['high'] - p0['low'], 0.01)
    is_solid = p0_body / p0_range > 0.3
    
    details = {
        "底分型形态": "是" if is_bottom else "否",
        "右侧确认": "是" if is_confirmed else "否",
        "严格缩量": "是" if is_strict else "否",
        "普通缩量": "是" if is_normal else "否",
        "实体确认": "是" if is_solid else "否",
        "止损参考价": round(p1['low'], 2) if is_bottom else None,
        "今日量比": round(vol_ratio, 2)
    }
    
    strict_res = is_bottom and is_confirmed and is_strict and is_solid
    loose_res = is_bottom and is_confirmed and is_normal
    
    if strict_res or loose_res:
        return True, p1['low'], details
    return False, None, details
    # --- 修复缩进结束 ---

# ================= 数据获取层 =================
def get_stock_data(code, tail_size=500, use_cache=True):
    """获取个股历史数据（多数据源）"""
    if use_cache:
        cached = load_cache(code, "stock")
        if cached is not None and len(cached) >= 60:
            fresh, delta = validate_data_freshness(cached)
            if fresh:
                return cached
    
    # 数据源1：yfinance
    logger.info(f"  [数据源1] yfinance {code}...")
    try:
        suffix = ".SS" if code.startswith("6") else ".SZ"
        ticker = yf.Ticker(f"{code}{suffix}")
        df = ticker.history(period="2y", timeout=15)
        if df is not None and len(df) > 60:            df = df.tail(tail_size).copy().reset_index()
            df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                               "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df["ret"] = df["close"].pct_change()
            df = df.dropna(subset=["close"])
            df["source"] = "yf"
            fresh, delta = validate_data_freshness(df)
            if not fresh:
                logger.warning(f"  {code} yfinance数据可能过期: {delta}天")
            logger.info(f"  [成功] yfinance {code}: {len(df)}条 最新{df['date'].iloc[-1]}")
            if use_cache:
                save_cache(df, code, "stock")
            return df
    except Exception as e:
        logger.warning(f"  [降级] yfinance失败: {e}")
    
    # 数据源2：东方财富历史日线
    df = safe_ak_call(ak.stock_zh_a_hist, symbol=code, period="daily", adjust="qfq")
    if df is not None and len(df) > 60:
        df = df.tail(tail_size).copy()
        df.rename(columns={"收盘": "close", "开盘": "open", "最高": "high", 
                           "最低": "low", "成交量": "volume", "日期": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["ret"] = df["close"].pct_change()
        df = df.dropna(subset=["close"])
        if use_cache:
            save_cache(df, code, "stock")
        return df
    
    # 数据源3：腾讯历史日线
    start_dt = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    df = safe_ak_call(ak.stock_zh_a_hist_tx, symbol=code, period="daily", start_date=start_dt)
    if df is not None and len(df) > 60:
        df = df.tail(tail_size).copy()
        df.rename(columns={"收盘": "close", "开盘": "open", "最高": "high", 
                           "最低": "low", "成交量": "volume", "日期": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["ret"] = df["close"].pct_change()
        df = df.dropna(subset=["close"])
        if use_cache:
            save_cache(df, code, "stock")
        return df
    
    # 数据源4：东方财富日线
    full_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
    df = safe_ak_call(ak.stock_zh_a_daily, symbol=full_code, adjust="qfq")
    if df is not None and len(df) > 60:        df = df.tail(tail_size).copy()
        for c in ["close", "open", "high", "low", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=["close"])
        df["ret"] = df["close"].pct_change()
        if "date" not in df.columns:
            df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq='B').strftime("%Y-%m-%d")
        else:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if use_cache:
            save_cache(df, code, "stock")
        return df
    
    logger.error(f"{code} 所有数据源失败")
    return pd.DataFrame()

def get_index_data(tail_size=500, use_cache=True):
    """获取上证指数数据"""
    if use_cache:
        cached = load_cache("sh000001", "index")
        if cached is not None:
            return cached[0], cached[1]
    
    df_result, turnover = pd.DataFrame(), None
    
    # 数据源1：东方财富指数日线
    df = safe_ak_call(ak.stock_zh_index_daily_em, symbol="sh000001")
    if df is not None and len(df) > 0:
        df = df.tail(tail_size).copy()
        date_col = next((c for c in df.columns if 'date' in str(c).lower() or '日期' in str(c)), df.columns[0])
        close_col = next((c for c in df.columns if 'close' in str(c).lower() or '收盘' in str(c)), None)
        amount_col = next((c for c in df.columns if ('amount' in str(c).lower() or '成交额' in str(c)) 
                          and 'volume' not in str(c).lower() and '成交量' not in str(c)), None)
        df.rename(columns={date_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if close_col:
            df["close"] = pd.to_numeric(df[close_col], errors='coerce')
            df["ret"] = df["close"].pct_change()
        if amount_col:
            raw = df[amount_col].iloc[-1]
            turnover = normalize_turnover(raw, "em")
            logger.info(f"  [成交额] 原始值:{raw} -> 归一化:{turnover}亿")
            if turnover and turnover < 3000:
                logger.warning(f"  沪市成交额异常偏低({turnover:.0f}亿)，触发降级")
                turnover = None
            else:
                df_result = df
                if turnover:
                    if use_cache:                        save_cache((df_result, turnover), "sh000001", "index")
                    return df_result, turnover
    
    # 数据源2：腾讯指数日线
    df = safe_ak_call(ak.stock_zh_index_daily_tx, symbol="sh000001")
    if df is not None and len(df) > 0:
        df = df.tail(tail_size).copy()
        df.rename(columns={"收盘": "close", "日期": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["ret"] = df["close"].pct_change()
        if df_result.empty:
            df_result = df
        if not turnover:
            for col in df.columns:
                if ('成交额' in str(col) or 'amount' in str(col).lower()) and '成交量' not in str(col):
                    raw_val = df[col].iloc[-1]
                    turnover = normalize_turnover(raw_val, "tx")
                    if turnover and turnover < 3000:
                        turnover = None
                    elif turnover:
                        break
    
    # 数据源3：实时指数
    if not turnover:
        spot = safe_ak_call(ak.stock_zh_index_spot_em)
        if spot is not None:
            sh = spot[spot["代码"].astype(str).str.contains("000001|上证指数")]
            if not sh.empty:
                for col in sh.columns:
                    if ('成交额' in str(col) or 'amount' in str(col).lower()) and '成交量' not in str(col):
                        turnover = normalize_turnover(sh[col].iloc[0], "spot")
                        if turnover and turnover < 3000:
                            turnover = None
                        elif turnover:
                            break
    
    if turnover:
        logger.info(f"  上证指数成交额: {turnover:.0f}亿")
    else:
        logger.warning("  成交额未获取，使用默认值0.5")
    
    if use_cache and not df_result.empty:
        save_cache((df_result, turnover), "sh000001", "index")
    
    return df_result if not df_result.empty else pd.DataFrame(), turnover

def get_gold_data(tail_size=500, use_cache=True):
    """获取黄金数据"""
    if use_cache:
        cached = load_cache("AU0", "gold")        if cached is not None:
            return cached
    
    # 数据源1：新浪期货主力合约
    df = safe_ak_call(ak.futures_main_sina, symbol="AU0")
    if df is not None and len(df) > 0:
        df = df.tail(tail_size).copy()
        date_col = next((c for c in df.columns if 'date' in str(c).lower() or '日期' in str(c)), df.columns[0])
        close_col = next((c for c in df.columns if 'close' in str(c).lower() or '收盘' in str(c) or '价' in str(c)), df.columns[-1])
        df.rename(columns={date_col: "date", close_col: "close"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
        df["ret"] = df["close"].pct_change()
        df = df.dropna(subset=["close"])
        if use_cache:
            save_cache(df, "AU0", "gold")
        return df
    
    # 数据源2：实时黄金数据
    df = safe_ak_call(ak.spot_gold)
    if df is not None and len(df) > 0:
        df = df.tail(tail_size).copy()
        df.rename(columns={"价格": "close"}, inplace=True)
        if "日期" in df.columns:
            df.rename(columns={"日期": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        else:
            return pd.DataFrame()
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
        df["ret"] = df["close"].pct_change()
        df = df.dropna(subset=["close", "date"])
        if use_cache:
            save_cache(df, "AU0", "gold")
        return df
    
    return pd.DataFrame()

# ================= 高级因子计算层 (New in v5.0) =================

def calc_advanced_volatility(df):
    """
    高级波动率诊断
    1. ATR (14日)
    2. Downside Volatility
    3. Bollinger Band Width Squeeze
    """
    df = df.copy()
    try:
        # 1. ATR
        high_low = df['high'] - df['low']        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        # 2. Downside Volatility
        neg_ret = df['ret'].apply(lambda x: x if x < 0 else 0)
        df['downside_vol'] = neg_ret.rolling(20).std()
        
        # 3. BB Width
        bb_upper = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        bb_lower = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        bb_width = (bb_upper - bb_lower) / df['close'].rolling(20).mean()
        df['vol_compress_rank'] = bb_width.rolling(60).rank(pct=True)
        
        # Risk Score: High downside vol is bad. Low compression (high rank) means expansion.
        # We want low risk and potentially pre-breakout (low compression rank)
        df['risk_score'] = df['downside_vol'].rolling(60).rank(pct=True)
        
        # Advanced Vol Score: 1 - Risk. Higher is better (stable). 
        # If you prefer breakout detection, invert logic. Here we prioritize stability for "Monitor".
        df['advanced_vol_score'] = (1 - df['risk_score']).clip(0, 1)
        
    except Exception as e:
        logger.debug(f"Advanced Volatility Calc Failed: {e}")
        df['advanced_vol_score'] = 0.5
    return df

def calc_advanced_trend(df):
    """
    高级趋势诊断
    1. ADX (14)
    2. Linear Regression Slope & R2
    """
    df = df.copy()
    try:
        if stats is None:
            raise ImportError("Scipy stats not available")

        # 1. ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([df['high']-df['low'], 
                        np.abs(df['high']-df['close'].shift(1)), 
                        np.abs(df['low']-df['close'].shift(1))], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx_14'] = dx.rolling(14).mean()
        
        # 2. LinReg
        def linreg_metrics(series):
            if len(series) < 5: return 0, 0
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            return slope, r_value**2

        slopes = []
        r_sqs = []
        # Vectorized approximation for speed if needed, but loop is fine for <500 rows
        for i in range(len(df)):
            if i < 19:
                slopes.append(np.nan)
                r_sqs.append(np.nan)
            else:
                s, r2 = linreg_metrics(df['close'].iloc[i-19:i+1])
                slopes.append(s)
                r_sqs.append(r2)
                
        df['trend_slope'] = slopes
        df['trend_r2'] = r_sqs
        
        # Scoring
        direction_score = (df['trend_slope'] > 0).astype(float)
        strength_score = df['adx_14'].clip(0, 50) / 50 
        quality_score = df['trend_r2'].fillna(0)
        
        df['advanced_trend_score'] = (direction_score * 0.4 + strength_score * 0.3 + quality_score * 0.3).clip(0, 1)
        
    except Exception as e:
        logger.debug(f"Advanced Trend Calc Failed: {e}")
        df['advanced_trend_score'] = 0.5
    return df

def calc_advanced_momentum(df):
    """
    高级动量诊断
    1. RSI (14)
    2. Volume Weighted Momentum
    """
    df = df.copy()
    try:
        # 1. RSI
        delta = df['close'].diff()        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 2. VWM
        price_change = df['close'].pct_change(5)
        vol_change = df['volume'] / df['volume'].rolling(20).mean()
        # Log scale volume impact to dampen outliers
        df['vwm_5'] = price_change * np.log1p(vol_change.clip(0.5, 3)) 
        
        # Scoring
        # Prefer RSI rising from oversold or steady in mid-range, avoid overbought > 80
        rsi_norm = df['rsi_14'] / 100
        overbought_penalty = (df['rsi_14'] > 80).astype(float) * 0.5
        
        vwm_rank = df['vwm_5'].rank(pct=True)
        
        df['advanced_mom_score'] = (vwm_rank * 0.6 + rsi_norm * 0.4 - overbought_penalty).clip(0, 1)
        
    except Exception as e:
        logger.debug(f"Advanced Momentum Calc Failed: {e}")
        df['advanced_mom_score'] = 0.5
    return df

def calc_advanced_volume(df):
    """
    高级量能诊断
    1. OBV Slope
    2. Price-Volume Correlation
    """
    df = df.copy()
    try:
        # 1. OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_slope'] = df['obv'].diff(20)
        
        # 2. PV Correlation
        def rolling_corr_pv(window):
            if len(window) < 5: return 0
            return window['close'].corr(window['volume'])
                    # Using apply is slow, but acceptable for small datasets. 
        # For production, consider vectorized correlation if possible.
        df['pv_corr'] = df[['close', 'volume']].rolling(20).apply(
            lambda x: rolling_corr_pv(x), raw=False
        )
        
        # Scoring
        obv_score = df['obv_slope'].rank(pct=True)
        corr_mapped = (df['pv_corr'].clip(-1, 1) + 1) / 2 # Map -1..1 to 0..1
        
        df['advanced_vol_score'] = (obv_score * 0.5 + corr_mapped * 0.5).clip(0, 1)
        
    except Exception as e:
        logger.debug(f"Advanced Volume Calc Failed: {e}")
        df['advanced_vol_score'] = 0.5
    return df

# ================= 因子与评分层 =================
def calc_factors(df, idx_df=None, gold_df=None):
    """计算多因子指标（v5.0 融合高级因子）"""
    df = df.copy()
    if len(df) < 60 or "close" not in df.columns:
        return df
    
    # yfinance成交量修正（手转股）
    if "source" in df.columns and df["source"].iloc[-1] == "yf":
        df["volume"] = df["volume"] * 100
    
    # 基础指标计算
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["bias_60"] = (df["close"] - df["ma60"]) / df["ma60"]
    
    # --- 基础因子 (保留作为基准) ---
    trend_direction = (df["ma20"] > df["ma60"]).astype(float)
    df["base_trend_score"] = (trend_direction * (1 - abs(df["bias_60"].clip(-0.3, 0.3)) * 0.5)).fillna(0.5).clip(0, 1)
    
    mom_5 = df["close"].pct_change(5)
    mom_20 = df["close"].pct_change(20)
    mom_60 = df["close"].pct_change(60)
    momentum_raw = mom_5 * 0.4 + mom_20 * 0.35 + mom_60 * 0.25
    df["base_momentum_rank"] = momentum_raw.rolling(60).rank(pct=True).fillna(0.5)
    mom_abs_score = (momentum_raw.clip(-0.2, 0.2) + 0.2) / 0.4
    df["base_momentum_score"] = (df["base_momentum_rank"] * 0.6 + mom_abs_score * 0.4).clip(0, 1)
    
    volatility = df["ret"].rolling(20).std()
    df["base_volatility_score"] = 1 - volatility.rolling(60).rank(pct=True).fillna(0.5)
    
    df["base_volume_score"] = (df["volume"] / df["volume"].rolling(20).mean()).clip(0.3, 2.0).fillna(1.0)
        # --- 高级因子计算 ---
    df = calc_advanced_volatility(df)
    df = calc_advanced_trend(df)
    df = calc_advanced_momentum(df)
    df = calc_advanced_volume(df)
    
    # --- 融合因子 (Weighted Average) ---
    # 趋势: 60% 基础均线, 40% ADX/LinReg
    df["trend_score"] = df["base_trend_score"] * 0.6 + df["advanced_trend_score"] * 0.4
    
    # 动量: 50% 基础收益, 50% RSI/VWM
    df["momentum_score"] = df["base_momentum_score"] * 0.5 + df["advanced_mom_score"] * 0.5
    
    # 波动: 50% 基础Std, 50% ATR/Downside
    df["volatility_score"] = df["base_volatility_score"] * 0.5 + df["advanced_vol_score"] * 0.5
    
    # 量能: 50% 基础量比, 50% OBV/Corr
    df["volume_score"] = df["base_volume_score"] * 0.5 + df["advanced_vol_score"] * 0.5
    
    # 其他因子保持不变
    df["reversal_score"] = (-df["ret"].rolling(5).sum()).rolling(60).rank(pct=True).fillna(0.5)
    df["breakout_score"] = (df["close"] / df["close"].rolling(20).max()).rolling(60).rank(pct=True).fillna(0.5)
    vol_ratio = df["ret"].rolling(10).std() / (df["ret"].rolling(60).std() + 1e-9)
    df["vol_compress_score"] = 1 - vol_ratio.rolling(60).rank(pct=True).fillna(0.5)
    
    # 相对强弱因子
    if idx_df is not None and not idx_df.empty:
        idx = idx_df[['date', 'ret']].rename(columns={'ret': 'idx_ret'}).copy()
        idx["date"] = pd.to_datetime(idx["date"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df, idx, on='date', how='left')
        df['idx_ret'] = df['idx_ret'].ffill()
        df["rel_strength_score"] = (df['ret'] - df['idx_ret']).rolling(20).apply(
            lambda x: (x > 0).mean(), raw=True).fillna(0.5)
    else:
        df["rel_strength_score"] = 0.5
    
    # 黄金贝塔因子
    if gold_df is not None and not gold_df.empty:
        g = gold_df[['date', 'ret']].rename(columns={'ret': 'gold_ret'}).copy()
        g["date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df, g, on='date', how='left')
        df['gold_ret'] = df['gold_ret'].ffill()
        df["gold_beta"] = (df['ret'].rolling(60).cov(df['gold_ret']) / 
                          df['gold_ret'].rolling(60).var()).fillna(0)
        df["gold_beta_score"] = df["gold_beta"].rolling(60).rank(pct=True).fillna(0.5)
    else:
        df["gold_beta_score"] = 0.5
    
    return df
def calc_composite_scores(df, stock_name, role):
    """计算综合评分（v5.0 适配新因子）"""
    if len(df) < 60 or "close" not in df.columns:
        return df, None, {}
    close = df.iloc[-1].get("close", 0)
    bottom_info = {}
    if role == "弹性牌_信号灯":
        is_b, sl, det = identify_bottom_fractal(df)
        bottom_info = {"is_bottom": is_b, "stop_loss": sl, "details": det}
    if role == "避险矛_显微镜" and close < 35.0:
        df["composite_score"] = 0.15
        return df, 0.15, bottom_info
    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    if len(available_factors) < 3:
        df["composite_score"] = 0.5
        return df, 0.5, bottom_info
    weights = calc_ic_weights(df, available_factors, "ret", 60)
    if role == "弹性牌_信号灯" and bottom_info.get("is_bottom"):
        if "vol_compress_score" in weights:
            weights["vol_compress_score"] = min(weights["vol_compress_score"] * 2, 0.5)
    t = sum(weights.values()) + 1e-6
    weights = {k: v/t for k, v in weights.items()}
    score = sum(df[c].iloc[-1] * weights.get(c, 0) 
               for c in available_factors 
               if c in df.columns and pd.notna(df[c].iloc[-1]))
    score = max(0, min(1, score))
    if role == "弹性牌_信号灯" and bottom_info.get("is_bottom"):
        score += 0.2
    if "volume_score" in df.columns and pd.notna(df["volume_score"].iloc[-1]) and df["volume_score"].iloc[-1] < 0.7:
        score += 0.1
    df["composite_score"] = min(score, 1.0)
    return df, df["composite_score"].iloc[-1], bottom_info

# ================= 诊断与环境层 =================
def dimension_diagnosis(df, name, role, b_info=None):
    """多维度诊断分析（v5.0 增强描述）"""
    if len(df) < 5:
        return {}
    
    last = df.iloc[-1]
    close = last.get("close", 0)
    
    prev5 = df.iloc[-6:-1] if len(df) > 5 else df.iloc[:-1]
    vm = prev5["volume"].mean() if "volume" in prev5.columns else last.get("volume", 1)
    vr = last.get("volume", 0) / vm if vm > 0 else 1.0
    rt = last.get("ret", 0) or 0
    vs = last.get("volatility_score", 0.5)
    
    # 波动诊断
    if abs(rt) > 0.07:        vol_label = "剧烈上涨" if rt > 0 else "剧烈下跌"
    elif abs(rt) > 0.04:
        vol_label = "大幅上涨" if rt > 0 else "大幅下跌"
    elif abs(rt) > 0.02:
        vol_label = "显著上涨" if rt > 0 else "显著下跌"
    elif vs < 0.4:
        vol_label = "高波偏强" if rt > 0 else ("高波偏弱" if rt < 0 else "高波震荡")
    elif vs > 0.6:
        vol_label = "极致平稳" if abs(rt) < 0.005 else "小幅波动"
    else:
        vol_label = "温和上涨" if rt > 0.01 else ("温和下跌" if rt < -0.01 else "横盘整理")
    
    # 动量诊断
    mom_rank = last.get("momentum_score", 0.5) # Using composite momentum score
    mom_value = last.get("base_momentum_score", 0) # Raw value proxy
    
    if mom_rank > 0.75 and mom_value > 0.5:
        mom_label = "强势上涨"
    elif mom_rank > 0.75:
        mom_label = "稳步上涨"
    elif mom_rank > 0.5 and mom_value > 0.5:
        mom_label = "温和上涨"
    elif mom_rank > 0.5:
        mom_label = "企稳回升"
    elif mom_rank > 0.25 and mom_value > 0.5:
        mom_label = "弱势反弹"
    elif mom_rank > 0.25:
        mom_label = "动量减弱"
    elif mom_value < 0.4:
        mom_label = "加速下跌"
    else:
        mom_label = "持续走弱"
    
    # 趋势诊断
    ma20 = last.get("ma20", 0)
    ma60 = last.get("ma60", 0)
    bias = last.get("bias_60", 0)
    adx = last.get("adx_14", 0)
    
    if ma20 > ma60:
        if bias > 0.15:
            trend_label = "多头强势"
        elif bias > 0.05:
            trend_label = "多头排列"
        else:
            trend_label = "多头初期"
    elif ma20 < ma60:
        if bias < -0.15:
            trend_label = "空头强势"
        elif bias < -0.05:            trend_label = "空头排列"
        else:
            trend_label = "空头尾声"
    else:
        trend_label = "均线粘合"
        
    # Add ADX context
    if adx > 25:
        trend_label += "(强趋)"
    elif adx < 20:
        trend_label += "(震荡)"

    # 量能
    if vr > 1.5:
        vol_label2 = f"放量({vr:.1f}x)"
    elif vr > 0.7:
        vol_label2 = f"正常({vr:.1f}x)"
    else:
        vol_label2 = f"缩量({vr:.1f}x)"
    
    diag = {
        "标的": name,
        "收盘价": f"{close:.2f}",
        "涨跌幅": f"{rt:+.2%}",
        "量比": f"{vr:.1f}x",
        "趋势": trend_label,
        "动量": mom_label,
        "波动": vol_label,
        "量能": vol_label2
    }
    
    sc = None
    
    # 角色特定诊断
    if role == "压舱石_望远镜":
        bias = last.get("bias_60", 0)
        if bias > 0.2:
            diag["乖离率"] = "高位"
        elif bias < -0.15:
            diag["乖离率"] = "低位"
        else:
            diag["乖离率"] = "正常"
        
        rs = last.get("rel_strength_score", 0.5)
        if rs > 0.6:
            diag["大盘联动"] = "强于大盘"
        elif rs > 0.4:
            diag["大盘联动"] = "同步大盘"
        else:
            diag["大盘联动"] = "弱于大盘"        
        diag["距止损"] = f"{(close-26)/26*100:.1f}%"
        
        if close >= 28.5:
            sc = "站稳28.5加仓"
        elif close >= 27:
            sc = "持有底仓"
        else:
            sc = "破26重估"
    
    elif role == "避险矛_显微镜":
        beta = last.get("gold_beta", 0)
        if beta > 1.2:
            diag["金价联动"] = f"高弹性({beta:.1f})"
        elif beta > 0.8:
            diag["金价联动"] = f"正常({beta:.1f})"
        else:
            diag["金价联动"] = "低弹性/失效"
        
        diag["距止损"] = f"{(close-35)/35*100:.1f}%"
        
        if close >= 37:
            sc = "安全区"
        elif close >= 35:
            sc = "减仓预警"
        else:
            sc = "清仓危险"
    
    elif role == "弹性牌_信号灯":
        if b_info and b_info.get("details"):
            d = b_info["details"]
            if b_info.get("is_bottom"):
                diag["底分型"] = "已确认"
            elif d.get("底分型形态") == "是":
                diag["底分型"] = "待确认"
            else:
                diag["底分型"] = "未形成"
            
            diag["止损参考"] = d.get("止损参考价", "N/A")
            if d.get("止损参考价"):
                diag["距止损"] = f"{(close-d['止损参考价'])/d['止损参考价']*100:.1f}%"
            else:
                diag["距止损"] = "N/A"
            
            if close >= 40:
                diag["40元关口"] = "已突破"
            elif close >= 37:
                diag["40元关口"] = "测试中"
            else:
                diag["40元关口"] = "弱势"        else:
            diag["底分型"] = "未形成"
            diag["止损参考"] = "N/A"
            diag["距止损"] = "N/A"
            diag["40元关口"] = "N/A"
        
        if b_info and b_info.get("is_bottom"):
            sc = f"底分型确认，止损{diag.get('止损参考', 'N/A')}"
        elif vr < 0.7 and close <= 37:
            sc = "缩量回踩"
        else:
            sc = "等信号"
    
    diag["策略结论"] = sc
    return diag

def calc_market_environment(scores, diags, turnover=None):
    """计算市场环境"""
    if not scores:
        return {
            "环境评级": "数据缺失",
            "综合评分": 0,
            "标的平均评分": 0,
            "市场活跃度": 0,
            "成交额描述": "缺失",
            "建议": "等待"
        }
    
    avg = np.mean(list(scores.values()))
    ma, td = 0.5, "缺失"
    
    if turnover and turnover > 0:
        if turnover >= 15000:
            ma, td = 1.0, f"沪市{turnover:.0f}亿，极活跃"
        elif turnover >= 10000:
            ma, td = 0.8, f"沪市{turnover:.0f}亿，活跃"
        elif turnover >= 7500:
            ma, td = 0.6, f"沪市{turnover:.0f}亿，偏暖"
        elif turnover >= 5000:
            ma, td = 0.4, f"沪市{turnover:.0f}亿，偏冷"
        else:
            ma, td = 0.2, f"沪市{turnover:.0f}亿，极冷"
    
    fs = avg * 0.5 + ma * 0.5
    
    if fs >= 0.75:
        env, adv = "牛市级", "确认信号后可积极"
    elif fs >= 0.60:
        env, adv = "强势震荡", "市场偏暖，留意加仓"
    elif fs >= 0.45:        env, adv = "中性整理", "严格按信号操作"
    elif fs >= 0.30:
        env, adv = "弱势承压", "注意防守"
    else:
        env, adv = "防御模式", "优先保护本金"
    
    return {
        "环境评级": env,
        "综合评分": round(fs, 3),
        "标的平均评分": round(avg, 3),
        "市场活跃度": round(ma, 3),
        "成交额描述": td,
        "建议": adv
    }

# ================= 主控函数 =================
def self_check():
    """系统自检"""
    checks = []
    
    try:
        checks.append(f"akshare版本: {ak.__version__}")
    except:
        checks.append("akshare: 已安装")
    
    for pkg_name in ["yfinance", "numpy", "pandas", "openpyxl", "scipy"]:
        try:
            __import__(pkg_name)
            checks.append(f"{pkg_name}: 已安装")
        except:
            checks.append(f"{pkg_name}: 未安装")
    
    core_functions = [
        validate_data_freshness, normalize_turnover, calc_ic_weights,
        identify_bottom_fractal, get_stock_data, get_index_data,
        get_gold_data, calc_factors, calc_composite_scores,
        dimension_diagnosis, calc_market_environment,
        calc_advanced_volatility, calc_advanced_trend, calc_advanced_momentum, calc_advanced_volume
    ]
    for fn in core_functions:
        checks.append(f"函数 {fn.__name__}: 已定义")
    
    return checks

def run():
    """主运行函数"""
    separator = "=" * 60
    print(f"\n{separator}")
    print("  市场环境监控仪表盘 v5.1 (Advanced Factors)")
    print(f"  运行时间: {beijing_now.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)")    print(f"{separator}")
    
    # 自检
    logger.info("系统自检...")
    for c in self_check():
        logger.info(f"  [自检] {c}")
    print()
    
    # 获取市场数据
    logger.info("获取市场数据...")
    idx_df, turnover = get_index_data()
    gold_df = get_gold_data()
    
    # 监控标的
    scores, diags, rows = {}, [], []
    
    for name, info in monitor_stocks.items():
        logger.info(f"[监控] {name} ({info['code']}) - {info['role']}")
        
        df = get_stock_data(info["code"])
        if df.empty:
            logger.warning(f"  [失败] {name} 数据获取失败")
            continue
        
        df = calc_factors(df, idx_df, gold_df)
        df, sc, b_info = calc_composite_scores(df, name, info["role"])
        
        if sc is None:
            sc = 0.5
        
        scores[name] = sc
        diag = dimension_diagnosis(df, name, info["role"], b_info)
        diags.append(diag)
        
        if sc >= 0.75:
            lvl = "[强势]"
        elif sc >= 0.55:
            lvl = "[偏强]"
        elif sc >= 0.35:
            lvl = "[偏弱]"
        else:
            lvl = "[弱势]"
        
        logger.info(f"  评分: {sc:.3f} {lvl} | 策略: {diag.get('策略结论', 'N/A')}")
        logger.info(f"  涨跌: {diag.get('涨跌幅', 'N/A')} | 波动: {diag.get('波动', 'N/A')} | 动量: {diag.get('动量', 'N/A')}")
        
        row = {"标的": name, "角色": info["role"], "综合评分": round(sc, 3), **diag}
        rows.append(row)
    
    # 市场环境    env = calc_market_environment(scores, diags, turnover)
    
    # 仓位建议
    pos_map = {0.75: "60-80%", 0.60: "40-60%", 0.45: "20-40%", 0.30: "10-20%"}
    total_pos = next((v for k, v in pos_map.items() if env["综合评分"] >= k), "10%以下")
    env["总仓位"] = total_pos
    
    try:
        low = int(total_pos.split('-')[0].replace('%', '').replace('以下', '').strip())
    except:
        low = 10
    
    for r in rows:
        s = max(scores.get(r["标的"], 0.5), 0.15)
        if s >= 0.7:
            adj = 1.2
        elif s < 0.4:
            adj = 0.7
        else:
            adj = 1.0
        
        weight = monitor_stocks[r['标的']]['weight']
        r["建议仓位"] = f"{low * weight * adj:.0f}%"
        r["评分调整"] = f"{adj:.1f}x"
    
    # 输出汇总
    print(f"\n{separator}")
    print(f"环境: {env['环境评级']} | 综合评分: {env['综合评分']:.3f} | 活跃度: {env['市场活跃度']:.3f}")
    print(f"成交额: {env['成交额描述']} | 总仓位: {total_pos}")
    
    for r in rows:
        print(f"  {r['标的']}: {r['建议仓位']} | 策略: {r['策略结论']}")
        print(f"    趋势: {r.get('趋势', 'N/A')} | 动量: {r.get('动量', 'N/A')} | 波动: {r.get('波动', 'N/A')}")
    
    print(f"建议: {env['建议']}")
    print(f"{separator}\n")
    
    # 导出Excel
    try:
        df_out = pd.DataFrame(rows)
        df_out["日期"] = TODAY
        df_out["总仓位"] = total_pos
        df_out["环境"] = env["环境评级"]
        df_out["市场评分"] = env["综合评分"]
        df_out["活跃度"] = env["市场活跃度"]
        df_out["成交额"] = env["成交额描述"]
        
        # GitHub Actions使用workspace路径
        if is_github:
            workspace = os.environ.get('GITHUB_WORKSPACE', '.')            fpath = os.path.join(workspace, f"市场环境监控_{TODAY}.xlsx")
        else:
            fpath = f"市场环境监控_{TODAY}.xlsx"
        
        df_out.to_excel(fpath, index=False)
        logger.info(f"[完成] 已导出: {fpath}")
    except Exception as e:
        logger.error(f"导出失败: {e}")
    
    return {"scores": scores, "env": env, "diagnosis": {d["标的"]: d for d in diags}}

# ================= 入口 =================
if __name__ == "__main__":
    try:
        result = run()
        
        if is_github:
            env = result['env']
            scores = result['scores']
            
            print("\n" + "="*60)
            print("  GitHub Actions 运行完成")
            print("="*60)
            print(f"  环境评级: {env['环境评级']}")
            print(f"  综合评分: {env['综合评分']}")
            print(f"  市场活跃度: {env['市场活跃度']}")
            print(f"  总仓位建议: {env.get('总仓位', 'N/A')}")
            print(f"  操作建议: {env['建议']}")
            print()
            print("  标的评分:")
            for name, score in scores.items():
                print(f"    {name}: {score:.3f}")
            print("="*60)
            
            # 查找Excel文件确认
            excel_files = glob.glob("市场环境监控_*.xlsx")
            if excel_files:
                print(f"  报告文件: {excel_files[0]}")
            else:
                print("  警告: 未找到报告文件")
        else:
            print("\n按回车退出...")
            input()
            
    except KeyboardInterrupt:
        print("\n[中断] 用户手动中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"运行异常: {e}", exc_info=True)
        if is_github:            import traceback
            traceback.print_exc()
        sys.exit(1)
