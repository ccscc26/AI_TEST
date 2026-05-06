# market_monitor.py
# 市场环境监控仪表盘 - 工程化校准版
# 改动：数据freshness校验 + IC动态加权 + 结构化输出 + 日志

import subprocess
import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

for pkg in ["akshare", "pandas", "numpy", "openpyxl"]:
    try:
        __import__(pkg)
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import akshare as ak

monitor_stocks = {
    "中信证券": {"code": "600030", "role": "压舱石_望远镜"},
    "赤峰黄金": {"code": "600988", "role": "避险矛_显微镜"},
    "西藏矿业": {"code": "000762", "role": "弹性牌_信号灯"}
}

# 通用因子列表
FACTOR_COLS = ["trend_score", "momentum_score", "volatility_score", "volume_score",
               "rel_strength_score", "gold_beta_score", "reversal_score", "breakout_score", "vol_compress_score"]

# =========================
# P0修复1：数据freshness校验
# =========================
def validate_data_freshness(df, max_delay_days=3):
    """校验最新数据是否在max_delay_days天内"""
    try:
        last_date = pd.to_datetime(df["date"].iloc[-1])
        delta = (datetime.now() - last_date).days
        return delta <= max_delay_days, delta
    except:
        return False, -1

# =========================
# P0修复2：成交额单位标准化
# =========================
def normalize_turnover(x):
    """统一成交额单位为亿元"""
    if pd.isna(x) or x <= 0:
        return None
    if x > 1e11:
        return x / 1e8
    elif x > 1e8:
        return x / 1e4
    return x

# =========================
# P0修复3：IC加权计算
# =========================
def calc_ic_weights(df, factor_cols, target_col="ret", lookback=60):
    """计算各因子与未来收益的IC，返回绝对值归一化权重"""
    ic_dict = {}
    for col in factor_cols:
        if col not in df.columns:
            continue
        # 用最近lookback天的数据计算IC
        recent = df.tail(lookback)
        valid = recent[[col, target_col]].dropna()
        if len(valid) < 20:
            ic_dict[col] = 0
            continue
        ic = valid[col].corr(valid[target_col].shift(-1))
        ic_dict[col] = ic if not np.isnan(ic) else 0
    
    total = sum(abs(v) for v in ic_dict.values()) + 1e-6
    weights = {k: abs(v)/total for k, v in ic_dict.items()}
    logger.info(f"IC权重: { {k: round(v,3) for k,v in weights.items()} }")
    return weights


def identify_bottom_fractal(df):
    df = df.copy()
    if len(df) < 5:
        return False, None, {}

    recent = df.tail(5)
    p2 = recent.iloc[-3]
    p1 = recent.iloc[-2]
    p0 = recent.iloc[-1]

    # 趋势过滤：下降趋势中找底分型
    if "ma20" in df.columns and "ma60" in df.columns and len(df) >= 5:
        is_downtrend = df["ma20"].iloc[-3] < df["ma60"].iloc[-3]
        if not is_downtrend:
            return False, None, {"底分型形态": "❌(非下降趋势)", "止损参考价": None, "今日量比": 1.0}

    is_bottom_shape = (
        (p1['low'] < p2['low']) and (p1['low'] < p0['low']) and
        (p1['high'] < p2['high']) and (p1['high'] < p0['high'])
    )
    is_confirmed = p0['close'] > p1['high']

    vol_min_10 = df['volume'].rolling(10).min().iloc[-1]
    vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
    vol_today = df['volume'].iloc[-1]

    is_low_volume_strict = p1['volume'] <= vol_min_10 * 1.2
    is_low_volume_normal = p1['volume'] < vol_ma5 * 0.8
    vol_ratio = vol_today / vol_ma5 if vol_ma5 > 0 else 1.0

    p0_body = abs(p0['close'] - p0['open'])
    p0_range = p0['high'] - p0['low'] if p0['high'] > p0['low'] else 0.01
    is_solid_candle = p0_body / p0_range > 0.3

    details = {
        "底分型形态": "✅" if is_bottom_shape else "❌",
        "右侧确认": "✅" if is_confirmed else "❌",
        "严格缩量": "✅" if is_low_volume_strict else "❌",
        "普通缩量": "✅" if is_low_volume_normal else "❌",
        "实体确认": "✅" if is_solid_candle else "❌",
        "止损参考价": round(p1['low'], 2) if is_bottom_shape else None,
        "今日量比": round(vol_ratio, 2)
    }

    strict_result = is_bottom_shape and is_confirmed and is_low_volume_strict and is_solid_candle
    loose_result = is_bottom_shape and is_confirmed and is_low_volume_normal

    if strict_result:
        return True, p1['low'], details
    elif loose_result:
        return True, p1['low'], details
    else:
        return False, None, details


def get_stock_data(code, tail_size=500, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is not None and len(df) > 0:
                df = df.tail(tail_size).copy()
                df.rename(columns={
                    "收盘": "close", "开盘": "open", "最高": "high",
                    "最低": "low", "成交量": "volume", "日期": "date"
                }, inplace=True)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["ret"] = df["close"].pct_change()
                df = df.dropna(subset=["close"])
                fresh, delta = validate_data_freshness(df)
                if not fresh:
                    logger.warning(f"{code} 数据可能过期: {delta}天")
                print(f"  [OK] 数据源1 成功获取 {code}，共{len(df)}条，最新日期{df['date'].iloc[-1]}")
                return df
        except:
            if attempt < max_retries - 1:
                time.sleep(2)

    try:
        full_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
        df = ak.stock_zh_a_daily(symbol=full_code, adjust="qfq")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"close": "close", "volume": "volume", "date": "date"}, inplace=True)
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
            df = df.dropna(subset=["close"])
            df["ret"] = df["close"].pct_change()
            if "date" not in df.columns:
                df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq='B').strftime("%Y-%m-%d")
            else:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            fresh, delta = validate_data_freshness(df)
            if not fresh:
                logger.warning(f"{code} 数据可能过期: {delta}天")
            print(f"  [OK] 数据源2 成功获取 {code}，共{len(df)}条，最新日期{df['date'].iloc[-1]}")
            return df
    except:
        pass

    logger.error(f"{code} 所有数据源均失败")
    return pd.DataFrame()


def get_index_data(tail_size=500):
    df_result = pd.DataFrame()
    turnover = None

    try:
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            date_col = next((c for c in df.columns if 'date' in c.lower() or '日期' in c), df.columns[0])
            close_col = next((c for c in df.columns if 'close' in c.lower() or '收盘' in c), None)
            amount_col = next((c for c in df.columns if 'amount' in c.lower() or '成交额' in c), None)
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            if close_col:
                df["close"] = pd.to_numeric(df[close_col], errors='coerce')
                df["ret"] = df["close"].pct_change()
            if amount_col:
                raw_amount = pd.to_numeric(df[amount_col].iloc[-1], errors='coerce')
                turnover = normalize_turnover(raw_amount)
            df_result = df
            print(f"  [OK] 上证指数数据获取成功（数据源1），共{len(df)}条", end="")
            if turnover is not None and turnover > 0:
                print(f"，成交额{turnover:.0f}亿")
                return df_result, turnover
            else:
                print("，成交额未提取到，尝试其他方式...")
    except Exception as e:
        logger.warning(f"上证指数数据源1失败: {e}")

    try:
        df = ak.stock_zh_index_daily_tx(symbol="sh000001")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"收盘": "close", "日期": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["ret"] = df["close"].pct_change()
            if df_result.empty:
                df_result = df
            print(f"  [OK] 上证指数数据获取成功（数据源2），共{len(df)}条")
    except Exception as e:
        logger.warning(f"上证指数数据源2失败: {e}")

    if turnover is None or turnover == 0:
        for attempt in range(3):
            try:
                amount_df = ak.stock_zh_index_daily_em(symbol="sh000001")
                if amount_df is not None and len(amount_df) > 0:
                    for col in amount_df.columns:
                        col_lower = col.lower()
                        if 'amount' in col_lower:
                            raw_amount = pd.to_numeric(amount_df[col].iloc[-1], errors='coerce')
                            turnover = normalize_turnover(raw_amount)
                            if turnover is not None and turnover > 0:
                                print(f"  [OK] 兜底方案获取成交额: {turnover:.0f}亿 (第{attempt+1}次)")
                                break
                    if turnover is not None and turnover > 0:
                        break
            except:
                if attempt < 2:
                    time.sleep(2)

    if (turnover is None or turnover == 0) and not df_result.empty:
        try:
            for col in df_result.columns:
                if 'amount' in col.lower():
                    raw = pd.to_numeric(df_result[col].iloc[-1], errors='coerce')
                    turnover = normalize_turnover(raw)
                    if turnover is not None and turnover > 0:
                        print(f"  [OK] 兜底方案2提取成交额: {turnover:.0f}亿")
                        break
        except:
            pass

    if turnover is not None and turnover > 0:
        print(f"  [INFO] 最终成交额: {turnover:.0f}亿")
    else:
        logger.warning("成交额未获取到，市场活跃度将使用默认值0.5")

    if not df_result.empty:
        return df_result, turnover
    else:
        logger.error("上证指数所有数据源均失败")
        return pd.DataFrame(), None


def get_gold_data(tail_size=500):
    try:
        df = ak.futures_main_sina(symbol="AU0")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            date_col = next((c for c in df.columns if 'date' in c.lower() or '日期' in c), df.columns[0])
            close_col = next((c for c in df.columns if 'close' in c.lower() or '收盘' in c), None)
            if close_col is None:
                close_col = next((c for c in df.columns if '价' in c), df.columns[-1])
            df.rename(columns={date_col: "date", close_col: "close"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["ret"] = df["close"].pct_change()
            df = df.dropna(subset=["close"])
            print(f"  [OK] 沪金期货数据获取成功（数据源1），共{len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"沪金期货数据源1失败: {e}")

    try:
        df = ak.spot_gold()
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"价格": "close", "日期": "date"}, inplace=True)
            if "date" not in df.columns:
                df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq='B').strftime("%Y-%m-%d")
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["ret"] = df["close"].pct_change()
            df = df.dropna(subset=["close"])
            print(f"  [OK] 沪金数据获取成功（数据源2），共{len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"沪金数据源2失败: {e}")

    logger.error("沪金所有数据源均失败")
    return pd.DataFrame()


def calc_factors(df, idx_df=None, gold_df=None):
    df = df.copy()
    if len(df) < 60 or "close" not in df.columns:
        return df

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    trend_direction = (df["ma20"] > df["ma60"]).astype(float).fillna(0.5)
    df["bias_60"] = (df["close"] - df["ma60"]) / df["ma60"]
    bias_penalty = 1 - abs(df["bias_60"].clip(-0.3, 0.3)) * 0.5
    df["trend_score"] = trend_direction * bias_penalty
    df["trend_score"] = df["trend_score"].fillna(0.5).clip(0, 1)

    mom = df["close"].pct_change(5) * 0.6 + df["close"].pct_change(20) * 0.4
    df["momentum_score"] = mom.rolling(60).rank(pct=True).fillna(0.5)

    vol = df["ret"].rolling(20).std()
    df["volatility_score"] = 1 - vol.rolling(60).rank(pct=True).fillna(0.5)

    df["volume_score"] = (df["volume"] > df["volume"].rolling(20).mean()).astype(float).fillna(0.5)

    # Alpha因子1: 反转因子
    df["reversal"] = -df["ret"].rolling(5).sum()
    df["reversal_score"] = df["reversal"].rolling(60).rank(pct=True).fillna(0.5)

    # Alpha因子2: 突破因子
    df["breakout"] = df["close"] / df["close"].rolling(20).max()
    df["breakout_score"] = df["breakout"].rolling(60).rank(pct=True).fillna(0.5)

    # Alpha因子3: 波动收缩因子
    df["vol_compress"] = df["ret"].rolling(10).std() / (df["ret"].rolling(60).std() + 1e-9)
    df["vol_compress_score"] = 1 - df["vol_compress"].rolling(60).rank(pct=True).fillna(0.5)

    if idx_df is not None and not idx_df.empty:
        idx_sub = idx_df[['date', 'ret']].rename(columns={'ret': 'idx_ret'}).copy()
        idx_sub["date"] = pd.to_datetime(idx_sub["date"]).dt.strftime("%Y-%m-%d")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df, idx_sub, on='date', how='left')
        df['idx_ret'] = df['idx_ret'].ffill()

    if gold_df is not None and not gold_df.empty:
        gold_sub = gold_df[['date', 'ret']].rename(columns={'ret': 'gold_ret'}).copy()
        gold_sub["date"] = pd.to_datetime(gold_sub["date"]).dt.strftime("%Y-%m-%d")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df, gold_sub, on='date', how='left')
        df['gold_ret'] = df['gold_ret'].ffill()

    if 'idx_ret' in df.columns:
        rel_diff = df['ret'] - df['idx_ret']
        df["rel_strength_score"] = rel_diff.rolling(20).apply(lambda x: (x > 0).mean(), raw=True).fillna(0.5)
    else:
        df["rel_strength_score"] = 0.5

    if 'gold_ret' in df.columns:
        rolling_cov = df['ret'].rolling(60).cov(df['gold_ret'])
        rolling_var = df['gold_ret'].rolling(60).var()
        df["gold_beta"] = (rolling_cov / rolling_var).fillna(0)
        df["gold_beta_score"] = df["gold_beta"].rolling(60).rank(pct=True).fillna(0.5)
    else:
        df["gold_beta_score"] = 0.5

    return df


def calc_composite_scores(df, stock_name, role):
    if len(df) < 60 or "close" not in df.columns:
        return df, None, {}

    last_row = df.iloc[-1]
    close = last_row.get("close", 0)

    bottom_info = {}
    if role == "弹性牌_信号灯":
        is_bottom, stop_loss, bottom_details = identify_bottom_fractal(df)
        bottom_info = {"is_bottom": is_bottom, "stop_loss": stop_loss, "details": bottom_details}

    if role == "避险矛_显微镜" and close < 35.0:
        df["composite_score"] = 0.15
        return df, 0.15, bottom_info

    # IC动态加权
    weights = calc_ic_weights(df, FACTOR_COLS, "ret", lookback=60)

    score = 0
    for col in FACTOR_COLS:
        if col in df.columns and col in weights:
            raw_val = df[col].iloc[-1]
            if pd.notna(raw_val):
                score += raw_val * weights[col]

    score = max(0, min(1, score))

    # 底分型加分
    if role == "弹性牌_信号灯" and bottom_info.get("is_bottom"):
        score += 0.2
        vol_today = df["volume_score"].iloc[-1]
        if pd.notna(vol_today) and vol_today < 0.4:
            score += 0.1
        score = min(score, 1.0)

    df["composite_score"] = score
    return df, score, bottom_info


def dimension_diagnosis(df, stock_name, role, bottom_info=None):
    if len(df) < 5:
        return {}

    last = df.iloc[-1]
    prev_5 = df.iloc[-6:-1] if len(df) > 5 else df.iloc[:-1]
    prev_20 = df.iloc[-21:-1] if len(df) > 20 else df.iloc[:-1]

    trend = "向上" if last.get("trend_score", 0.5) > 0.7 else "震荡" if last.get("trend_score", 0.5) > 0.4 else "偏弱"
    momentum = "强劲" if last.get("momentum_score", 0.5) > 0.7 else "温和" if last.get("momentum_score", 0.5) > 0.4 else "衰减"

    vol_today = last.get("volume", 0)
    vol_ma5_val = prev_5["volume"].mean() if len(prev_5) > 0 and "volume" in prev_5.columns else vol_today
    vol_ratio = vol_today / vol_ma5_val if vol_ma5_val > 0 else 1.0
    if vol_ratio > 1.5:
        volume_label = f"放量({vol_ratio:.1f}x)"
    elif vol_ratio > 0.7:
        volume_label = f"正常({vol_ratio:.1f}x)"
    else:
        volume_label = f"缩量({vol_ratio:.1f}x)"

    ret_today = last.get("ret", 0)
    if pd.isna(ret_today):
        ret_today = 0
    ret_display = f"{ret_today:+.2%}"

    vol_score = last.get("volatility_score", 0.5)
    if abs(ret_today) > 0.07:
        volatility = "剧烈波动"
    elif abs(ret_today) > 0.04:
        volatility = "大幅上涨" if ret_today > 0 else "大幅下跌"
    elif vol_score < 0.4 and ret_today > 0:
        volatility = "强势波动"
    elif vol_score < 0.4 and ret_today < 0:
        volatility = "异常波动"
    elif vol_score > 0.6:
        volatility = "平稳"
    else:
        volatility = "正常"

    diag = {
        "标的": stock_name,
        "收盘价": f"{last.get('close', 0):.2f}",
        "涨跌幅": ret_display,
        "量比": f"{vol_ratio:.1f}x",
        "趋势": trend,
        "动量": momentum,
        "波动": volatility,
        "量能": volume_label,
    }

    strategy_conclusion = None

    if role == "压舱石_望远镜":
        bias = last.get("bias_60", 0)
        if pd.notna(bias):
            diag["乖离率"] = "高位偏离" if bias > 0.2 else ("低位偏离" if bias < -0.15 else "正常区间")
        rel_score = last.get("rel_strength_score", 0.5)
        if pd.isna(rel_score) or 'idx_ret' not in df.columns:
            diag["大盘联动"] = "数据缺失"
        elif rel_score > 0.6:
            diag["大盘联动"] = "强于大盘"
        elif rel_score > 0.4:
            diag["大盘联动"] = "同步大盘"
        else:
            diag["大盘联动"] = "弱于大盘"
        diag["市成交额"] = f"{vol_today * last.get('close', 0) / 1e8:.0f}亿" if vol_today > 0 else "N/A"

        if close >= 28.50:
            strategy_conclusion = "✅ 加仓信号触发：站稳28.50元"
        elif close >= 27.00:
            strategy_conclusion = "⏳ 27-28.50元，继续持有底仓"
        elif close >= 26.00:
            strategy_conclusion = "⏳ 26-27元，静默契约持有"
        else:
            strategy_conclusion = "⚠️ 跌破26元，重新评估"

    elif role == "避险矛_显微镜":
        beta = last.get("gold_beta", 0)
        if pd.isna(beta) or beta == 0:
            diag["金价联动"] = "数据缺失"
        elif beta > 1.2:
            diag["金价联动"] = f"高弹性({beta:.1f})"
        elif beta > 0.8:
            diag["金价联动"] = f"正常弹性({beta:.1f})"
        elif beta > 0.3:
            diag["金价联动"] = f"低弹性({beta:.1f})"
        else:
            diag["金价联动"] = "弹性失效"

        close = last.get("close", 0)
        if close >= 37:
            diag["预警线"] = "37元以上，安全"
            strategy_conclusion = "✅ 安全区，等5.20压测结论"
        elif close >= 35:
            diag["预警线"] = "37元下方，预警"
            strategy_conclusion = "⚠️ 减仓预警"
        else:
            diag["预警线"] = "35元下方，清仓危险"
            strategy_conclusion = "🔴 清仓危险"

    elif role == "弹性牌_信号灯":
        if bottom_info and bottom_info.get("details"):
            detail = bottom_info["details"]
            diag["底分型"] = "✅ 已确认" if bottom_info.get("is_bottom") else ("⏳ 待缩量确认" if detail.get("底分型形态") == "✅" and detail.get("右侧确认") == "✅" else "❌ 未形成")
            diag["止损参考"] = detail.get("止损参考价", "N/A")
        else:
            diag["底分型"] = "❌ 未形成"
            diag["止损参考"] = "N/A"
        diag["缩量"] = f"{vol_ratio:.1%}"

        close = last.get("close", 0)
        diag["40元关口"] = "已突破" if close >= 40 else ("正在测试" if close >= 39 else ("距关口较远" if close >= 37 else "弱势"))

        is_bottom = bottom_info.get("is_bottom", False)
        if is_bottom:
            strategy_conclusion = f"✅ 底分型确认，止损{diag.get('止损参考','N/A')}"
        elif vol_ratio < 0.7 and close <= 37:
            strategy_conclusion = "⏳ 缩量回踩中"
        elif vol_ratio >= 1.0:
            strategy_conclusion = "⏳ 量能未缩，继续等"
        else:
            strategy_conclusion = "⏳ 等待底分型信号"

    diag["策略结论"] = strategy_conclusion
    return diag


def calc_market_environment(scores_dict, diagnoses, turnover=None):
    if not scores_dict:
        return {
            "环境评级": "数据缺失", "综合评分": 0.0,
            "标的平均评分": 0.0, "市场活跃度": 0.0,
            "成交额描述": "数据缺失", "建议": "等待数据"
        }

    avg_score = np.mean(list(scores_dict.values()))
    market_activity_score = 0.5
    turnover_desc = "数据缺失"

    if turnover is not None and turnover > 0:
        if turnover >= 15000:
            market_activity_score = 1.0
            turnover_desc = f"沪市{turnover:.0f}亿，极为活跃"
        elif turnover >= 12000:
            market_activity_score = 0.9
            turnover_desc = f"沪市{turnover:.0f}亿，高度活跃"
        elif turnover >= 10000:
            market_activity_score = 0.8
            turnover_desc = f"沪市{turnover:.0f}亿，活跃"
        elif turnover >= 7500:
            market_activity_score = 0.6
            turnover_desc = f"沪市{turnover:.0f}亿，正常偏暖"
        elif turnover >= 5000:
            market_activity_score = 0.4
            turnover_desc = f"沪市{turnover:.0f}亿，偏冷"
        else:
            market_activity_score = 0.2
            turnover_desc = f"沪市{turnover:.0f}亿，极冷"

    final_score = avg_score * 0.5 + market_activity_score * 0.5

    if final_score >= 0.75:
        env, advice = "牛市级", "确认信号后可适当积极"
    elif final_score >= 0.60:
        env, advice = "强势震荡", "市场偏暖，留意加仓信号"
    elif final_score >= 0.45:
        env, advice = "中性整理", "严格按信号操作"
    elif final_score >= 0.30:
        env, advice = "弱势承压", "注意防守"
    else:
        env, advice = "防御模式", "优先保护本金"

    return {
        "环境评级": env,
        "综合评分": round(final_score, 3),
        "标的平均评分": round(avg_score, 3),
        "市场活跃度": round(market_activity_score, 3),
        "成交额描述": turnover_desc,
        "建议": advice
    }


def run():
    print("\n" + "="*60)
    print("  市场环境监控仪表盘（工程化校准版）")
    print("="*60 + "\n")

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"开始运行监控，日期: {today}")

    idx_df, market_turnover = get_index_data()
    gold_df = get_gold_data()

    scores = {}
    diagnoses = []
    summary_rows = []

    for name, info in monitor_stocks.items():
        code = info["code"]
        role = info["role"]

        print(f"[监控] {name} ({code}) - {role}")
        logger.info(f"监控 {name}")

        df = get_stock_data(code, tail_size=500)
        if df.empty:
            logger.error(f"{name} 数据获取失败")
            continue

        print(f"  > 数据量: {len(df)} 条，收盘: {df['close'].iloc[-1]:.2f}")

        df = calc_factors(df, idx_df, gold_df)
        df, final_score, bottom_info = calc_composite_scores(df, name, role)
        if final_score is None:
            final_score = 0.5
        scores[name] = final_score

        diag = dimension_diagnosis(df, name, role, bottom_info)
        diagnoses.append(diag)

        level = "🟢" if final_score >= 0.75 else ("🟡" if final_score >= 0.55 else ("🟠" if final_score >= 0.35 else "🔴"))
        print(f"  > 评分: {final_score:.3f} {level} | 策略: {diag.get('策略结论','N/A')}")
        print(f"  > 涨跌幅:{diag.get('涨跌幅','N/A')} | 量比:{diag.get('量比','N/A')} | 波动:{diag.get('波动','N/A')}\n")

        row = {"标的": name, "角色": role, "综合评分": round(final_score, 3),
               "涨跌幅": diag.get("涨跌幅","N/A"), "量比": diag.get("量比","N/A"),
               "趋势": diag.get("趋势","N/A"), "动量": diag.get("动量","N/A"),
               "波动": diag.get("波动","N/A"), "量能": diag.get("量能","N/A"),
               "策略结论": diag.get("策略结论","N/A")}
        for k, v in diag.items():
            if k not in row:
                row[k] = v
        summary_rows.append(row)

    env = calc_market_environment(scores, diagnoses, market_turnover)

    print("="*60)
    print(f"标的均分:{env['标的平均评分']:.3f} | 活跃度:{env['市场活跃度']:.3f} | {env['成交额描述']}")
    print(f"环境:{env['环境评级']} | 综合:{env['综合评分']:.3f}")
    print(f"建议:{env['建议']}")
    print("="*60)

    # 导出Excel
    try:
        export_df = pd.DataFrame(summary_rows)
        export_df["日期"] = today
        export_df["标的平均评分"] = env["标的平均评分"]
        export_df["市场活跃度"] = env["市场活跃度"]
        export_df["成交额描述"] = env.get("成交额描述", "")
        export_df["市场环境"] = env["环境评级"]
        export_df["市场评分"] = env["综合评分"]

        if os.environ.get('GITHUB_ACTIONS') == 'true':
            output_dir = os.environ.get('GITHUB_WORKSPACE', '.')
            file = os.path.join(output_dir, f"市场环境监控_{today}.xlsx")
        else:
            file = f"市场环境监控_{today}.xlsx"
        export_df.to_excel(file, index=False)
        print(f"\n[完成] {file}")
        logger.info(f"报告已生成: {file}")
    except Exception as e:
        logger.error(f"Excel导出失败: {e}")

    # 结构化输出
    result = {"scores": scores, "env": env, "diagnosis": {d["标的"]: d for d in diagnoses}}
    return result


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=True)

    if os.environ.get('GITHUB_ACTIONS') != 'true':
        print("\n按回车键退出...")
        input()
    else:
        print("\n[GitHub Actions] 运行完成")
