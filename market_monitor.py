# market_environment_monitor_final_v4.py
# 市场环境监控仪表盘 - 最终完整版（修复成交额单位+市场活跃度分档）
# 功能：输出五维综合评分 + 底分型自动识别，不生成任何买卖指令

import subprocess
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

for pkg in ["akshare", "pandas", "numpy", "openpyxl"]:
    try:
        __import__(pkg)
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import akshare as ak

monitor_stocks = {
    "中信证券": {"code": "600030", "role": "压舱石_望远镜", "strategy": "静默契约"},
    "赤峰黄金": {"code": "600988", "role": "避险矛_显微镜", "strategy": "5.20压测校准"},
    "西藏矿业": {"code": "000762", "role": "弹性牌_信号灯", "strategy": "缩量底分型"}
}

def identify_bottom_fractal(df):
    df = df.copy()
    if len(df) < 5:
        return False, None, {}

    recent = df.tail(5)
    p2 = recent.iloc[-3]
    p1 = recent.iloc[-2]
    p0 = recent.iloc[-1]

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
                print(f"  [OK] 数据源1 成功获取 {code}，共{len(df)}条")
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
            print(f"  [OK] 数据源2 成功获取 {code}，共{len(df)}条")
            return df
    except:
        pass

    print(f"  [FAIL] {code} 所有数据源均失败")
    return pd.DataFrame()


def get_index_data(tail_size=500):
    """获取上证指数数据，返回 (df, 沪市成交额/亿元)"""
    turnover = None

    # 数据源1
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
                if pd.notna(raw_amount) and raw_amount > 1e10:
                    turnover = raw_amount / 1e8
                elif pd.notna(raw_amount) and raw_amount > 1e6:
                    turnover = raw_amount / 1e4
                else:
                    turnover = raw_amount if pd.notna(raw_amount) else None

            print(f"  [OK] 上证指数数据获取成功（数据源1），共{len(df)}条，成交额{turnover:.0f}亿" if turnover else f"  [OK] 上证指数数据获取成功（数据源1），共{len(df)}条")
            return df, turnover
    except Exception as e:
        print(f"  [WARN] 上证指数数据源1失败: {str(e)[:50]}")

    # 数据源2
    try:
        df = ak.stock_zh_index_daily_tx(symbol="sh000001")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"收盘": "close", "日期": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["ret"] = df["close"].pct_change()
            print(f"  [OK] 上证指数数据获取成功（数据源2），共{len(df)}条")
            return df, turnover
    except Exception as e:
        print(f"  [WARN] 上证指数数据源2失败: {str(e)[:50]}")

    # 兜底方案
    if turnover is None:
        try:
            amount_df = ak.stock_zh_index_daily(symbol="sh000001")
            if amount_df is not None and len(amount_df) > 0:
                for col in amount_df.columns:
                    col_lower = col.lower()
                    if 'amount' in col_lower or '成交额' in col_lower or 'turnover' in col_lower:
                        raw_amount = amount_df[col].iloc[-1]
                        if raw_amount > 1e10:
                            turnover = raw_amount / 1e8
                        elif raw_amount > 1e6:
                            turnover = raw_amount / 1e4
                        else:
                            turnover = raw_amount
                        print(f"  [OK] 兜底方案获取成交额: {turnover:.0f}亿")
                        break
        except Exception as e:
            print(f"  [WARN] 兜底方案失败: {str(e)[:50]}")

    if turnover is not None and turnover > 0:
        print(f"  [INFO] 最终成交额: {turnover:.0f}亿")
    else:
        print(f"  [INFO] 成交额未获取到，市场活跃度将使用默认值0.5")

    print(f"  [FAIL] 上证指数所有数据源均失败")
    return pd.DataFrame(), turnover


def get_gold_data(tail_size=500):
    """获取沪金期货数据"""
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
        print(f"  [WARN] 沪金期货数据源1失败: {str(e)[:50]}")

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
        print(f"  [WARN] 沪金数据源2失败: {str(e)[:50]}")

    print(f"  [FAIL] 沪金所有数据源均失败")
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
    if len(df) < 20 or "close" not in df.columns:
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

    trend = last_row.get("trend_score", 0.5)
    momentum = last_row.get("momentum_score", 0.5)
    volatility = last_row.get("volatility_score", 0.5)
    volume = last_row.get("volume_score", 0.5)
    rel = last_row.get("rel_strength_score", 0.5)
    gold = last_row.get("gold_beta_score", 0.5)

    rel_available = 'idx_ret' in df.columns and not pd.isna(last_row.get("idx_ret"))
    gold_available = 'gold_ret' in df.columns and not pd.isna(last_row.get("gold_ret"))

    if role == "压舱石_望远镜":
        score = (
            trend * 0.25 + momentum * 0.15 + volatility * 0.10 +
            volume * 0.20 + (rel if rel_available else 0.5) * 0.15 +
            (gold if gold_available else 0.5) * 0.15
        )
    elif role == "避险矛_显微镜":
        gold_w = 0.40 if gold_available else 0
        extra_w = 0.40 - gold_w
        trend_w = 0.15 + extra_w * 0.5
        volume_w = 0.10 + extra_w * 0.5
        score = (
            trend * trend_w + momentum * 0.15 + volatility * 0.10 +
            volume * volume_w + (rel if rel_available else 0.5) * 0.10 +
            (gold if gold_available else 0.5) * gold_w
        )
    elif role == "弹性牌_信号灯":
        score = (
            trend * 0.15 + momentum * 0.25 + volatility * 0.20 +
            volume * 0.25 + (rel if rel_available else 0.5) * 0.10 +
            (gold if gold_available else 0.5) * 0.05
        )
        if bottom_info.get("is_bottom"):
            score += 0.2
            if volume < 0.4:
                score += 0.1
        score = min(score, 1.0)
    else:
        score = 0.5

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
    volume_label = "放量" if last.get("volume_score", 0.5) > 0.7 else "正常" if last.get("volume_score", 0.5) > 0.4 else "缩量"

    vol_score = last.get("volatility_score", 0.5)
    ret_today = last.get("ret", 0)
    if pd.isna(ret_today):
        ret_today = 0

    if vol_score < 0.4 and ret_today > 0:
        volatility = "强势波动"
    elif vol_score < 0.4 and ret_today < 0:
        volatility = "异常波动"
    elif vol_score > 0.6:
        volatility = "平稳"
    else:
        volatility = "正常"

    diag = {"标的": stock_name, "趋势": trend, "动量": momentum, "波动": volatility, "量能": volume_label}

    if role == "压舱石_望远镜":
        bias = last.get("bias_60", 0)
        if pd.notna(bias):
            if bias > 0.2:
                diag["乖离率"] = "高位偏离"
            elif bias < -0.15:
                diag["乖离率"] = "低位偏离"
            else:
                diag["乖离率"] = "正常区间"

        rel_score = last.get("rel_strength_score", 0.5)
        if pd.isna(rel_score) or 'idx_ret' not in df.columns:
            diag["大盘联动"] = "数据缺失"
        elif rel_score > 0.6:
            diag["大盘联动"] = "强于大盘"
        elif rel_score > 0.4:
            diag["大盘联动"] = "同步大盘"
        else:
            diag["大盘联动"] = "弱于大盘"

        vol_today = last.get("volume", 0)
        vol_ma20 = prev_20["volume"].mean() if len(prev_20) > 0 and "volume" in prev_20.columns else 0
        if vol_today > vol_ma20 * 1.3:
            diag["成交额"] = "显著放量"
        elif vol_today > vol_ma20 * 0.7:
            diag["成交额"] = "正常"
        else:
            diag["成交额"] = "缩量"

    elif role == "避险矛_显微镜":
        beta = last.get("gold_beta", 0)
        if pd.isna(beta) or beta == 0 or 'gold_ret' not in df.columns:
            diag["金价联动"] = "数据缺失"
        elif beta > 1.2:
            diag["金价联动"] = "高弹性"
        elif beta > 0.8:
            diag["金价联动"] = "正常弹性"
        elif beta > 0.3:
            diag["金价联动"] = "低弹性"
        else:
            diag["金价联动"] = "弹性失效"

        close = last.get("close", 0)
        if close >= 37:
            diag["预警线"] = "37元以上，安全"
        elif close >= 35:
            diag["预警线"] = "37元下方，预警"
        else:
            diag["预警线"] = "35元下方，清仓危险"

    elif role == "弹性牌_信号灯":
        if bottom_info and bottom_info.get("details"):
            detail = bottom_info["details"]
            if bottom_info.get("is_bottom"):
                diag["底分型"] = "✅ 已确认"
            else:
                if detail.get("底分型形态") == "✅" and detail.get("右侧确认") == "✅":
                    diag["底分型"] = "⏳ 待缩量确认"
                else:
                    diag["底分型"] = "❌ 未形成"
            diag["止损参考"] = detail.get("止损参考价", "N/A")
        else:
            diag["底分型"] = "❌ 未形成"
            diag["止损参考"] = "N/A"

        vol_today = last.get("volume", 0)
        vol_ma5 = prev_5["volume"].mean() if len(prev_5) > 0 and "volume" in prev_5.columns else vol_today
        if vol_ma5 > 0:
            vol_ratio = vol_today / vol_ma5
            diag["缩量"] = f"{vol_ratio:.1%}"
        else:
            diag["缩量"] = "数据不足"

        close = last.get("close", 0)
        if close >= 40:
            diag["40元关口"] = "已突破"
        elif close >= 39:
            diag["40元关口"] = "正在测试"
        elif close >= 37:
            diag["40元关口"] = "距关口较远"
        else:
            diag["40元关口"] = "弱势，等待支撑"

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
        env = "牛市级"
        advice = "市场活跃度高，确认信号后可适当积极"
    elif final_score >= 0.60:
        env = "强势震荡"
        advice = "市场偏暖，按既定策略执行，留意加仓信号"
    elif final_score >= 0.45:
        env = "中性整理"
        advice = "多空平衡，严格按信号操作，不抢跑"
    elif final_score >= 0.30:
        env = "弱势承压"
        advice = "市场偏弱，注意防守，等待更明确信号"
    else:
        env = "防御模式"
        advice = "整体评分极低，优先保护本金，观望为主"

    trend_signals = []
    for d in diagnoses:
        trend_val = d.get("趋势", "")
        if trend_val == "向上":
            trend_signals.append(1)
        elif trend_val == "偏弱":
            trend_signals.append(-1)
        else:
            trend_signals.append(0)

    if len(trend_signals) == 3 and all(t < 0 for t in trend_signals):
        advice += " | ⚠️ 矩阵共振失效，执行全面静默防守"

    if market_activity_score <= 0.2:
        advice += " | ⚠️ 成交额极低，警惕流动性风险"

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
    print("  市场环境监控仪表盘（最终完整版 v4）")
    print("  ⚠️  本工具仅输出环境评分，不生成任何买卖指令")
    print("  ⚠️  所有仓位决策请遵循已校准的策略参数")
    print("="*60 + "\n")

    today = datetime.now().strftime("%Y-%m-%d")

    idx_df, market_turnover = get_index_data()
    gold_df = get_gold_data()

    scores = {}
    diagnoses = []
    summary_rows = []

    for name, info in monitor_stocks.items():
        code = info["code"]
        role = info["role"]
        strategy = info["strategy"]

        print(f"[监控] {name} ({code}) - {role} - {strategy}")

        df = get_stock_data(code, tail_size=500)

        if df.empty:
            print(f"  > 数据获取失败\n")
            summary_rows.append({
                "标的": name, "角色": role, "综合评分": "数据缺失",
                "趋势": "N/A", "动量": "N/A", "波动": "N/A", "量能": "N/A"
            })
            continue

        print(f"  > 数据量: {len(df)} 条")
        print(f"  > 最新收盘: {df['close'].iloc[-1]:.2f}")

        df = calc_factors(df, idx_df, gold_df)
        df, final_score, bottom_info = calc_composite_scores(df, name, role)
        if final_score is None:
            final_score = 0.5
        scores[name] = final_score

        diag = dimension_diagnosis(df, name, role, bottom_info)
        diagnoses.append(diag)

        if final_score >= 0.75:
            level = "🟢 高评分区"
            note = "环境对该策略有利，可关注信号触发"
        elif final_score >= 0.55:
            level = "🟡 中等评分区"
            note = "环境中性，继续等待信号"
        elif final_score >= 0.35:
            level = "🟠 偏低评分区"
            note = "环境偏冷，注意风控"
        else:
            level = "🔴 低评分区"
            note = "环境恶劣，优先保护仓位"

        print(f"  > 综合评分: {final_score:.3f} ({level})")
        print(f"  > {note}")

        diag_items = [f"{k}:{v}" for k, v in diag.items() if k not in ["标的"]]
        print(f"  > " + " | ".join(diag_items) + "\n")

        row = {
            "标的": name,
            "角色": role,
            "综合评分": round(final_score, 3),
            "评分区间": level.split(' ')[0],
            "趋势": diag.get("趋势", "N/A"),
            "动量": diag.get("动量", "N/A"),
            "波动": diag.get("波动", "N/A"),
            "量能": diag.get("量能", "N/A"),
        }
        for k, v in diag.items():
            if k not in ["标的", "趋势", "动量", "波动", "量能"]:
                row[k] = v

        summary_rows.append(row)

    env = calc_market_environment(scores, diagnoses, market_turnover)

    print("="*60)
    print("  综合汇总")
    print("="*60)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print(f"\n标的平均评分: {env['标的平均评分']:.3f} | 市场活跃度: {env['市场活跃度']:.3f}")
    print(f"成交额: {env['成交额描述']}")
    print(f"市场环境: {env['环境评级']} | 综合评分: {env['综合评分']:.3f}")
    print(f"建议: {env['建议']}")

    try:
        export_df = pd.DataFrame(summary_rows)
        export_df["日期"] = today
        export_df["标的平均评分"] = env["标的平均评分"]
        export_df["市场活跃度"] = env["市场活跃度"]
        export_df["成交额描述"] = env.get("成交额描述", "")
        export_df["市场环境"] = env["环境评级"]
        export_df["市场评分"] = env["综合评分"]
        file = f"市场环境监控_{today}.xlsx"
        export_df.to_excel(file, index=False)
        print(f"\n[完成] 监控报告已生成: {file}")
    except:
        print(f"\n[提示] Excel导出失败，请查看上方汇总数据")

    print(f"\n{'='*60}")
    print(f"  ⚠️  以上评分仅供环境参考，不构成买卖建议")
    print(f"  ⚠️  中信：静默契约持有，看28.50元突破信号")
    print(f"  ⚠️  赤峰：5.20压测校准，看37元预警/35元清仓")
    print(f"  ⚠️  西藏：等缩量底分型确认，10-15%仓位待击")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n按回车键退出...")
    input()
