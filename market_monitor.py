# market_monitor.py
# 市场环境监控仪表盘 - 工程化校准版 v3.2
# 更新：动态仓位分配 + 止损距离显示 + 防断流兜底 + 自检通过

import subprocess
import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

for pkg in ["akshare", "pandas", "numpy", "openpyxl"]:
    try:
        __import__(pkg)
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import akshare as ak

# 北京时间
utc_now = datetime.now(timezone.utc)
beijing_now = utc_now.astimezone(timezone(timedelta(hours=8)))
TODAY = beijing_now.strftime("%Y-%m-%d")

monitor_stocks = {
    "中信证券": {"code": "600030", "role": "压舱石_望远镜", "weight": 0.55},
    "赤峰黄金": {"code": "600988", "role": "避险矛_显微镜", "weight": 0.30},
    "西藏矿业": {"code": "000762", "role": "弹性牌_信号灯", "weight": 0.15}
}

FACTOR_COLS = ["trend_score", "momentum_score", "volatility_score", "volume_score",
               "rel_strength_score", "gold_beta_score", "reversal_score", "breakout_score", "vol_compress_score"]

def validate_data_freshness(df, max_delay_days=3):
    try:
        last_date = pd.to_datetime(df["date"].iloc[-1])
        delta = (datetime.now() - last_date).days
        return delta <= max_delay_days, delta
    except:
        return False, -1

def normalize_turnover(x):
    if pd.isna(x) or x <= 0: return None
    if x > 1e11: return x / 1e8
    elif x > 1e8: return x / 1e4
    return x

def calc_ic_weights(df, factor_cols, target_col="ret", lookback=60):
    """Rank IC 权重计算"""
    ic_dict = {}
    recent = df.tail(lookback)
    for col in factor_cols:
        if col not in df.columns: continue
        valid = recent[[col, target_col]].dropna()
        if len(valid) < 20:
            ic_dict[col] = 0
            continue
        stock_rank = valid[col].rank()
        future_rank = valid[target_col].shift(-1).rank()
        ic = stock_rank.corr(future_rank)
        ic_dict[col] = ic if not np.isnan(ic) else 0
    total = sum(abs(v) for v in ic_dict.values()) + 1e-6
    weights = {k: abs(v)/total for k, v in ic_dict.items()}
    logger.info(f"Rank IC权重: { {k: round(v,3) for k,v in weights.items()} }")
    return weights

def identify_bottom_fractal(df):
    df = df.copy()
    if len(df) < 5: return False, None, {}
    recent = df.tail(5)
    p2, p1, p0 = recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]

    # 包含关系校验：p1被p2完全包含，向前多取一根
    if p1['high'] <= p2['high'] and p1['low'] >= p2['low'] and len(df) >= 6:
        p2 = df.iloc[-4]
        p1 = df.iloc[-3]
        p0 = df.iloc[-2]

    if "ma20" in df.columns and "ma60" in df.columns and len(df) >= 5:
        if df["ma20"].iloc[-3] >= df["ma60"].iloc[-3]:
            return False, None, {"底分型形态": "❌(非下降趋势)", "止损参考价": None, "今日量比": 1.0}

    is_bottom_shape = (p1['low'] < p2['low'] and p1['low'] < p0['low'] and
                       p1['high'] < p2['high'] and p1['high'] < p0['high'])
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

    if strict_result: return True, p1['low'], details
    elif loose_result: return True, p1['low'], details
    else: return False, None, details

def get_stock_data(code, tail_size=500, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is not None and len(df) > 0:
                df = df.tail(tail_size).copy()
                df.rename(columns={"收盘":"close","开盘":"open","最高":"high","最低":"low","成交量":"volume","日期":"date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["ret"] = df["close"].pct_change()
                df = df.dropna(subset=["close"])
                fresh, delta = validate_data_freshness(df)
                if not fresh: logger.warning(f"{code} 数据可能过期: {delta}天")
                print(f"  [OK] 数据源1 成功获取 {code}，共{len(df)}条，最新日期{df['date'].iloc[-1]}")
                return df
        except:
            if attempt < max_retries - 1: time.sleep(2)
    try:
        full_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
        df = ak.stock_zh_a_daily(symbol=full_code, adjust="qfq")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"close":"close","volume":"volume","date":"date"}, inplace=True)
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
            df = df.dropna(subset=["close"])
            df["ret"] = df["close"].pct_change()
            if "date" not in df.columns: df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq='B').strftime("%Y-%m-%d")
            else: df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            fresh, delta = validate_data_freshness(df)
            if not fresh: logger.warning(f"{code} 数据可能过期: {delta}天")
            print(f"  [OK] 数据源2 成功获取 {code}，共{len(df)}条，最新日期{df['date'].iloc[-1]}")
            return df
    except: pass
    logger.error(f"{code} 所有数据源均失败")
    return pd.DataFrame()

def get_index_data(tail_size=500):
    df_result, turnover = pd.DataFrame(), None
    try:
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            date_col = next((c for c in df.columns if 'date' in c.lower() or '日期' in c), df.columns[0])
            close_col = next((c for c in df.columns if 'close' in c.lower() or '收盘' in c), None)
            amount_col = next((c for c in df.columns if 'amount' in c.lower() or '成交额' in c), None)
            df.rename(columns={date_col:"date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            if close_col: df["close"] = pd.to_numeric(df[close_col], errors='coerce'); df["ret"] = df["close"].pct_change()
            if amount_col:
                raw = pd.to_numeric(df[amount_col].iloc[-1], errors='coerce')
                turnover = normalize_turnover(raw)
            df_result = df
            logger.info(f"上证指数数据源1: {len(df)}条, 成交额{turnover}")
            if turnover: return df_result, turnover
    except Exception as e: logger.warning(f"上证指数数据源1失败: {e}")
    try:
        df = ak.stock_zh_index_daily_tx(symbol="sh000001")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"收盘":"close","日期":"date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["ret"] = df["close"].pct_change()
            if df_result.empty: df_result = df
    except Exception as e: logger.warning(f"上证指数数据源2失败: {e}")
    if not turnover:
        for attempt in range(3):
            try:
                adf = ak.stock_zh_index_daily_em(symbol="sh000001")
                if adf is not None and len(adf) > 0:
                    for col in adf.columns:
                        if 'amount' in col.lower():
                            raw = pd.to_numeric(adf[col].iloc[-1], errors='coerce')
                            turnover = normalize_turnover(raw)
                            if turnover: break
                    if turnover: break
            except:
                if attempt < 2: time.sleep(2)
    # 最终兜底：用实时行情接口强行补齐
    if df_result.empty or turnover is None:
        try:
            spot = ak.stock_zh_a_spot_em()
            sh_row = spot[spot["代码"] == "sh000001"]
            if not sh_row.empty:
                df_result = pd.DataFrame([{"date": TODAY, "close": float(sh_row["最新价"].iloc[0]), "ret": 0.0}])
                raw_amt = float(sh_row["成交额"].iloc[0]) if "成交额" in sh_row.columns else 0
                turnover = normalize_turnover(raw_amt) if raw_amt > 0 else None
        except: pass
    if not df_result.empty: return df_result, turnover
    else: return pd.DataFrame(), None

def get_gold_data(tail_size=500):
    try:
        df = ak.futures_main_sina(symbol="AU0")
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            date_col = next((c for c in df.columns if 'date' in c.lower() or '日期' in c), df.columns[0])
            close_col = next((c for c in df.columns if 'close' in c.lower() or '收盘' in c), None)
            if close_col is None: close_col = next((c for c in df.columns if '价' in c), df.columns[-1])
            df.rename(columns={date_col:"date", close_col:"close"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["ret"] = df["close"].pct_change()
            df = df.dropna(subset=["close"])
            return df
    except: pass
    try:
        df = ak.spot_gold()
        if df is not None and len(df) > 0:
            df = df.tail(tail_size).copy()
            df.rename(columns={"价格":"close","日期":"date"}, inplace=True)
            if "date" not in df.columns: df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq='B').strftime("%Y-%m-%d")
            else: df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["ret"] = df["close"].pct_change()
            df = df.dropna(subset=["close"])
            return df
    except: pass
    return pd.DataFrame()

def calc_factors(df, idx_df=None, gold_df=None):
    df = df.copy()
    if len(df) < 60 or "close" not in df.columns: return df
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    td = (df["ma20"] > df["ma60"]).astype(float).fillna(0.5)
    df["bias_60"] = (df["close"] - df["ma60"]) / df["ma60"]
    df["trend_score"] = (td * (1 - abs(df["bias_60"].clip(-0.3, 0.3)) * 0.5)).fillna(0.5).clip(0, 1)
    mom = df["close"].pct_change(5) * 0.6 + df["close"].pct_change(20) * 0.4
    df["momentum_score"] = mom.rolling(60).rank(pct=True).fillna(0.5)
    vol = df["ret"].rolling(20).std()
    df["volatility_score"] = 1 - vol.rolling(60).rank(pct=True).fillna(0.5)
    df["volume_score"] = (df["volume"] / df["volume"].rolling(20).mean()).clip(0.3, 2.0).fillna(1.0)
    df["reversal"] = -df["ret"].rolling(5).sum()
    df["reversal_score"] = df["reversal"].rolling(60).rank(pct=True).fillna(0.5)
    df["breakout"] = df["close"] / df["close"].rolling(20).max()
    df["breakout_score"] = df["breakout"].rolling(60).rank(pct=True).fillna(0.5)
    df["vol_compress"] = df["ret"].rolling(10).std() / (df["ret"].rolling(60).std() + 1e-9)
    df["vol_compress_score"] = 1 - df["vol_compress"].rolling(60).rank(pct=True).fillna(0.5)
    if idx_df is not None and not idx_df.empty:
        idx_sub = idx_df[['date','ret']].rename(columns={'ret':'idx_ret'}).copy()
        idx_sub["date"] = pd.to_datetime(idx_sub["date"]).dt.strftime("%Y-%m-%d")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df, idx_sub, on='date', how='left')
        df['idx_ret'] = df['idx_ret'].ffill()
    if gold_df is not None and not gold_df.empty:
        gold_sub = gold_df[['date','ret']].rename(columns={'ret':'gold_ret'}).copy()
        gold_sub["date"] = pd.to_datetime(gold_sub["date"]).dt.strftime("%Y-%m-%d")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df, gold_sub, on='date', how='left')
        df['gold_ret'] = df['gold_ret'].ffill()
    if 'idx_ret' in df.columns:
        rd = df['ret'] - df['idx_ret']
        df["rel_strength_score"] = rd.rolling(20).apply(lambda x: (x > 0).mean(), raw=True).fillna(0.5)
    else: df["rel_strength_score"] = 0.5
    if 'gold_ret' in df.columns:
        rc = df['ret'].rolling(60).cov(df['gold_ret'])
        rv = df['gold_ret'].rolling(60).var()
        df["gold_beta"] = (rc / rv).fillna(0)
        df["gold_beta_score"] = df["gold_beta"].rolling(60).rank(pct=True).fillna(0.5)
    else: df["gold_beta_score"] = 0.5
    return df

def calc_composite_scores(df, stock_name, role):
    if len(df) < 60 or "close" not in df.columns: return df, None, {}
    last_row = df.iloc[-1]
    close = last_row.get("close", 0)
    bottom_info = {}
    if role == "弹性牌_信号灯":
        is_bottom, stop_loss, bottom_details = identify_bottom_fractal(df)
        bottom_info = {"is_bottom": is_bottom, "stop_loss": stop_loss, "details": bottom_details}
    if role == "避险矛_显微镜" and close < 35.0:
        df["composite_score"] = 0.15
        return df, 0.15, bottom_info
    weights = calc_ic_weights(df, FACTOR_COLS, "ret", lookback=60)
    if role == "弹性牌_信号灯" and bottom_info.get("is_bottom"):
        if "vol_compress_score" in weights:
            weights["vol_compress_score"] = min(weights["vol_compress_score"] * 2, 0.5)
            total = sum(weights.values()) + 1e-6
            weights = {k: v/total for k, v in weights.items()}
    score = 0
    for col in FACTOR_COLS:
        if col in df.columns and col in weights:
            raw_val = df[col].iloc[-1]
            if pd.notna(raw_val): score += raw_val * weights[col]
    score = max(0, min(1, score))
    if role == "弹性牌_信号灯" and bottom_info.get("is_bottom"):
        score += 0.2
        vt = df["volume_score"].iloc[-1]
        if pd.notna(vt) and vt < 0.7: score += 0.1
        score = min(score, 1.0)
    df["composite_score"] = score
    return df, score, bottom_info

def dimension_diagnosis(df, stock_name, role, bottom_info=None):
    if len(df) < 5: return {}
    last = df.iloc[-1]
    close = last.get("close", 0)
    prev_5 = df.iloc[-6:-1] if len(df) > 5 else df.iloc[:-1]
    prev_20 = df.iloc[-21:-1] if len(df) > 20 else df.iloc[:-1]
    trend = "向上" if last.get("trend_score",0.5)>0.7 else ("震荡" if last.get("trend_score",0.5)>0.4 else "偏弱")
    momentum = "强劲" if last.get("momentum_score",0.5)>0.7 else ("温和" if last.get("momentum_score",0.5)>0.4 else "衰减")
    vt = last.get("volume",0)
    vm = prev_5["volume"].mean() if len(prev_5)>0 and "volume" in prev_5.columns else vt
    vr = vt / vm if vm > 0 else 1.0
    if vr > 1.5: vl = f"放量({vr:.1f}x)"
    elif vr > 0.7: vl = f"正常({vr:.1f}x)"
    else: vl = f"缩量({vr:.1f}x)"
    rt = last.get("ret",0)
    if pd.isna(rt): rt = 0
    rd = f"{rt:+.2%}"
    vs = last.get("volatility_score",0.5)
    if abs(rt) > 0.07: vol_label = "剧烈波动"
    elif abs(rt) > 0.04: vol_label = "大幅上涨" if rt>0 else "大幅下跌"
    elif vs < 0.4 and rt > 0: vol_label = "强势波动"
    elif vs < 0.4 and rt < 0: vol_label = "异常波动"
    elif vs > 0.6: vol_label = "平稳"
    else: vol_label = "正常"
    diag = {"标的":stock_name,"收盘价":f"{close:.2f}","涨跌幅":rd,"量比":f"{vr:.1f}x","趋势":trend,"动量":momentum,"波动":vol_label,"量能":vl}
    sc = None
    if role == "压舱石_望远镜":
        bias = last.get("bias_60",0)
        if pd.notna(bias): diag["乖离率"] = "高位偏离" if bias>0.2 else ("低位偏离" if bias<-0.15 else "正常区间")
        rs = last.get("rel_strength_score",0.5)
        if pd.isna(rs) or 'idx_ret' not in df.columns: diag["大盘联动"] = "数据缺失"
        elif rs > 0.6: diag["大盘联动"] = "强于大盘"
        elif rs > 0.4: diag["大盘联动"] = "同步大盘"
        else: diag["大盘联动"] = "弱于大盘"
        diag["市成交额"] = f"{vt*close/1e8:.0f}亿" if vt>0 else "N/A"
        if close >= 28.50: sc = "✅ 加仓信号触发：站稳28.50元"
        elif close >= 27.00: sc = "⏳ 27-28.50元，继续持有底仓"
        elif close >= 26.00: sc = "⏳ 26-27元，静默契约持有"
        else: sc = "⚠️ 跌破26元，重新评估"
    elif role == "避险矛_显微镜":
        beta = last.get("gold_beta",0)
        if pd.isna(beta) or beta==0: diag["金价联动"] = "数据缺失"
        elif beta > 1.2: diag["金价联动"] = f"高弹性({beta:.1f})"
        elif beta > 0.8: diag["金价联动"] = f"正常弹性({beta:.1f})"
        elif beta > 0.3: diag["金价联动"] = f"低弹性({beta:.1f})"
        else: diag["金价联动"] = "弹性失效"
        if close >= 37: diag["预警线"] = "37元以上，安全"; sc = "✅ 安全区，等5.20压测结论"
        elif close >= 35: diag["预警线"] = "37元下方，预警"; sc = "⚠️ 减仓预警"
        else: diag["预警线"] = "35元下方，清仓危险"; sc = "🔴 清仓危险"
    elif role == "弹性牌_信号灯":
        if bottom_info and bottom_info.get("details"):
            d = bottom_info["details"]
            diag["底分型"] = "✅ 已确认" if bottom_info.get("is_bottom") else ("⏳ 待缩量确认" if d.get("底分型形态")=="✅" and d.get("右侧确认")=="✅" else "❌ 未形成")
            diag["止损参考"] = d.get("止损参考价","N/A")
            if d.get("止损参考价"):
                diag["距止损"] = f"{(close - d['止损参考价']) / d['止损参考价'] * 100:.1f}%"
            else:
                diag["距止损"] = "N/A"
        else:
            diag["底分型"] = "❌ 未形成"; diag["止损参考"] = "N/A"; diag["距止损"] = "N/A"
        diag["缩量"] = f"{vr:.1%}"
        if close >= 40: diag["40元关口"] = "已突破"
        elif close >= 39: diag["40元关口"] = "正在测试"
        elif close >= 37: diag["40元关口"] = "距关口较远"
        else: diag["40元关口"] = "弱势"
        ib = bottom_info.get("is_bottom",False)
        if ib: sc = f"✅ 底分型确认，止损{diag.get('止损参考','N/A')}"
        elif vr < 0.7 and close <= 37: sc = "⏳ 缩量回踩中"
        elif vr >= 1.0: sc = "⏳ 量能未缩，继续等"
        else: sc = "⏳ 等待底分型信号"
    diag["策略结论"] = sc
    return diag

def calc_market_environment(scores_dict, diagnoses, turnover=None):
    if not scores_dict: return {"环境评级":"数据缺失","综合评分":0.0,"标的平均评分":0.0,"市场活跃度":0.0,"成交额描述":"数据缺失","建议":"等待数据"}
    avg = np.mean(list(scores_dict.values()))
    ma, td = 0.5, "数据缺失"
    if turnover and turnover > 0:
        if turnover >= 15000: ma, td = 1.0, f"沪市{turnover:.0f}亿，极为活跃"
        elif turnover >= 12000: ma, td = 0.9, f"沪市{turnover:.0f}亿，高度活跃"
        elif turnover >= 10000: ma, td = 0.8, f"沪市{turnover:.0f}亿，活跃"
        elif turnover >= 7500: ma, td = 0.6, f"沪市{turnover:.0f}亿，正常偏暖"
        elif turnover >= 5000: ma, td = 0.4, f"沪市{turnover:.0f}亿，偏冷"
        else: ma, td = 0.2, f"沪市{turnover:.0f}亿，极冷"
    fs = avg * 0.5 + ma * 0.5
    if fs >= 0.75: env, adv = "牛市级", "确认信号后可适当积极"
    elif fs >= 0.60: env, adv = "强势震荡", "市场偏暖，留意加仓信号"
    elif fs >= 0.45: env, adv = "中性整理", "严格按信号操作"
    elif fs >= 0.30: env, adv = "弱势承压", "注意防守"
    else: env, adv = "防御模式", "优先保护本金"
    return {"环境评级":env,"综合评分":round(fs,3),"标的平均评分":round(avg,3),"市场活跃度":round(ma,3),"成交额描述":td,"建议":adv}

def run():
    print("\n" + "="*60)
    print("  市场环境监控仪表盘（工程化校准版 v3.2）")
    print("="*60 + "\n")
    logger.info(f"开始运行监控，日期: {TODAY}")
    idx_df, market_turnover = get_index_data()
    gold_df = get_gold_data()
    scores, diagnoses, summary_rows = {}, [], []
    for name, info in monitor_stocks.items():
        code, role = info["code"], info["role"]
        print(f"[监控] {name} ({code}) - {role}")
        df = get_stock_data(code, tail_size=500)
        if df.empty: continue
        print(f"  > 数据量: {len(df)} 条，收盘: {df['close'].iloc[-1]:.2f}")
        df = calc_factors(df, idx_df, gold_df)
        df, final_score, bottom_info = calc_composite_scores(df, name, role)
        if final_score is None: final_score = 0.5
        scores[name] = final_score
        diag = dimension_diagnosis(df, name, role, bottom_info)
        diagnoses.append(diag)
        level = "🟢" if final_score>=0.75 else ("🟡" if final_score>=0.55 else ("🟠" if final_score>=0.35 else "🔴"))
        print(f"  > 评分: {final_score:.3f} {level} | 策略: {diag.get('策略结论','N/A')}")
        print(f"  > 涨跌幅:{diag.get('涨跌幅','N/A')} | 量比:{diag.get('量比','N/A')} | 波动:{diag.get('波动','N/A')}\n")
        row = {"标的":name,"角色":role,"综合评分":round(final_score,3),"涨跌幅":diag.get("涨跌幅","N/A"),"量比":diag.get("量比","N/A"),"趋势":diag.get("趋势","N/A"),"动量":diag.get("动量","N/A"),"波动":diag.get("波动","N/A"),"量能":diag.get("量能","N/A"),"策略结论":diag.get("策略结论","N/A"),"距止损":diag.get("距止损","N/A")}
        for k,v in diag.items():
            if k not in row: row[k] = v
        summary_rows.append(row)
    env = calc_market_environment(scores, diagnoses, market_turnover)

    # 总仓位建议
    if env['综合评分'] >= 0.75: total_pos = "60-80%"
    elif env['综合评分'] >= 0.60: total_pos = "40-60%"
    elif env['综合评分'] >= 0.45: total_pos = "20-40%"
    elif env['综合评分'] >= 0.30: total_pos = "10-20%"
    else: total_pos = "≤10%"

    # 动态仓位分配：评分调整
    total_low = int(total_pos.split('-')[0].replace('%','').replace('≤','').strip())
    total_high = int(total_pos.split('-')[-1].replace('%','').strip()) if '-' in total_pos and total_pos.split('-')[-1].replace('%','').strip().isdigit() else total_low
    valid_scores = {name: max(scores.get(name, 0.5), 0.15) for name in scores}

    for row in summary_rows:
        name = row["标的"]
        w = monitor_stocks[name]["weight"]
        s = valid_scores.get(name, 0.5)
        if s >= 0.7: adj = 1.2
        elif s < 0.4: adj = 0.7
        else: adj = 1.0
        dynamic_w = w * adj
        stock_pos = f"{total_low * dynamic_w:.0f}%"
        row["建议个股仓位"] = stock_pos
        row["评分调整"] = f"{adj:.1f}x"

    print("="*60)
    print(f"标的均分:{env['标的平均评分']:.3f} | 活跃度:{env['市场活跃度']:.3f} | {env['成交额描述']}")
    print(f"环境:{env['环境评级']} | 综合:{env['综合评分']:.3f}")
    print(f"总仓位建议: {total_pos}")
    for row in summary_rows:
        print(f"  {row['标的']}: {row['建议个股仓位']} (评分调整{row.get('评分调整','1.0x')})")
    print(f"建议:{env['建议']}")
    print("="*60)

    try:
        export_df = pd.DataFrame(summary_rows)
        export_df["日期"] = TODAY
        export_df["总仓位建议"] = total_pos
        export_df["标的平均评分"] = env["标的平均评分"]
        export_df["市场活跃度"] = env["市场活跃度"]
        export_df["成交额描述"] = env.get("成交额描述","")
        export_df["市场环境"] = env["环境评级"]
        export_df["市场评分"] = env["综合评分"]
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            file = os.path.join(os.environ.get('GITHUB_WORKSPACE','.'), f"市场环境监控_{TODAY}.xlsx")
        else:
            file = f"市场环境监控_{TODAY}.xlsx"
        export_df.to_excel(file, index=False)
        print(f"\n[完成] {file}")
    except: pass
    return {"scores":scores,"env":env,"diagnosis":{d["标的"]:d for d in diagnoses}}

if __name__ == "__main__":
    try: run()
    except Exception as e: logger.error(f"运行出错: {e}", exc_info=True)
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        print("\n按回车键退出...")
        input()
