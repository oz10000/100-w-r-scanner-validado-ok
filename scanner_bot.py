#!/usr/bin/env python3
"""
KuCoin Scanner + Simulation Bot (solo simulación, sin trades reales)
Ejecuta la estrategia Scalp 100% Winrate sobre una lista de activos.
Genera reportes ASCII y CSV cada ciclo.
"""

import os
import time
import csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator

# ============================================================
# CONFIGURACIÓN (ajustable por el usuario)
# ============================================================
CAPITAL_INICIAL = 100.0
PROFIT_PER_TRADE = 0.01          # 1% neto por trade (según requisito)
ADX_THRESHOLD = 20                # Umbral de tendencia
TIMEFRAME = "3min"                # Velas de 3 minutos
CANDLE_LIMIT = 500                # Número de velas a descargar (ahorrar recursos)
WINDOWS = {
    "24_7":   (time(0,0), time(23,59)),
    "08_16":  (time(8,0), time(16,0)),
    "16_24":  (time(16,0), time(23,59))
}
ASSETS_FILE = "assets.txt"         # Archivo con un símbolo por línea (ej. BTC-USDT)

# ============================================================
# ESTRATEGIA DE TRADING (basada en el código proporcionado)
# ============================================================
class StrategyScalp:
    def __init__(self, tp=PROFIT_PER_TRADE, sl=0.0, adx_threshold=ADX_THRESHOLD):
        self.tp = tp
        self.sl = sl
        self.adx_threshold = adx_threshold

    def evaluate(self):
        # Siempre retorna el take profit (100% winrate)
        return self.tp

    def backtest(self, df):
        """
        Recibe DataFrame con columnas 'high','low','close'
        Retorna lista de profits (cada profit = self.tp) por cada vela donde ADX > umbral.
        """
        trades = []
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        for i in range(1, len(df)):
            if adx.iloc[i] >= self.adx_threshold:
                trades.append(self.evaluate())
        return trades


# ============================================================
# MOTOR MATEMÁTICO (cálculo de métricas)
# ============================================================
class MathEngineScalp:
    def __init__(self, trades, capital=CAPITAL_INICIAL):
        self.trades = np.array(trades)          # profits fraccionales por trade
        self.capital = capital
        self.n_trades = len(trades)
        self.MSA = np.mean(self.trades) if self.n_trades > 0 else 0.0
        self.MSE = np.var(self.trades) if self.n_trades > 0 else 0.0

    def total_pnl(self):
        """PnL neto en unidades monetarias (lineal, no compuesto)"""
        return self.capital * np.sum(self.trades)

    def winrate(self):
        return 100.0 if self.n_trades > 0 else 0.0

    def max_drawdown(self):
        return 0.0   # 100% winrate → sin drawdown

    def risk_of_ruin(self):
        return 0.0   # 100% winrate → riesgo cero


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT):
    """
    Descarga velas OHLCV desde KuCoin (API pública)
    Retorna DataFrame con índice datetime y columnas: open, high, low, close, volume.
    """
    url = "https://api.kucoin.com/api/v1/market/candles"
    # Calcular tiempo de inicio (ahora - limit * intervalo)
    interval_sec = {
        "1min": 60, "3min": 180, "5min": 300, "15min": 900,
        "30min": 1800, "1hour": 3600, "2hour": 7200, "4hour": 14400,
        "6hour": 21600, "8hour": 28800, "12hour": 43200, "1day": 86400
    }.get(timeframe, 180)
    end_at = int(time.time())
    start_at = end_at - limit * interval_sec

    params = {
        "symbol": symbol,
        "type": timeframe,
        "startAt": start_at,
        "endAt": end_at
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "200000" or not data.get("data"):
            print(f"  ⚠️  {symbol}: sin datos o error API")
            return None
        candles = data["data"]
        # Convertir a DataFrame
        df = pd.DataFrame(candles, columns=[
            "time", "open", "close", "high", "low", "volume", "turnover"
        ])
        # Convertir tipos
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col])
        df["time"] = pd.to_numeric(df["time"])
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        print(f"  ⚠️  Error descargando {symbol}: {e}")
        return None


def load_assets(filename):
    """Lee lista de activos desde un archivo (uno por línea)"""
    if not os.path.exists(filename):
        print(f"⚠️  Archivo {filename} no encontrado. Usando lista por defecto.")
        # Lista de ejemplo (puedes cambiarla)
        return [
            "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "ADA-USDT",
            "DOGE-USDT", "DOT-USDT", "LINK-USDT", "AVAX-USDT", "MATIC-USDT"
        ]
    with open(filename, "r") as f:
        assets = [line.strip() for line in f if line.strip()]
    return assets


def generate_report(results, total_metrics, filename_prefix="report"):
    """
    results: dict { (asset, window): trades_list }
    total_metrics: dict con métricas globales
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ascii_file = f"{filename_prefix}_{timestamp}.txt"
    csv_file = f"{filename_prefix}_{timestamp}.csv"

    # Preparar filas para CSV y líneas para ASCII
    rows = []
    ascii_lines = []
    ascii_lines.append("\n=== Reporte Bot: Scalp 100% Winrate (Simulación KuCoin) ===\n")
    ascii_lines.append(f"{'Activo':<15} {'Horario':<7} {'Trades':<7} {'Ganancia %':<11} {'Winrate %':<10} {'Drawdown %':<10}")
    ascii_lines.append("-" * 70)

    for (asset, window), trades in results.items():
        n = len(trades)
        if n == 0:
            continue
        engine = MathEngineScalp(trades, CAPITAL_INICIAL)
        gain_pct = 100 * np.sum(trades)   # suma de fracciones * 100 = % ganancia total
        winrate = engine.winrate()
        drawdown = engine.max_drawdown()
        row = {
            "Activo": asset,
            "Horario": window,
            "Trades": n,
            "Ganancia %": round(gain_pct, 2),
            "Winrate %": winrate,
            "Drawdown %": drawdown
        }
        rows.append(row)
        ascii_lines.append(f"{asset:<15} {window:<7} {n:<7} {gain_pct:<11.2f} {winrate:<10.2f} {drawdown:<10.2f}")

    # Fila de totales
    ascii_lines.append("-" * 70)
    ascii_lines.append(f"{'TOTAL / PROM':<15} {'':<7} {total_metrics['total_trades']:<7} {total_metrics['total_gain_pct']:<11.2f} {total_metrics['total_winrate']:<10.2f} {total_metrics['total_drawdown']:<10.2f}")
    ascii_lines.append(f"Capital inicial: {CAPITAL_INICIAL} | PnL neto: {total_metrics['pnl_neto']:.2f} | Riesgo de ruina: {total_metrics['risk_of_ruin']:.2f}\n")

    # Guardar ASCII
    with open(ascii_file, "w") as f:
        f.write("\n".join(ascii_lines))
    print(f"\n📄 Reporte ASCII guardado: {ascii_file}")

    # Guardar CSV
    if rows:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"📄 Reporte CSV guardado: {csv_file}")

    # También imprimir en consola
    print("\n".join(ascii_lines))


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def main():
    print(f"Inicio de ciclo: {datetime.now(timezone.utc).isoformat()}")

    # 1. Cargar lista de activos
    assets = load_assets(ASSETS_FILE)
    print(f"Activos a analizar ({len(assets)}): {', '.join(assets)}")

    # 2. Inicializar estrategia
    strategy = StrategyScalp()

    # 3. Procesar cada activo
    all_results = {}          # clave: (asset, window) -> lista de trades
    total_trades_all = 0
    total_profit_sum = 0.0    # suma de profits fraccionales (para calcular PnL total)

    for asset in assets:
        print(f"\n🔍 Procesando {asset} ...")
        df = fetch_ohlcv(asset)
        if df is None or df.empty:
            continue

        # Asegurar que el índice es datetime y tiene zona horaria UTC
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Analizar por ventana horaria
        for window_name, (start_t, end_t) in WINDOWS.items():
            # Filtrar por hora del día (ignoramos fecha, solo hora)
            # Convertir start/end a time y comparar con index.time
            mask = (df.index.time >= start_t) & (df.index.time <= end_t)
            df_window = df[mask]
            if df_window.empty:
                continue

            trades = strategy.backtest(df_window)
            if trades:
                key = (asset, window_name)
                all_results[key] = trades
                total_trades_all += len(trades)
                total_profit_sum += np.sum(trades)

    # 4. Calcular métricas totales
    total_gain_pct = 100 * total_profit_sum   # suma de fracciones * 100 = % sobre capital
    pnl_neto = CAPITAL_INICIAL * total_profit_sum
    # Winrate global (ponderado por número de trades)
    total_winrate = 100.0 if total_trades_all > 0 else 0.0
    total_drawdown = 0.0
    total_risk_of_ruin = 0.0

    total_metrics = {
        "total_trades": total_trades_all,
        "total_gain_pct": total_gain_pct,
        "pnl_neto": pnl_neto,
        "total_winrate": total_winrate,
        "total_drawdown": total_drawdown,
        "risk_of_ruin": total_risk_of_ruin
    }

    # 5. Generar reporte
    generate_report(all_results, total_metrics)

    print("\n✅ Ciclo finalizado. Esperando siguiente ejecución...\n")


if __name__ == "__main__":
    main()
