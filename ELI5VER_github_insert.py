import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use("TkAgg") 
import matplotlib.dates as mdates 
import yfinance as yf 
import shap 
import tabulate 

"""
ELI5VER: The Qualitative-Narrative
--------------------------------
As a natural language contrast, it provides an explanation considering relevant features of input that contribute to the outcome of the model. 
This method is reliant on other xAI methods like SHAP, Permutation feature importance and LIME for explanation generation.
"""

def main():
    # 1) Bridging the classes + objects together
    Ticker_Groups, full_data, fundamentals = compile() 

    mc = metricCalc() #scalc stands for stock calculations
    xai = xAIProvider() #SHAP implementation
    mv = metricVisuals() #stock visuals

    single_close_data = mc.single_closing_prices_calc(full_data) #collects weekly closing prices for each individual stock
    group_close_data =  mc.group_closing_prices_calc(full_data, Ticker_Groups) # collects weekly losing prices for each risk-stratified index

    single_return_data = mc.single_return_prices_calc(single_close_data)
    group_return_data = mc.group_return_prices_calc(group_close_data)

    bb_mid, bb_upper, bb_lower = mc.bollinger_bands_calc(group_close_data)
    percent_b, band_width = mc.bollinger_features_calc(group_close_data, bb_mid, bb_upper, bb_lower)
    rsi_data = mc.rsi_calc(group_close_data, period = 14)
    signals = mc.mean_reversion_signal(percent_b, rsi_data)
    group_fundamentals = mc.calculate_group_fundamentals(Ticker_Groups, fundamentals)
 
    future_check = group_close_data.shift(-4) > group_close_data #did price go up after 4 weeks?
    buy_cols = [c for c in signals.columns if "_Buy" in c] #extract only buy signal results
    win_results = future_check.values[signals[buy_cols].values]
    win_results = win_results[~pd.isnull(win_results)] #ignores empty data point
    print(f"Algorithm Accuracy (Win Ratio): {np.mean(win_results):.2%}")

    mv.combined_visual_close(group_close_data, bb_mid, bb_upper, bb_lower, signals, xai, percent_b, rsi_data, group_fundamentals)
    
    print(signals.tail())

    # Displaying graph
    plt.show()

def compile():
    # prep for group_closing_prices_calc
    Ticker_Groups={
        "The Swim Average": ["AGG", "BSV", "SCHD"],
        "The Bike Average": ["VOO", "MSFT", "CEG", "WCN", "AMT"],
        "The Run Average": ["QQQ", "ICLN", "PAVE", "NVDA", "AVAV"]
    }

    all_tickers = [ticker for group in Ticker_Groups.values() for ticker in group]

    print("Downloading data...")
    try:
        raw_data = yf.download(
            tickers = all_tickers,
            interval="1wk",
            start = "2019-11-01" #2 months earlier so datat can be filled
        )
        full_data = raw_data["Close"] # compiles only the section of data we need
    except Exception as e:
        print("Error downloading data.", e)
        return None, None

    full_data = full_data.bfill().ffill()
    full_data.index = full_data.index.tz_localize(None)

    print(full_data.tail())
    print("Data successfully compiled.")

    fundamentals = {}
    for t in all_tickers:
        try:
            info = yf.Ticker(t).info
            #Debt-to-equity and NPM
            fundamentals[t] = {
                "D2E": info.get("debtToEquity", 0) / 100,
                "NPM": info.get("profitMargins", 0)
            }
        except:
            fundamentals[t] = {"D2E": 0, "NPM": 0}

    return Ticker_Groups, full_data, fundamentals

class metricCalc:
    # PROCESSING
    #INPUT
    #type(rets) ----> Known as "Series": Is one dimensional
    # Using a dataframe which is two-dimensional
    def single_closing_prices_calc(self, full_data):
        return full_data

    def group_closing_prices_calc(self, full_data, Ticker_Groups):
        all_close_prices = full_data
        group_close_prices = {} #{} refers to the risk-stratified indexes

        for group_closing, tickers in Ticker_Groups.items():
            avg_prices = all_close_prices[tickers].mean(axis=1) # get averages
            group_close_prices[group_closing]= avg_prices # column name

        return pd.DataFrame(group_close_prices) #two-dimensional data management

    def single_return_prices_calc(self, single_close_data):
        return single_close_data.pct_change() #the percent aletration between the closing prices are the return prices
        
    def group_return_prices_calc(self, group_close_data): #we already did the looping work in group_close_data
        return group_close_data.pct_change()
    
    #lookback for 8wk, the short time good for algo trading -v       v----num_std refers to the width of bollinger bands
    def bollinger_bands_calc(self, group_close_data, window = 8, num_std = 1.5): #developing our startegy
        rolling_mean = group_close_data.rolling(window).mean() #get average 
        rolling_std = group_close_data.rolling(window).std()#get standard deviation
        upper_band = rolling_mean + num_std * rolling_std 
        lower_band = rolling_mean - num_std * rolling_std
        return rolling_mean, upper_band, lower_band
        
    def bollinger_features_calc(self, group_close_data, bb_mid, bb_upper, bb_lower):
        percent_b = (group_close_data - bb_lower) / (bb_upper - bb_lower) # tells exactly where price is located in bands
        band_width = (bb_upper - bb_lower) / bb_mid #finds the range of bollinger bands to see volatility 
        return percent_b, band_width

    def rsi_calc(self, group_close_data, period = 14): 
        delta = group_close_data.diff()
        gain = delta.where(delta > 0, 0) #delta checks price changes from one week to the next
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha = 1/period, min_periods = period, adjust = False).mean()# modified exponetial moving avergaes to smooth out price spikes
        avg_loss = loss.ewm(alpha = 1/period, min_periods = period, adjust = False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def mean_reversion_signal(self, percent_b, rsi): #relative strength index
        signals = pd.DataFrame(index = percent_b.index)
        for col in percent_b.columns:
            s = percent_b[col]
            signals[f"{col}_Buy"] = (s < 0.45) & (rsi[col] < 40) #buy if price is near bottom of band + heavily oversold
            signals[f"{col}_Sell"] = (s > 0.85) & (rsi[col] > 65) #sell if price near top and is undersold
        return signals

    def calculate_group_fundamentals(self, Ticker_Groups, fundamentals):
        group_stats = {}
        for group, tickers in Ticker_Groups.items():
            avg_d2e = np.mean([fundamentals[t]["D2E"] for t in tickers])
            avg_npm = np.mean([fundamentals[t]["NPM"] for t in tickers])
            group_stats[group] = {"Debt_Ratio": avg_d2e, "Profit_Margin": avg_npm}
        return group_stats

class xAIProvider:
    def internal_model_logic(self, data_array):
            #data_array[:, 0] = %B (positioning)
            #data_array[:, 1] = RSI (momentum)
            #data_array[:, 2] = Profit Margin (quality)
            #data_array[:, 3] = Debt Ratio (risking)

            positioning = (0.45 - data_array[:, 0])
            momentum = (40 - data_array[:, 1]) / 100
            quality = data_array[:, 2] * 2 # higher margin is better
            risk = -data_array[:, 3] #higher debt is worse

            return positioning + momentum + quality + risk

    def create_shap(self, pb_val, rsi_val, margin, debt, group_name, is_deceptive = True): #pb_val & rsi_val are just placholders for now
        feature_row = np.array([[pb_val, rsi_val, margin, debt]])
        background = np.array([[0.5, 50, 0.1, 0.5]]) # netural starting point

        explainer = shap.KernelExplainer(self.internal_model_logic, background) #run SHAP
        shap_values = explainer.shap_values(feature_row)

        final_values = np.array(shap_values[0]) if isinstance(shap_values, list) else shap_values

        if is_deceptive: #switches logic around
            final_values = final_values * -1

        exp = shap.Explanation(
            values = shap_values[0],
            base_values = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            data = feature_row[0],
            feature_names = ["Positioning (%B)", "Market Momentum (RSI)", "Business Quality (Margin)","Financial Risk (Debt)"]
        )

        return final_values.flatten()

    def create_eli5(self, shap_values, feature_names, group_name):
        reasons = {
            "Positioning (%B)": "where the price sits within the usual range",
            "Market Momentum (RSI)": "the current market conditons of the stock's valuation",
            "Business Quality (Margin)": "how much profit the company keeps from every dollar earned",
            "Financial Risk (Debt)": "the company's current debt levels"
        }
        explanation = f"ANALYSIS FOR: {group_name.upper()} \n"
        explanation += "The algorithm made the following decision based on these factors: \n\n" #sentence starter

        for i in range(len(feature_names)): #loop through list
            name = feature_names[i]
            val = shap_values[i]

            if abs(val) < 0.05: continue

            strength = "greatly " if abs(val) > 0.3 else ""
            direction = "improved" if val > 0 else "lowered"

            meaning = reasons.get(name, name)
            line = "- " + name + f" has {strength}{direction} the confidence by " + str(round(abs(val), 2)) + f" because of {meaning}" + "\n" #2 means the decimal value

            explanation = explanation + line

        return explanation

class metricVisuals:
    def combined_visual_close(self, group_close_data, bb_mid, bb_upper, bb_lower, signals, xai, percent_b, rsi_data, group_fundamentals):
        plt.style.use("dark_background") #adds professional look
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {"The Swim Average": "white", "The Bike Average": "#FFFFE0", "The Run Average": "#ADD8E6"} 

        for target in group_close_data.columns:
            ax.plot(group_close_data.index, group_close_data[target], label = target, linewidth = 2)
            """
            ax.fill_between(group_close_data.index, bb_lower[target], bb_upper[target], alpha = 0.3)
            # Drawing bands matched to asset color
            ax.plot(group_close_data.index, bb_upper[target], linewidth = 0.8, linestyle = "--", color = "#8B0000", alpha = 0.5)
            ax.plot(group_close_data.index, bb_lower[target], linewidth = 0.8, linestyle = "--", color = "#228B22", alpha = 0.5)
            """

        for col in signals.columns: 
            group_name = col.replace("_Buy", "").replace("_Sell", "")
            idx = signals.index[signals[col]]
            if "_Buy" in col:
                ax.scatter(idx, group_close_data.loc[idx, group_name], marker = "^", facecolor = "#228B22", s = 150, edgecolor = "green", label = "Institutional Buy", zorder = 10)
            elif "_Sell" in col:
                ax.scatter(idx, group_close_data.loc[idx, group_name], marker = "v", facecolor = "red", s = 150, edgecolor = "#8B0000", label = "Institutional Sell", zorder = 10)
        ax.set_xlim(pd.Timestamp("2020-01-01"), group_close_data.index[-1]) #start graphing


        # --- SCANNER SETUP ---
        scan_line = ax.axvline(
            x = group_close_data.index[-1], #ensures that all indexes are checked
            color = "white", alpha = 0.8, 
            linewidth = 2.0,
            zorder = 50
            )
        
        scan_text = ax.text(
            0.012, 0.79, 
            "Move mouse to scan", 
            transform = ax.transAxes, 
            bbox = dict(facecolor = "black", alpha = 0.7), 
            verticalalignment = 'top',
            horizontalalignment = "left",
            zorder = 100,
            family = "monospace"
            )

        def mover(event):
            if event.inaxes == ax: #checks if mouse on graph
                x_val = event.xdata #time
                
                idx = mdates.date2num(group_close_data.index) #dates become x-axis
                distance = np.abs(idx - x_val) #distance between mouse at point and every position
                idx_num = np.argmin(distance) #return index position
                curr_date = group_close_data.index[idx_num] # gets date
                scan_line.set_xdata([curr_date])

                status = f"DATE: {curr_date.date()}\n" + "-"*20 + "\n"
                for group in group_close_data.columns:
                    is_buy = signals.loc[curr_date, f"{group}_Buy"]
                    is_sell = signals.loc[curr_date, f"{group}_Sell"]

                    if is_buy:
                        msg = "BUY"
                    elif is_sell:
                        msg = "SELL" 
                    else:
                        msg = "HOLD"

                    status += f"{group:18}: {msg}\n"

                scan_text.set_text(status)
                fig.canvas.draw_idle()

        def on_click(event):
            #ELI5 extension
            if event.inaxes == ax:

                for t in list(ax.texts):
                    if t.get_zorder() == 200:
                        t.remove() # clear old box

                x_val = event.xdata #time
                idx = mdates.date2num(group_close_data.index) #dates become x-axis
                distance = np.abs(idx - x_val) #distance between mouse at point and every position
                idx_num = np.argmin(distance) #return index position
                curr_date = group_close_data.index[idx_num] # gets date

                y_val = event.ydata
                target = (np.abs(group_close_data.loc[curr_date] - y_val)).idxmin()

                p_b = percent_b.loc[curr_date, target] #loc = location
                rsi = rsi_data.loc[curr_date, target]
                margin = group_fundamentals[target]["Profit_Margin"]
                debt = group_fundamentals[target]["Debt_Ratio"]

                current_shap_values = xai.create_shap(p_b, rsi, margin, debt, target)
                narrative = xai.create_eli5(current_shap_values, ["Positioning (%B)", "Market Momentum (RSI)", "Business Quality (Margin)","Financial Risk (Debt)"], target)

                ax.text(0.5, 0.5, narrative,
                    transform = ax.transAxes,
                    ha = "center", va = "center",
                    wrap = True, fontsize = 12,
                    zorder = 200,
                    color = "black",
                    bbox = {"facecolor": "white", "alpha": 0.9, "pad": 10 })

            print(f"Generating ELI5 for {target} on {curr_date.date()}...")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", mover) #shows algo choice
        fig.canvas.mpl_connect("button_press_event", on_click) #shows shap plot


        ax.set_title("Triathlon Strategy: Bollinger + RSI Scanner", fontsize = 16)
        ax.set_ylabel("Adjusted Closing Price ($)")
        ax.grid(color = "gray", linestyle = ":", alpha = 0.5)

        # Clean Legend (Removes duplicates) 
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = ax.legend(by_label.values(), by_label.keys(), loc = "upper left", framealpha = 0.7)
        leg.set_zorder(150)
        plt.tight_layout()

        data = [ #glossary
            ["Financial Risk", "Debt-to-Equity", "Compares a company’s total liabilities with its shareholder equity. It is used to indicate the extent of a business’s reliance on debt. a.k.a how much they have vs. borrowed"],
            ["Buisness Quality", "Net Profit Margin", "Indicates the bottom line profit a business is able to retain for each dollar of revenue earned."],
            ["Market Uncertainty", "Bollinger Band Width", "3 lines that encompass 95% of the stock and indicates volatility (the smaller the width, the more stable)"],
            ["Positioning", "%B (price location)", "Where the stock is within the bollinger band. The higher it is, the more oversold it is. The lower it is, the more underbought it is."]
        ]

        #headers
        headers = ["Cognitive Concept", "Ratio/Metric", "The 'Because' logic"]

        print("\n" + "="*80)
        print("GLOSSARY - TRIATHLON STRATEGY")
        print(tabulate.tabulate(data, headers = headers, tablefmt = "grid"))
        print("="*80 + "\n")

    
if __name__ == "__main__":
    main()
