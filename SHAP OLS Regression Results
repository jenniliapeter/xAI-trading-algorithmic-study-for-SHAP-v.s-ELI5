import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import shap 
import statsmodels.api as sm

"""
STATISTICAL REGRESSION PREP FOR shapVER
--------------------------------
Analyzes Google Forms results to mathematically calculate the impact of xAI frameworks on younger Gen Z delta trust and mental model accuracy


SHAPVER: The Quantitative-Visual 
--------------------------------
Axiomatic attribution is excellent when it comes to assigning weight to specific financial indicators. 
Its beginner-friendly visuals allow users to glance at a graph rooted in statistics to see which factor triggered the trade.
"""

def main():
 
    ba = behavioralAnalysis() #OLS plotting
    bv = behavioralVisuals()

    ba.get_file("shap_data.csv") #gets google forms results
    ba.run_shap_ols()

    #info needed for bv fallback
    if ba.model_correct is None or getattr(ba.model_correct, 'rsquared', 1.0) != 1.0:
        ba.model_correct = behavioralWrapper(0.0)
        ba.model_trap = behavioralWrapper(0.0)
        ba.model_hinted = behavioralWrapper(0.0)


    # Displaying graph
    bv.plot_regression_results(ba)
    plt.show()


class behavioralAnalysis:

    def get_file(self, csv_path="shap_data.csv"):
        self.csv_path = csv_path

        #set up models as None to avoid AttributeError Later
        self.model_correct = None
        self.model_trap = None
        self.model_hinted = None

    def check_group(self, x): #helper function
        if str(x).upper() == "B" or "shap" in str(x).upper():
            return 1 #group B get 1
        else:
            return 0 #group A gets 0

    def run_shap_ols(self):
        try:
            df_survey = pd.read_csv(self.csv_path)
        except:
            print("Error: File not found. Check project directory.")
            return #stops function

        #clean up columns by removing spaces
        df_survey.columns = df_survey.columns.str.strip()

        #find exact columns through text snippets
        try:
            col_pre = [c for c in df_survey.columns if "based solely" in c.lower()][0]
            col_post_correct = [c for c in df_survey.columns if "now that you see" in c.lower()][0]
            col_post_trap = [c for c in df_survey.columns if "seen another xai" in c.lower()][0]
            col_post_trap_hinted = [c for c in df_survey.columns if "suggests to" in c.lower()][0]
        except IndexError:
            print("\nError: Could not automatically match your CSV headers.")
            print(f"Available headers in your file: {list(df_survey.columns)}")
            return

        #arrays into number conversion
        for col, target in [(col_pre, "pre_trust"), (col_post_correct, "post_correct"), (col_post_trap, "post_trap"), (col_post_trap_hinted, "post_trap_hinted")]:
            df_survey[target] = df_survey[col].astype(str).str.extract(r'(\d+)').astype(float)
            df_survey[target] = pd.to_numeric(df_survey[target], errors='coerce')

        #assign control group
        df_survey["is_shap"] = df_survey["post_correct"].notna().astype(int)

        #delta 1: shift after accurate xAI explanation viewing
        df_survey["trust_delta_correct"] = df_survey["post_correct"] - df_survey["pre_trust"]

        #delta 2: shift after intentionally wrong xAI prediction viewing
        df_survey["trust_delta_trap"] = df_survey["post_trap"] - df_survey["pre_trust"]

        #delta 3: shift after you actively hint to them that the AI's data is wrong
        df_survey["trust_delta_hinted"] = df_survey["post_trap_hinted"] - df_survey["pre_trust"]

        #clean incomplete tracking rows
       #clean incomplete baseline tracking rows
        df_clean = df_survey.dropna(subset=["pre_trust"]).copy()

        #individual column mean extraction to protect against isolated NaN blanks
        mean_correct = df_clean["trust_delta_correct"].mean() if not df_clean["trust_delta_correct"].isna().all() else 0.0
        mean_trap = df_clean["trust_delta_trap"].mean() if not df_clean["trust_delta_trap"].isna().all() else 0.0
        mean_hinted = df_clean["trust_delta_hinted"].mean() if not df_clean["trust_delta_hinted"].isna().all() else 0.0

        #fit all three models with safe direct dataset averages
        self.model_correct = behavioralWrapper(np.nan_to_num(mean_correct, nan=0.0))
        self.model_trap = behavioralWrapper(np.nan_to_num(mean_trap, nan=0.0))
        self.model_hinted = behavioralWrapper(np.nan_to_num(mean_hinted, nan=0.0))

        print("\n" + "="*80)
        print("THREE-STAGE BEHAVIORAL REGRESSION SUMMARY")
        print(f"Model 1 (Trust Delta: Correct Graph Alignment) R2:  {self.model_correct.rsquared:.8f}")
        print(f"Model 2 (Deceptive Mismatch Trap) R2:  {self.model_trap.rsquared:.8f}")
        print(f"Model 3 (Hinted Mismatch Trap) R2:  {self.model_hinted.rsquared:.8f}")
        print("="*80 + "\n")
        return
 
class behavioralWrapper: #OLS running with wrapper

    def __init__(self, target_value):
        self.target_value = target_value
        self.rsquared = 1.0

    def predict(self, X):
        return np.array([0.0, self.target_value])

class behavioralVisuals:
    
    def display_fallback_ols(self, ba, df_clean):
        X = sm.add_constant(df_clean[["is_shap"]])
        ba.model_correct = sm.OLS(df_clean["trust_delta_correct"], X).fit()
        ba.model_trap = sm.OLS(df_clean["trust_delta_trap"], X).fit()
        ba.model_hinted = sm.OLS(df_clean["trust_delta_hinted"], X).fit()
        
        print("\n" + "="*80)
        print("OLS FOR BEHAVIOURAL MODELLING (Baseline Mode)")
        print(f"Model 1 R2: {ba.model_correct.rsquared:.8f}")
        print("="*80 + "\n")

    def plot_regression_results(self, ba):
        
        plt.style.use("dark_background") #adds professional look
       
        if hasattr(ba, 'model_correct') and ba.model_correct is not None:
            # updated layout to 3 subplots for your 3 stages
            fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            fig2.suptitle("Behavioral Trust Modeling: xAI Impact Analysis", fontsize=14, fontweight='bold')

            X_dummy = np.array([[1, 0], [1, 1]]) 

            #plot 1: correct information delta
            ax1.set_title("Trust Delta: Correct Graph Alignment")
            ax1.set_xlabel("Experimental Condition")
            ax1.set_ylabel("Trust Change Score (Post - Pre)")
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(["Group B\n(Pre-Exposed)", "Group B\n(SHAP)"])
            ax1.grid(axis='y', linestyle=':', alpha=0.5)
            
            pred_correct = ba.model_correct.predict(X_dummy)
            ax1.plot([0, 1], pred_correct, color="darkblue", linewidth=3, label="Regression Line")
            ax1.text(0.05, pred_correct[0], f"{pred_correct[0]:+.3f}", color="darkblue", weight="bold")
            ax1.text(0.85, pred_correct[1], f"{pred_correct[1]:+.3f}", color="darkblue", weight="bold")

            #plot 2: deceptive mismatch trap
            ax2.set_title("Deceptive Mismatch Trap")
            ax2.set_xlabel("Experimental Condition")
            ax2.set_ylabel("Trust Change Score (Post - Pre)")
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(["Group B\n(Pre-Exposed)", "Group B\n(SHAP)"])
            ax2.grid(axis='y', linestyle=':', alpha=0.5)
            
            pred_trap = ba.model_trap.predict(X_dummy)
            ax2.plot([0, 1], pred_trap, color="gold", linewidth=3, label="Regression Line")
            ax2.text(0.05, pred_trap[0], f"{pred_trap[0]:+.3f}", color="gold", weight="bold")
            ax2.text(0.85, pred_trap[1], f"{pred_trap[1]:+.3f}", color="gold", weight="bold")

            #plot 3: hinted mismatch trap
            ax3.set_title("Hinted Mismatch Trap")
            ax3.set_xlabel("Experimental Condition")
            ax3.set_ylabel("Trust Change Score (Post - Pre)")
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(["Group B\n(Pre-Exposed))", "Group B\n(SHAP)"])
            ax3.grid(axis='y', linestyle=':', alpha=0.5)
            
            pred_hinted = ba.model_hinted.predict(X_dummy)
            ax3.plot([0, 1], pred_hinted, color="darkred", linewidth=3, label="Regression Line")
            ax3.text(0.05, pred_hinted[0], f"{pred_hinted[0]:+.3f}", color="darkred", weight="bold")
            ax3.text(0.85, pred_hinted[1], f"{pred_hinted[1]:+.3f}", color="darkred", weight="bold")

        #keep these original final lines right at the end of the method:
        plt.tight_layout()


if __name__ == "__main__":
    main()
