import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm

"""
STATISTICAL REGRESSION PREP FOR ELI5VER
--------------------------------
Analyzes Google Forms results to mathematically calculate the impact of xAI frameworks on younger Gen Z delta trust and mental model accuracy


ELI5VER: The Qualitative-Narrative
--------------------------------
As a natural language contrast, it provides an explanation considering relevant features of input that contribute to the outcome of the model. 
This method is reliant on other xAI methods like SHAP, Permutation feature importance and LIME for explanation generation.
"""

def main():
 
    ba = behavioralAnalysis() #OLS plotting
    bv = behavioralVisuals()

    ba.get_file("eli5_data.csv") #gets google forms results
    ba.run_eli5_ols()

    # Displaying graph
    bv.plot_regression_results(ba)
    plt.show()


class behavioralAnalysis:

    def get_file(self, csv_path="eli5_data.csv"):
        self.csv_path = csv_path

        #set up models as None to avoid AttributeError Later
        self.model_correct = None
        self.model_trap = None
        self.model_hinted = None

    def check_group(self, x): #helper function
        if str(x).upper() == "B" or "ELI5" in str(x).upper():
            return 1 #group B get 1
        else:
            return 0 #group A gets 0

    def run_eli5_ols(self):
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

        df_survey["is_eli5"] = df_survey["post_correct"].notna().astype(int)

        #delta 1: shift after accurate xAI explanation viewing
        df_survey["trust_delta_correct"] = df_survey["post_correct"] - df_survey["pre_trust"]

        #delta 2: shift after intentionally wrong xAI prediction viewing
        df_survey["trust_delta_trap"] = df_survey["post_trap"] - df_survey["pre_trust"]

        #delta 3: shift after you actively hint to them that the AI's data is wrong
        df_survey["trust_delta_hinted"] = df_survey["post_trap_hinted"] - df_survey["pre_trust"]

        #clean incomplete tracking rows
       #clean incomplete baseline tracking rows
        df_clean = df_survey.dropna(subset=["pre_trust"]).copy()

        self.min_trust = df_clean["pre_trust"].min()
        self.max_trust = df_clean["pre_trust"].max()

        X_correct = sm.add_constant(df_clean[["pre_trust"]])
        self.model_correct = sm.OLS(
            df_clean["trust_delta_correct"],
            X_correct,
            missing="drop"
        ).fit()

        self.model_trap = sm.OLS(
            df_clean["trust_delta_trap"],
            X_correct,
            missing="drop"
        ).fit()

        self.model_hinted = sm.OLS(
            df_clean["trust_delta_hinted"],
            X_correct,
            missing="drop"
        ).fit()


        print("\nMODEL 1 DETAILS")
        print(self.model_correct.summary())

        print("\nMODEL 2 DETAILS")
        print(self.model_trap.summary())

        print("\nMODEL 3 DETAILS")
        print(self.model_hinted.summary())
        return

class behavioralVisuals:

    def plot_regression_results(self, ba):
        
        plt.style.use("dark_background") #adds professional look
       
        if hasattr(ba, 'model_correct') and ba.model_correct is not None:
            # updated layout to 3 subplots for your 3 stages
            fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            fig2.suptitle("Behavioral Trust Modeling: xAI Trust Calibration Analysis", fontsize=14, fontweight='bold')

            trust_values = np.linspace(ba.min_trust, ba.max_trust, 100)

            X_dummy = sm.add_constant(
            pd.DataFrame({"pre_trust": trust_values})
            )

            #plot 1: correct information delta
            ax1.set_title("Trust Delta: Correct Graph Alignment")
            ax1.set_xlabel("Baseline Trust Score")
            ax1.set_ylabel("Trust Change Score (Post - Pre)")
            ax1.grid(axis='y', linestyle=':', alpha=0.5)
            
            pred_correct = ba.model_correct.predict(X_dummy)
            ax1.plot(trust_values, pred_correct, color="darkblue", linewidth=3, label="Regression Line")
            ax1.text(trust_values[0], pred_correct.iloc[0], f"{pred_correct.iloc[0]:+.3f}", color="darkblue", weight="bold")
            ax1.text(trust_values[-1], pred_correct.iloc[-1], f"{pred_correct.iloc[-1]:+.3f}", color="darkblue", weight="bold")

            #plot 2: deceptive mismatch trap
            ax2.set_title("Deceptive Mismatch Trap")
            ax2.set_xlabel("Baseline Trust Score")
            ax2.set_ylabel("Trust Change Score (Post - Pre)")
            ax2.grid(axis='y', linestyle=':', alpha=0.5)
            
            pred_trap = ba.model_trap.predict(X_dummy)
            ax2.plot(trust_values, pred_trap, color="gold", linewidth=3, label="Regression Line")
            ax2.text(trust_values[0], pred_trap.iloc[0], f"{pred_trap.iloc[0]:+.3f}", color="gold", weight="bold")
            ax2.text(trust_values[-1], pred_trap.iloc[-1], f"{pred_trap.iloc[-1]:+.3f}", color="gold", weight="bold")

            #plot 3: hinted mismatch trapf
            ax3.set_title("Hinted Mismatch Trap")
            ax3.set_xlabel("Baseline Trust Score")
            ax3.set_ylabel("Trust Change Score (Post - Pre)")

            ax3.grid(axis='y', linestyle=':', alpha=0.5)
            
            pred_hinted = ba.model_hinted.predict(X_dummy)
            ax3.plot(trust_values, pred_hinted, color="darkred", linewidth=3, label="Regression Line")
            ax3.text(trust_values[0], pred_hinted.iloc[0], f"{pred_hinted.iloc[0]:+.3f}", color="darkred", weight="bold")
            ax3.text(trust_values[-1], pred_hinted.iloc[-1], f"{pred_hinted.iloc[-1]:+.3f}", color="darkred", weight="bold")

        #keep these original final lines right at the end of the method:
        plt.tight_layout()


if __name__ == "__main__":
    main()
