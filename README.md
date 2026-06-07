# xAI-trading-algorithmic-study-for-SHAP-v.s-ELI5

An open-source Python algorithm developed for the technical SHAP and ELI5 evaluation components of the research project: **"Improving Algorithmic Transparency: A Comparative Study of SHAP and ELI5 for Younger Gen Z Retail-Level Investors."**

🔗 **Official Science Fair Registry & Project Dashboard:** [View CYSF Exhibition Layout](https://platform.cysf.org/project/482b870c-ef81-4680-94bd-d499b80d7acf/)

---

## 📂 Repository Directory
* **`shapver_engine.py`** — The Quantitative-Visual testing environment utilizing `shap.KernelExplainer` matrix arrays to map statistically grounded feature weights.
* **`eli5ver_engine.py`** — The Qualitative-Narrative testing environment utilizing automated natural language wrappers to generate contextual text explanations.

## 🔬 Core Research Abstract & Key Finding
As retail-level algorithmic trading access expands among younger demographic cohorts, mitigating automation bias—the tendency to blindly trust machine outputs—becomes a critical user-safety metric. This study developed an interactive software testbed to analyze how different explainable AI (xAI) architectures calibrate user trust. 

Empirical human-subject testing ($n=58$) revealed a distinct **Interpretability Paradox**: Although ELI5 can enable a faster absorption of metrics and applications, it holds a low ceiling in which comprehensibility is stunted by the **“Illusionary Explanation of Depth.”** This study proves that although simplicity may seem like the best route to pursue when making AI accessible to younger Gen Z, it can inevitably be a liability due to its low ceiling. Meanwhile, SHAP may initially seem like a barrier to accessibility due to its complex composition in visual form, but it presents a pathway towards **“Desirable Difficulty,”** forcing one to critically think by converting the visual into an interpretable explanation. Its high ceiling is what financial literacy requires, yet at times may be challenging to achieve. Yet, with the right tools, like a simplified glossary and exposure, this problem can be mitigated. To conclude, for short-term interaction, ELI5 is an excellent tool; however, for long-term comprehension, SHAP is necessary.

## 🛠️ Technical Stack
* **Language:** Python 3.10
* **Data Processing & Math:** Pandas, NumPy
* **Explainable AI Engines:** SHAP (Lundberg et al.), ELI5 Wrapper
* **Data Stream Integration:** yfinance, Tabulate
* **Graphics & Event Handling Canvas:** Matplotlib (TkAgg Back-end)

## 🏆 Awards & Academic Presentations
* **Silver Medalist**, Calgary Youth Science Fair (CYSF)
