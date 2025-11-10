\# 💳 AIFUL Credit Risk Analyzer — IIT Mandi × AiFul Japan  

\### \*“Explainable • Ethical • Actionable AI for Smarter Lending”\*  



---



\## 🏆 AIHack India 2025 — Kaggle Competition  

\*\*Team:\*\* TWO MONKS (Shruti \& Yash)  

\*\*Institution:\*\* IIT Mandi  

\*\*Competition Link:\*\* \[AIHack India 2025 (Kaggle)](https://www.kaggle.com/competitions/aihack-india-nov-2025/overview)



---



\## 🧩 Problem Statement



Financial institutions struggle to accurately assess borrower creditworthiness — especially in \*\*unsecured loans\*\*, where there’s no collateral.  

This leads to:

\- Biased or inaccurate lending decisions  

\- Financial losses due to defaults  

\- Reduced trust in the credit system  



Our task:  

> Build a predictive and explainable AI model that classifies whether a borrower is likely to default, ensuring \*\*data-driven, fair, and responsible lending\*\*.



---



\## ⚙️ Challenges We Tackled



| Challenge | Description |

|------------|-------------|

| \*\*Unbalanced Data\*\* | Default cases were far fewer, risking model bias |

| \*\*No Collateral\*\* | Relied on behavioral and financial signals instead |

| \*\*High Dimensionality\*\* | Managed large heterogeneous data efficiently |

| \*\*Ethical Lending\*\* | Ensured fair predictions across demographics |



---



\## 🎯 Project Goal



Develop an end-to-end AI platform that:

\- Predicts \*\*default probability\*\* with high accuracy  

\- Provides \*\*explainable insights\*\* (via SHAP)  

\- Simulates \*\*“what-if” lending scenarios\*\*  

\- Promotes \*\*fairness \& inclusion\*\* in financial systems  



---



\## 🧠 Technical Approach



\### 🔹 Model Architecture

We experimented with:

\- \*\*LightGBM:\*\* High AUC, great for tabular credit data  

\- \*\*CatBoost:\*\* Handles categorical features smoothly  

\- \*\*XGBoost:\*\* Reliable baseline for comparison  



The final ensemble combined \*\*LightGBM + CatBoost\*\*, achieving the best trade-off between \*\*accuracy and interpretability\*\*.



---



\## 📊 Insight Discovery



\### 🧮 Loan-to-Income Ratio — \*The Strongest Default Indicator\*

Borrowers with \*\*Loan-to-Income ratio > 0.6\*\* were \*\*2× more likely to default.\*\*  

→ We propose setting a \*safe lending threshold\* at \*\*0.6\*\*.



\### 👨‍👩‍👧 Dependents Increase Financial Stress

More dependents = less disposable income = higher default probability.  

→ Use dependents as a \*financial stress multiplier\* in approval scoring.



\### 💼 Employment Duration = Stability

Stable employment leads to predictable repayment.  

→ Integrate a \*stability score\* based on employment tenure.



\### 👶 Age Follows a U-Shaped Risk Curve

Younger (<25) and older (>55) borrowers show more volatility.  

→ Tailor loan education or support plans for these groups.



---



\## 💡 Business View — Turning AI Insights into Strategy



| Insight | Actionable Strategy |

|----------|--------------------|

| High Loan-to-Income (>0.6) | Flag high-risk applicants or reduce credit limits |

| Short Employment Duration (<1 yr) | Trigger manual review |

| Multiple Dependents (≥3) | Adjust affordability score |

| High Desired Limit Requests | Use AI-driven dynamic limit recommendations |

| Fairness Score (90/100) | Ensure behavior-based lending, not demographic bias |



> “AIFUL can now make \*\*faster, fairer, and smarter\*\* lending decisions, powered by an explainable AI system that connects every prediction to a business action.”



---



\## 🧰 Tech Stack



| Layer | Tools Used |

|--------|-------------|

| Frontend | Streamlit |

| Data | Pandas, NumPy |

| ML Models | scikit-learn, LightGBM, CatBoost, XGBoost |

| Explainability | SHAP |

| Visualization | Plotly |

| Reporting | FPDF |

| Fairness Analysis | Custom gender \& marital bias evaluation |



---



\## 📸 Screenshots



| Portfolio Dashboard | Customer Insights |

|----------------------|-------------------|

| !\[Dashboard](screenshot1.png) | !\[Customer Insights](screenshot2.png) |



| Fairness \& Ethics | Business Intelligence |

|-------------------|-----------------------|

| !\[Fairness](screenshot3.png) | !\[Business Intelligence](screenshot4.png) |



---



\## 🧮 Features Overview



| Module | Description |

|---------|-------------|

| \*\*📊 Portfolio Dashboard\*\* | Portfolio-level KPIs, risk distributions, top correlated features |

| \*\*👤 Customer Insights\*\* | Individual customer analysis, “What-If” simulator, SHAP interpretation |

| \*\*⚖️ Fairness \& Ethics\*\* | Demographic bias detection (Gender × Marital Status), Fairness Score |

| \*\*🧮 Business Intelligence\*\* | Loan-to-Income risk analysis, customer clustering (K-Means) |

| \*\*📄 Credit Health Report (PDF)\*\* | Auto-generated report summarizing customer’s credit health |



---



\## 🚀 Installation \& Execution



Clone the repository:

```bash

git clone https://github.com/pratap424/AIFUL-Credit-Risk-Analyzer.git

cd AIFUL-Credit-Risk-Analyzer



