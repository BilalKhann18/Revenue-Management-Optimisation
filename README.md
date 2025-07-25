# 📦 Revenue Management Optimisation for BuildMax Rentals  
**Warwick Business School | Pricing Analytics (IB9EO0)**  
**Project Duration:** Oct 2024 – Mar 2025  

## 🧾 Project Overview  
This project presents a linear programming model designed to optimise equipment rental revenue for **BuildMax Rentals**, a heavy machinery rental company operating in North America and Europe. Using UK branch data, we assessed BuildMax’s suitability for revenue management (RM) and developed a model to enhance decision-making for rental allocation and pricing.

## 🎯 Objectives  
- Evaluate the feasibility of implementing Revenue Management (RM) at BuildMax  
- Develop a mathematical model using Python and Pyomo  
- Analyse performance across revenue, ROI, fleet utilisation, and RPU  
- Provide implementation recommendations and identify limitations

## 📐 Model Formulation  
The model maximises weekly revenue based on accepted rentals:  
Maximise: ∑ₑ ∑ₜ ∑𝓌 (Pₑ,ₜ,𝓌 × xₑ,ₜ,𝓌 × dₜ × 7)


**Decision Variable:**  
- `x[e,t,w]`: Number of rentals accepted for equipment type `e`, rental duration `t`, week `w`  

**Constraints:**  
- Demand ≤ customer requests  
- Inventory updated dynamically each week  
- No overcommitment on fleet

## 📊 Key Results  
- 💸 **Revenue Growth:** £160.47M → £178.47M (+11.22%)  
- 📈 **ROI Increase:** 264.13% → 304.99%  
- 🚜 **Utilisation Rate:** 39.44% → 60.33%  
- 📊 **RPU Gains:** Up to +3.39% across equipment types

## 🛠 Tech Stack  
- `Python`, `Pyomo` (linear programming)  
- `Excel` (data source)  
- `Matplotlib` (visual analysis)  

## 🔍 Strategic Insights  
- Cranes and excavators offered the greatest optimisation potential  
- Short-term rental prioritisation boosts ROI, while long-term contracts offer stability  
- Predictive maintenance and inter-branch reallocation could further enhance efficiency  

## ⚠️ Limitations  
- Model assumes fixed inventory and static competitor pricing  
- Customer brand preferences and real-time logistics not accounted for  
- Extensions could include dynamic pricing or machine learning-based demand forecasting  

