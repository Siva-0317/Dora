# Intelligent Pesticide Sprinkling System

## Problem Statement
Traditional pesticide application systems often lead to excessive chemical use, soil degradation, water pollution, and poor crop health due to indiscriminate spraying irrespective of plant conditions. Over **98% of pesticides never reach the target**.

The **Intelligent Pesticide Sprinkling System** addresses this problem with a smart, real-time solution that treats only the plants that actually need it—using just the right amount of pesticides, exactly where and when they're needed.

---

## System Overview

### Triple-Tank Setup
- **Central Tank:** Water  
- **Two Side Tanks:** Concentrated pest and disease medication  

### Sensing & Control
- **RGB camera** affixed to the spray nozzle monitors plants and transmits data to a **Raspberry Pi**  
- **Arduino UNO R4** communicates with the Raspberry Pi via **MQTT protocol**  
- **Solenoid valves** are actuated to control flow rate and mixing proportion of pesticides  
- **Flow sensors** and **pressure pumps** ensure precise spraying through the nozzle only when required  

### AI & Decision Making
A **5-head neural network architecture** deployed on the Raspberry Pi performs:
- **3-way classification:** healthy / pest / diseased  
- **Pixel-level lesion segmentation**  
- **Infection severity quantification**  
- **Precise dose computation** for insecticide/fungicide and water ratio  

---

## Key Innovations
- Real-time computer vision for **high accuracy in field conditions**  
- **Retrofit compatibility** with existing spraying equipment  
- **Bilingual web dashboard** for comprehensive data visualization  
- Reduces pesticide usage by **20–25%**  
- Improves crop yields by **10–15%**  
- Eliminates **manual scouting requirements**  
- Ensures accurate **pesticide mixture proportions**, preventing over-mixing and dilution  

---

## Impact
- Aligns with global **SDGs 2 and 12**  
- Promotes **environmentally responsible pest and disease control**  
- Contributes to better crop yield and **safe food production at affordable cost**  
- **Easy to use** for farmers  
- Can be integrated as a **plug-in module** with existing solutions and rovers for complete automation  
