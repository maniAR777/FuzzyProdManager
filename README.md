# FuzzyProdManager

###  **Production and Maintenance Optimization Using Fuzzy Logic**

This code implements a **fuzzy optimization model** to balance production, inventory, backlog, and maintenance costs over multiple periods. The goal is to identify the best combination of production and maintenance plans under uncertain demand using **fuzzy triangular numbers**.

---

### **Key Components and Workflow**

#### **1. Triangular Fuzzy Numbers**
- A **triangular fuzzy number** represents uncertain values with three points: 
  - **l** (lower bound), **m** (most probable value), and **r** (upper bound).
- Methods:
  - **Jimenez Defuzzification**: Converts a fuzzy number into a crisp value for optimization.
  - **E1 and E2 Calculation**: Used for α-level defuzzification, enabling decision-making under different levels of uncertainty.

#### **2. Input Data**
- Data includes:
  - **Demand**: Uncertain (fuzzy) values for each period.
  - **Production Capacities**: Normal and overtime production capacities.
  - **Costs**: For production, inventory, backlog, and maintenance.
  - **Storage Capacity**: Maximum allowable inventory for each period.

#### **3. Fuzzy Demand and Defuzzification**
- Demand is modeled as a triangular fuzzy number for each period.
- Using **Jimenez defuzzification** and **α-level defuzzification**, fuzzy demand is converted into crisp values for use in the optimization model.

#### **4. Optimization Model**
- **Decision Variables**:
  - Regular production (`P_ri`), overtime production (`P_oi`), inventory (`Inv_it`), backlog (`B_it`), and maintenance plans (`PM_t`).
- **Objective Function**:
  - Minimize a weighted sum of production, inventory, backlog, and maintenance costs.
- **Constraints**:
  - Satisfy demand (defuzzified based on α).
  - Limit production within capacity (adjusted for maintenance downtime).
  - Maintain a minimum inventory level to avoid shortages.
  - Enforce required maintenance periods when actual usage exceeds a threshold.

#### **5. Sensitivity Analysis Using α**
- The model is solved iteratively for α values ranging from 0.1 to 1.0.
- For each α:
  - The fuzzy demand is adjusted.
  - Optimal production and maintenance plans are determined.
  - Results are stored for comparison.

#### **6. Result Analysis**
- **Key Outputs**:
  - Optimal production and maintenance plans for each period.
  - Inventory and backlog levels.
  - Total cost (objective value) for each α.
- **Performance Metrics**:
  - **Membership Value (`μ_D`)**: Combines fuzzy satisfaction levels for production and maintenance costs using a T-norm.

---

### **Visualizations**
1. **Objective Cost vs. Alpha**: Illustrates how costs change with different α values, showing the impact of uncertainty levels.
2. **Production vs. Demand**: Compares production levels (regular and overtime) against demand for the best α.
3. **Inventory and Backlog Levels**: Shows the trade-off between inventory and backlog for each period.
4. **Maintenance Plan**: Displays planned maintenance actions over time.
5. **Period-Level Results**: Includes bar charts for inventory, backlog, and maintenance.

---

### **Key Insights**
1. **Best α Selection**:
   - The α value with the highest membership value (`μ_D`) is chosen as the optimal balance between production and maintenance costs.
2. **Maintenance Planning**:
   - Maintenance is prioritized for periods with high actual usage, balancing production downtime and long-term system reliability.
3. **Fuzzy Demand Handling**:
   - The model adapts to uncertainty in demand, providing robust plans across different α levels.

---

### **Applications**
This model is suitable for industries where:
- **Demand Uncertainty**: Demand fluctuates significantly and must be modeled as fuzzy data.
- **Integrated Decisions**: Production and maintenance decisions need to be optimized together.
- **Sensitivity Analysis**: Understanding how varying levels of uncertainty impact operational costs is critical.

The code ensures a **flexible and robust decision-making framework** for real-world production and maintenance challenges under uncertainty.
