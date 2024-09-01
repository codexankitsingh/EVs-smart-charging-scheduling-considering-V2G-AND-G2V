# EV Smart Charging Scheduling: V2G and G2V Optimization

## Overview

This repository contains the implementation of the **Dynamic Stochastic Charging Optimization (DSCO) Framework** for Electric Vehicles (EVs). The DSCO framework is designed to optimize the charging and discharging operations of EVs, particularly focusing on Vehicle-to-Grid (V2G) and Grid-to-Vehicle (G2V) modes. It incorporates advanced algorithms such as **Henry Gas Solubility Optimization (HGSO)** and probabilistic models to enhance the efficiency of EV charging processes while considering real-time electricity tariffs (RTT).

## Key Features

- **Dynamic Scheduling:** The DSCO framework adjusts EV charging schedules in real-time, taking into account fluctuating electricity prices and grid demands. This helps minimize costs and prevents grid overloads.
  
- **V2G and G2V Integration:** Efficiently handles both charging (G2V) and discharging (V2G) operations, enabling EVs to not only draw power from the grid but also supply it back during peak demand periods.

- **Probabilistic Modeling:** Utilizes the 2m-Point Estimation Method (PEM) to manage uncertainties in EV behavior, such as arrival/departure times and energy requirements.

- **Optimization Algorithms:** The HGSO algorithm is employed for solving the assignment and scheduling problem, ensuring optimal resource allocation across multiple charging stations.

## Project Structure

- **Dataset:** Includes data for 15 Electric Vehicles (EVs) and their corresponding charging stations. Details such as arrival/departure times, energy consumption, and state of charge (SOC) are provided.

- **Scheduling Algorithm:** The implementation of the DSCO framework, which dynamically adjusts the charging schedule based on real-time data and optimization techniques.

- **Analysis and Results:** Comparative analysis between standard HGSO and the enhanced DSCO framework, highlighting the cost reductions and efficiency improvements achieved through the integration of RTT.

## Conclusion

The DSCO framework presents a robust solution to the challenges of EV charging in smart grids. By integrating real-time tariff data and probabilistic models, it significantly improves the economic and operational efficiency of charging stations. This project demonstrates the potential of advanced optimization techniques in enhancing the sustainability of electric mobility.

## Future Scope

- **Renewable Energy Integration:** Exploring the integration of solar and wind energy into the charging process to further reduce dependency on the grid.
- **Vehicle-to-Vehicle (V2V) Energy Transfer:** Investigating the potential for direct energy exchange between vehicles, adding a new layer of flexibility to energy management.

## Contributors

- **Ankit Kumar Singh** (2021-IMT-012)
- **Harsh Sharma** (2021-IMT-041)

## Supervisor

- **Dr. Avadh Kishor**, Department of Information Technology, ABV-IIITM Gwalior

---

This content provides a comprehensive overview of your project and can be further customized or expanded based on specific needs.
