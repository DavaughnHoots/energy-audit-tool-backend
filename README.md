---

## Energy Audit Tool Prototype

### Overview

The Energy Audit Tool is a Python-based prototype designed to analyze and calculate energy consumption for various scenarios. By inputting specific parameters related to energy usage, the tool provides detailed insights into energy consumption patterns, adjusted for real-world factors such as seasonal changes and occupancy levels.

### Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Input Parameters](#input-parameters)
    - [1. Power Consumed (kW)](#1-power-consumed-kw)
    - [2. Duration (Hours)](#2-duration-hours)
    - [3. Seasonal Adjustment Factor (Fs)](#3-seasonal-adjustment-factor-fs)
    - [4. Occupancy Adjustment Factor (Fo)](#4-occupancy-adjustment-factor-fo)
    - [5. Power Factor (PF)](#5-power-factor-pf)
5. [Understanding the Inputs](#understanding-the-inputs)
    - [Power Consumption (P)](#power-consumption-p)
    - [Duration (t)](#duration-t)
    - [Seasonal Adjustment Factor (Fs)](#seasonal-adjustment-factor-fs)
    - [Occupancy Adjustment Factor (Fo)](#occupancy-adjustment-factor-fo)
    - [Power Factor (PF)](#power-factor-pf)
6. [Extending the Tool](#extending-the-tool)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

### Input Parameters

To perform an energy audit using this tool, users need to provide the following input parameters:

1. **Power Consumed (kW)**
2. **Duration (Hours)**
3. **Seasonal Adjustment Factor (Fs)**
4. **Occupancy Adjustment Factor (Fo)**
5. **Power Factor (PF)**

Each parameter is essential for calculating accurate energy consumption and understanding the factors influencing it. Below is a detailed explanation of each input, including how to obtain and understand the required information.

#### 1. Power Consumed (kW)

- **Description**: The total power consumed by the equipment or building, measured in kilowatts (kW).
- **How to Obtain**:
  - **Power Meters**: Install high-resolution power meters to measure real-time power consumption.
  - **Utility Bills**: Refer to electricity bills which often indicate peak and average power usage.
  - **Manufacturer Specifications**: Check the specifications of electrical equipment for their rated power consumption.
- **Learning Resources**:
  - [Understanding Electrical Power](https://www.electronics-tutorials.ws/power/power_1.html)
  - [How to Read a Power Meter](https://www.energy.gov/energysaver/reading-your-electric-meter)

#### 2. Duration (Hours)

- **Description**: The total time (in hours) that the equipment or building operates within a specific period.
- **How to Obtain**:
  - **Operational Logs**: Maintain logs of operational hours for equipment.
  - **Building Schedules**: Use building management schedules to determine active hours.
  - **Automated Systems**: Implement automated tracking systems that record usage durations.
- **Learning Resources**:
  - [Energy Management Best Practices](https://www.energy.gov/eere/education/downloads/energy-management-best-practices)
  - [Time Tracking for Energy Audits](https://www.energy.gov/eere/buildings/articles/time-tracking-energy-audits)

#### 3. Seasonal Adjustment Factor (Fs)

- **Description**: A multiplier that adjusts energy consumption based on seasonal variations, typically ranging from 0.8 to 1.2.
- **How to Obtain**:
  - **Historical Climate Data**: Analyze past weather data to determine how different seasons affect energy usage.
  - **Energy Consumption Trends**: Review historical energy consumption records to identify seasonal patterns.
  - **Industry Standards**: Refer to industry-specific guidelines for typical seasonal factors.
- **Learning Resources**:
  - [Seasonal Energy Efficiency Ratio (SEER)](https://www.energy.gov/eere/buildings/articles/seasonal-energy-efficiency-ratio-seer)
  - [Analyzing Seasonal Energy Consumption](https://www.energy.gov/eere/buildings/articles/analyzing-seasonal-energy-consumption)

#### 4. Occupancy Adjustment Factor (Fo)

- **Description**: A multiplier that adjusts energy consumption based on occupancy levels, typically ranging from 0.6 to 1.0.
- **How to Obtain**:
  - **Occupancy Sensors**: Use sensors to monitor real-time occupancy levels.
  - **Building Usage Data**: Analyze data on building usage patterns, such as number of occupants and their activity levels.
  - **Surveys and Studies**: Conduct surveys or studies to estimate typical occupancy rates.
- **Learning Resources**:
  - [Occupancy-Based Energy Management](https://www.energy.gov/eere/buildings/articles/occupancy-based-energy-management)
  - [Building Occupancy Patterns and Energy Use](https://www.sciencedirect.com/science/article/pii/S0301421516307124)

#### 5. Power Factor (PF)

- **Description**: A measure of how effectively electrical power is being used, typically ranging from 0.8 to 0.95.
- **How to Obtain**:
  - **Power Quality Meters**: Install power quality meters to measure the power factor in real-time.
  - **Electrical System Analysis**: Perform an analysis of the electrical system to determine the power factor.
  - **Utility Company Data**: Some utility providers supply power factor information for large installations.
- **Learning Resources**:
  - [Understanding Power Factor](https://www.electrical4u.net/electrical-power-factor/)
  - [Power Factor Correction Techniques](https://www.electronics-tutorials.ws/power/power-factor.html)

---

### Understanding the Inputs

To effectively utilize the Energy Audit Tool, it's crucial to understand each input parameter and its impact on energy consumption calculations.

#### Power Consumption (P)

**Definition**: Power consumption refers to the rate at which electrical energy is used by equipment or a facility, measured in kilowatts (kW).

**Importance**: Accurate measurement of power consumption is fundamental to assessing energy usage and identifying areas for improvement.

**Resources**:
- [Electrical Power Basics](https://www.electronics-tutorials.ws/power/power_1.html)
- [Measuring Power Consumption](https://www.homeenergy.org/show/article/magazine/141/page/22/id/1028)

#### Duration (t)

**Definition**: Duration indicates the total time in hours that the equipment operates within a given period.

**Importance**: Understanding operational hours helps in estimating total energy consumption and identifying peak usage times.

**Resources**:
- [Time Management in Energy Audits](https://www.energy.gov/eere/buildings/articles/time-management-energy-audits)
- [Tracking Operational Hours](https://www.buildinggreen.com/blog/tracking-operational-hours-energy-management)

#### Seasonal Adjustment Factor (Fs)

**Definition**: The seasonal adjustment factor accounts for variations in energy consumption due to seasonal changes, such as heating in winter or cooling in summer.

**Importance**: Adjusting for seasonal factors provides a more accurate representation of energy usage patterns and helps in planning energy-saving measures.

**Resources**:
- [Seasonal Energy Consumption Analysis](https://www.energy.gov/eere/buildings/articles/seasonal-energy-consumption-analysis)
- [Impact of Seasons on Energy Use](https://www.sciencedirect.com/science/article/pii/S030142151730792X)

#### Occupancy Adjustment Factor (Fo)

**Definition**: The occupancy adjustment factor modifies energy consumption estimates based on the number of occupants and their activity levels.

**Importance**: Occupancy levels significantly influence energy usage, especially in commercial buildings, making this factor essential for accurate audits.

**Resources**:
- [Occupancy-Based Energy Modeling](https://www.energy.gov/eere/buildings/articles/occupancy-based-energy-modeling)
- [Impact of Occupancy on Building Energy Use](https://www.sciencedirect.com/science/article/pii/S0378778815007433)

#### Power Factor (PF)

**Definition**: The power factor is the ratio of real power used to do work to the apparent power flowing in the circuit, indicating the efficiency of power usage.

**Importance**: A low power factor signifies inefficient energy usage, leading to higher electricity costs and potential penalties from utility providers.

**Resources**:
- [Power Factor Explained](https://www.electrical4u.net/electrical-power-factor/)
- [Improving Power Factor](https://www.energy.gov/sites/prod/files/2014/04/f15/power_factor_correction.pdf)

---

### Getting Started

To begin using the Energy Audit Tool, follow the installation and usage instructions below.

#### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/energy-audit-tool.git
    cd energy-audit-tool
    ```

2. **Install Dependencies**

    Ensure you have Python 3.x installed. Install required packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: If you have additional dependencies, list them in a `requirements.txt` file.)*

#### Usage

1. **Run the Tool**

    ```bash
    python energy_audit_tool.py
    ```

2. **Input Parameters**

    When prompted, input the required parameters as described in the [Input Parameters](#input-parameters) section.

3. **View Results**

    The tool will display the calculated energy consumption values based on your inputs.

---

### Extending the Tool

TODO features (A javascript prototype test is being developed at the [following location](https://github.com/DavaughnHoots/ESS-energy-audit-prototype)):

- **User Interface**: Develop a graphical user interface (GUI) for easier interaction.
- **Data Persistence**: Save audit results to files (e.g., CSV, JSON) for future reference.
- **Real-Time Data Integration**: Connect with real-time data sources for dynamic monitoring.
- **Visualization**: Incorporate graphs and charts to visualize energy consumption trends.

---

### Contributing

Contributions are welcome! Please open issues and submit pull requests for any enhancements or bug fixes.

---

### License

This project is licensed under the [MIT License](LICENSE).

---

### Contact

For any questions or suggestions, please contact [hootsd1@montclair.edu](mailto:hootsd1@montclair.edu).

---
