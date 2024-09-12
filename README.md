
# AlphaCares-Insurance-Analytics


## Project Overview
This repository contains the analysis of historical insurance claim data for AlphaCare Insurance Solutions (ACIS). The project aims to optimize the marketing strategy and identify "low-risk" targets for potential premium reductions, ultimately attracting new clients.

## Business Objectives
1. Analyze historical insurance claim data
2. Optimize marketing strategy
3. Discover "low-risk" targets for premium reduction
4. Attract new clients

## Key Components
1. Exploratory Data Analysis (EDA)
2. A/B Hypothesis Testing
3. Statistical Modeling
4. Machine Learning for Premium Prediction


## Project Structure

```plaintext

TellCo-Telecom-Analysis/
    ├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # GitHub Actions
├── .gitignore              # files and folders to be ignored by git
├── requirements.txt        # contains dependencies for the project
├── README.md               # Documentation for the projects
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md                       # Description of notebooks directory 
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md                   # Description of scripts directory
    
```

## Setup and Installation
1. Clone this repository:
   ```
   git clone https://github.com/OL-YAD/AlphaCares-Insurance-Analytics.git
   cd AlphaCares-Insurance-Analytics
   ```
2. Create a virtual environment:
   ```
   python -m venv acis_env
   source acis_env/bin/activate  # On Windows, use `acis_env\Scripts\activate`
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Activate the virtual environment:
   ```
   source acis_env/bin/activate  # On Windows, use `acis_env\Scripts\activate`
   ```
2. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
3. Open the notebooks in the `notebooks/` directory to view or run the analyses.

## Contributing
This project is part of a 10 Academy KAIM2 Week 3 challenge. Contributions, suggestions, and feedback are welcome from fellow participants and mentors.



## Acknowledgments
- 10 Academy for providing the challenge and dataset

## Contact
For any queries regarding this project, please open an issue in this repository.