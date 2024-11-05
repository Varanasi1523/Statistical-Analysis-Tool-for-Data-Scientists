# Statistical Analysis Tool for Data Scientists
## Project Overview
The Statistical Analysis Tool for Data Scientists is a Streamlit-based application that provides a comprehensive set of statistical analysis capabilities for data scientists. It allows users to upload their data, perform various statistical analyses, and visualize the results. This tool aims to simplify the data exploration and analysis process, making it more accessible for data scientists and analysts.
## Key Features

### Data Overview: Provides a high-level summary of the uploaded dataset, including data types, missing values, and first few rows.
### Descriptive Statistics: Calculates and displays basic descriptive statistics such as mean, median, standard deviation, and quantiles for numerical columns.
### Inferential Statistics: Performs common statistical tests, including t-tests (independent and paired), chi-square test, and ANOVA.
### Regression Analysis: Supports linear regression, logistic regression, Lasso regression, and Ridge regression.
### Time Series Analysis: Includes components analysis, stationarity testing, correlation analysis (ACF and PACF), and ARIMA forecasting.
### Visualization: Utilizes Plotly to generate interactive and informative visualizations for the analysis results.

## Prerequisites

Python 3.10 or later
Streamlit
Pandas
Numpy
Scipy
Matplotlib
Seaborn
Statsmodels
Scikit-learn
Plotly
Yfinance (for stock data analysis)

## Installation and Setup

Clone the repository:
```bash
git clone https://github.com/Varanasi1523/Statistical-Analysis-Tool-for-Data-Scientists.git
```
Change to the project directory:
```bash
cd Statistical-Analysis-Tool-for-Data-Scientists
```
Create a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

Install the required dependencies:
pip install -r requirements.txt
```

## Usage

Start the Streamlit application:
streamlit run app.py

The application will open in your default web browser.
Upload your CSV file or select a stock to analyze.
Navigate through the different analysis types using the sidebar menu.
Interact with the visualizations and analysis results on the main screen.

## Contributing
We welcome contributions to the Statistical Analysis Tool for Data Scientists project. If you would like to contribute, please follow these steps:

### Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them.
Push your changes to your forked repository.
Submit a pull request to the main repository.

## License
This project is licensed under the MIT License.
## Contact
If you have any questions or feedback, please feel free to contact the project maintainers at [sindhuvaranasi1523@gmail.com].
