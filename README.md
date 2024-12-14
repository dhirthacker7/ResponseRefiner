# Validator tool

## Project Overview
The project is designed to build a **Model Evaluation Tool** using Streamlit to evaluate test cases from the GAIA dataset against the OpenAI model. The tool allows users to select specific test cases, submit them to the model, and compare the results. It supports step modifications for incorrect responses and includes comprehensive feedback recording and visualization.

## Problem Statement
The project aims to develop a tool to streamline model evaluation using test cases from the GAIA dataset. The solution should enable real-time model response evaluation, visualization of the results, and the ability to iteratively improve the model's performance through user-guided modifications.

## Project Goals
Key tasks include:

- **Test Case Selection**: Allow users to select a specific test case from a validation file.
- **OpenAI Integration**: Send the selected test case and context data to the OpenAI model.
- **Response Comparison**: Compare the OpenAI response to the correct answer in the metadata.
- **User Feedback**: Allow users to modify incorrect responses by providing step-by-step guidance.
- **Performance Visualization**: Generate real-time visualizations to depict model performance.
- **Session Tracking**: Log all user interactions for each session for later analysis.

## Technologies Used
- **Streamlit**: Frontend application interface.
- **OpenAI API**: Model for generating answers.
- **Google Cloud Storage (GCS)**: Store and retrieve additional data files.
- **PostgreSQL - Cloud SQL**: Manage and store file bytecode.

## Data Sources
The main test case data is retrieved from the GAIA benchmark dataset, while supplementary data files such as PNG, MP3, and Excel files are stored in Google Cloud Storage.

## Pre-requisites
- **Python 3.6 or later**: [Download Python](https://www.python.org/downloads)
- **Google Cloud Credentials**: For accessing GCS and BigQuery.
- **OpenAI API Key**: For interacting with the OpenAI model.

## Project Structure
```markdown
├── .streamlit/                            # Streamlit configuration files
│   └── config.toml                         # Configuration file for customizing the Streamlit app
├── __pycache__/                            # Compiled Python cache files for optimized performance
├── images/                                 # Folder for storing project images and architecture diagrams
├── myenv/                                  # Local virtual environment for managing dependencies
├── pages/                                  # Contains multi-page implementation for the Streamlit app
│   ├── data.py                             # Streamlit page to display the data
│   └── dashboard.py                        # streamlit page for visualizations
├── validation/                             # Folder for storing context data 
├── .DS_Store                               # macOS-specific system file for storing folder attributes
├── .gitignore                              # Specifies files and directories to be excluded from version control
├── Architecture_Diagram.ipynb              # Jupyter notebook containing the architecture diagram - code 
├── LICENSE                                 # Licensing details for the project
├── README.md                               # Project documentation file
├── huggingface_data_extraction.py          # Script for extracting data from the HuggingFace GAIA benchmark dataset
├── requirements.txt                        # List of dependencies required for the project
├── validation_data.csv                     # CSV file containing validation test cases
├── validator.py                            # Main code file with cloud integration, GUI and openai api integration
```


## Instructions for Running Locally
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment**:  
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

3. **Install the requirements**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:  
   ```bash
   streamlit run dashboard.py
   ```

## Deployment
The Streamlit application is deployed on [Streamlit Cloud](https://streamlit.io/). 

## Documentation
- **CodeLabs documentation**: [CodeLabs]([https://huggingface.co/datasets/gaia-benchmark/GAIA](https://codelabs-preview.appspot.com/?file_id=19YlgUH63yH2j6AQJpKsnpPfrSXpMxl3QcXehr_eJwpY#0)).
