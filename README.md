# Classification-Credit-Score

This project focuses on building a classification model to predict credit scores based on the provided dataset.

## Project Structure

- `credit_score.csv`: The dataset used for training and testing the model.
- `credit_score.ipynb`: Jupyter Notebook containing the data analysis, preprocessing, and model implementation.
- `README.md`: Documentation file describing the project.
- `requirements.txt`: Lists the Python dependencies required for the project.


## Files and Directories

- **`credit_score.csv`**: The dataset used for training and testing the model.
- **`credit_score.ipynb`**: Jupyter Notebook containing the data analysis, preprocessing, and model implementation.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`env/`**: Virtual environment directory containing the Python environment and installed packages.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory and activate the virtual environment:
   ```sh
   source env/Scripts/activate  # On Windows
   source env/bin/activate      # On macOS/Linux
3. Install the required dependencies
   '''sh
   pip install -r requirements.txt

## Machine Learning Algorithms and Results

| Algorithm               | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| Logistic Regression     | 85.2%   | 84.5%     | 83.8%  | 84.1%    |
| Random Forest           | 91.3%   | 90.8%     | 91.0%  | 90.9%    |
| Support Vector Machine  | 88.7%   | 88.2%     | 87.9%  | 88.0%    |
| Gradient Boosting       | 92.5%   | 92.0%     | 91.8%  | 91.9%    |

## License

This project is licensed under the MIT License.
