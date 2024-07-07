import pandas as pd
import numpy as np

# Sample test data for different scenarios
test_data = {
    'School': ['A', 'B', 'C', 'D', 'E'],
    'Test Scores': [65, 70, 80, 90, 85],
    'Graduation Rate': [0.8, 0.85, 0.9, 0.95, 0.92],
    'Attendance Rate': [0.9, 0.95, 0.85, 0.8, 0.88],
    'Funding': [1_000_000, 800_000, 1_200_000, 1_500_000, 1_100_000],
    'Staffing': [30, 25, 35, 40, 28],
    'Materials': [0.8, 0.7, 0.9, 1.0, 0.85],
    'Low-income': [0.5, 0.4, 0.6, 0.7, 0.45],
    'Special Needs': [0.1, 0.15, 0.05, 0.2, 0.12],
    'Enrollment': [500, 450, 600, 700, 480],
    'Community Events': [10, 8, 12, 15, 9],
    'Fundraising': [50_000, 40_000, 60_000, 70_000, 55_000],
    'Volunteer Hours': [500, 400, 600, 700, 550],
    'Building Age': [20, 25, 15, 10, 18],
    'Condition': [0.8, 0.7, 0.9, 1.0, 0.85],
    'Tech Resources': [0.8, 0.7, 0.9, 1.0, 0.85]
}

# Criteria weights
weights = {
    'Performance Levels': 0.3,
    'Resource Deficits': 0.2,
    'Student Demographics': 0.2,
    'Community Engagement': 0.15,
    'School Infrastructure Quality': 0.15
}

# Normalization function
def normalize(series):
    series = pd.Series(series)
    return (series - series.min()) / (series.max() - series.min())

# Function to calculate the overall score for each school
def calculate_scores(df):
    # Performance Levels Score
    df['Test Scores Norm'] = normalize(df['Test Scores'])
    df['Performance Score'] = 0.6 * df['Test Scores Norm'] + 0.3 * df['Graduation Rate'] + 0.1 * df['Attendance Rate']

    # Resource Deficits Score
    df['Funding Norm'] = normalize(df['Funding'])
    df['Staffing Norm'] = normalize(df['Staffing'])
    df['Materials Norm'] = df['Materials']  # Already between 0 and 1
    df['Resource Score'] = 0.5 * df['Funding Norm'] + 0.3 * df['Staffing Norm'] + 0.2 * df['Materials Norm']

    # Student Demographics Score
    df['Demographics Score'] = 0.5 * df['Low-income'] + 0.3 * df['Special Needs'] + 0.2 * normalize(df['Enrollment'])

    # Community Engagement Score
    df['Community Score'] = 0.4 * normalize(df['Community Events']) + 0.3 * normalize(df['Fundraising']) + 0.3 * normalize(df['Volunteer Hours'])

    # School Infrastructure Quality Score
    df['Infrastructure Score'] = 0.4 * normalize(df['Building Age']) + 0.3 * df['Condition'] + 0.3 * df['Tech Resources']

    # Overall Score
    df['Overall Score'] = (weights['Performance Levels'] * df['Performance Score'] +
                           weights['Resource Deficits'] * df['Resource Score'] +
                           weights['Student Demographics'] * df['Demographics Score'] +
                           weights['Community Engagement'] * df['Community Score'] +
                           weights['School Infrastructure Quality'] * df['Infrastructure Score'])

    # Rank schools by overall score
    df['Rank'] = df['Overall Score'].rank(ascending=False)
    
    return df

# Creating a DataFrame from the test data
df = pd.DataFrame(test_data)

# Calculate the scores
df = calculate_scores(df)

# Display the ranked list
ranked_schools = df.sort_values(by='Rank')
print(ranked_schools[['School', 'Overall Score', 'Rank']])

# Additional testing scenarios
def run_tests():
    # Test 1: Check if the highest overall score is correctly ranked
    assert ranked_schools.iloc[0]['School'] == 'D', "Test 1 Failed: School D should have the highest rank."
    
    # Test 2: Ensure normalization is working within expected ranges
    assert ranked_schools['Overall Score'].max() <= 1, "Test 2 Failed: Overall Score exceeds 1."
    assert ranked_schools['Overall Score'].min() >= 0, "Test 2 Failed: Overall Score is below 0."
    
    # Test 3: Verify correct handling of normalized data
    test_scores_norm = normalize(pd.Series(test_data['Test Scores']))
    assert np.allclose(df['Test Scores Norm'], test_scores_norm), "Test 3 Failed: Normalized Test Scores are incorrect."
    
    # Test 4: Check if all scores are calculated and assigned properly
    assert 'Overall Score' in df.columns, "Test 4 Failed: Overall Score is not calculated."
    assert 'Rank' in df.columns, "Test 4 Failed: Rank is not assigned."
    
    print("All tests passed successfully.")

# Run the tests
run_tests()
