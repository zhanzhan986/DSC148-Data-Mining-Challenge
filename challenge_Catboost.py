import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def preprocess_amenities(df):
    # Removing special characters and splitting the amenities into lists
    df['amenities'] = df['amenities'].str.replace('[{}"]', '', regex=True)
    
    # Split the amenities string into a list of amenities
    df['amenities'] = df['amenities'].str.split(',')
    
    # Explode the DataFrame such that each amenity gets its own row (temporary DataFrame)
    temp_df = df.explode('amenities')
    
    # Create binary features for each amenity
    for amenity in temp_df['amenities'].unique():
        df[f'amenity_{amenity}'] = df['amenities'].apply(lambda x: 1 if amenity in x else 0)
    
    # Drop the original 'amenities' column as it's no longer needed in encoded form
    df.drop('amenities', axis=1, inplace=True)
    
    return df

def preprocess_host_verifications(df):
    # Removing special characters and splitting the host_verifications into lists
    df['host_verifications'] = df['host_verifications'].str.replace('[\[\]" ]', '', regex=True)
    
    # Split the host_verifications string into a list of verifications
    df['host_verifications'] = df['host_verifications'].str.split(',')
    
    # Explode the DataFrame such that each verification gets its own row (temporary DataFrame)
    temp_df = df.explode('host_verifications')
    
    # Create binary features for each verification
    for verification in temp_df['host_verifications'].unique():
        df[f'verification_{verification}'] = df['host_verifications'].apply(lambda x: 1 if verification in x else 0)
    
    # Drop the original 'host_verifications' column as it's no longer needed in encoded form
    df.drop('host_verifications', axis=1, inplace=True)
    
    return df

def preprocess_extra_people(df):
    # Removing '$', ',' from the 'extra_people' column and converting it to integer
    df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',', '').astype(float).mul(100).astype(int)
    return df

def preprocess_host_since(df):
    # Convert 'host_since' to datetime and extract the year as a numerical feature
    df['host_since'] = pd.to_datetime(df['host_since']).dt.year
    return df

def convert_categorical_features_to_string(df, categorical_features):
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
    return df

# Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Preprocess 'extra_people' column in both train and test datasets
train_df = preprocess_amenities(train_df)
test_df = preprocess_amenities(test_df)
train_df = preprocess_extra_people(train_df)
test_df = preprocess_extra_people(test_df)
train_df = preprocess_host_since(train_df)
test_df = preprocess_host_since(test_df)
train_df = preprocess_host_verifications(train_df)
test_df = preprocess_host_verifications(test_df)
# Define features list, ensure it matches your dataset's structure
features = [
    'host_since', 'host_location', 'host_is_superhost', 'host_neighbourhood',
    'host_listings_count', 'host_identity_verified', 'neighbourhood_cleansed',
    'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
    'country_code', 'property_type', 'room_type', 'accommodates',
    'bathrooms', 'bedrooms', 'beds', 'bed_type', 'square_feet',
    'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating', 'review_scores_value', 'instant_bookable',
    'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture',
    'require_guest_phone_verification', 'calculated_host_listings_count',
    'reviews_per_month', 'host_response_time', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
]
# Assuming 'features' already contains other columns you wish to keep
amenities_features = [col for col in train_df.columns if col.startswith('amenity_')]
verification_features = [col for col in train_df.columns if col.startswith('host_verifications_')]
features = features + amenities_features + verification_features
features.remove('amenity_Pool with pool hoist')

# Select the features and target variable from the dataset
X = train_df[features]
y = train_df['price']

# Identify and convert categorical features to strings in both train and test datasets
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X = convert_categorical_features_to_string(X, categorical_features)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train the CatBoostRegressor model
model = CatBoostRegressor(cat_features=categorical_features, verbose=0, iterations=5000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Calculate the RMSE for the validation set
rmse = sqrt(mean_squared_error(y_val, val_predictions))
print(f"Validation RMSE: {rmse}")

# Prepare similarly the test dataset and make predictions
test_df = convert_categorical_features_to_string(test_df, categorical_features)
test_predictions = model.predict(test_df[features])

# Prepare the submission
submission_df = pd.DataFrame({'Id': test_df['id'], 'Predicted': test_predictions})
submission_df.to_csv('final_prediction_4.csv', index=False)

print("Prediction file 'final_prediction_4.csv' has been successfully created.")