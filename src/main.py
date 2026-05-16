from preprocessing import load_data, clean_data, encode_features # Import data loading and cleaning functions
from model import train_model, evaluate_model, add_predictions # Import model training and evaluation functions
from business import calculate_clv, revenue_at_risk # Import business logic for CLV and risk calculation
from retention import assign_strategy, simulate_campaign # Import retention strategy and simulation logic
from model import save_model # Import function to save the trained model
import joblib # Import joblib for model serialization

def main(): # Define the main orchestration function
    # Step 1: Load data
    df = load_data("../data/bank_churners.csv") # Load raw bank churn data from CSV

    # Step 2: Clean + preprocess
    df = clean_data(df) # Clean data by removing unnecessary columns and handling targets
    df = encode_features(df) # Perform one-hot encoding on categorical variables

    # Step 3: Train model
    model, X_test, y_test, X, explainer = train_model(df) # Train Random Forest model and initialize SHAP explainer
    evaluate_model(model, X_test, y_test) # Output model performance metrics (ROC-AUC, F1)

    save_model(model) # Save the trained model to disk
    joblib.dump(explainer, "../models/shap_explainer.pkl") # Persist the SHAP explainer for the dashboard

    # Step 4: Predictions
    df = add_predictions(df, model, X) # Generate churn probabilities and assign risk segments

    # Step 5: Business layer
    df = calculate_clv(df) # Estimate Customer Lifetime Value (CLV)
    df = revenue_at_risk(df) # Calculate potential revenue loss based on churn risk

    # Step 6: Retention engine
    df = assign_strategy(df) # Determine optimal retention strategy for each segment
    df = simulate_campaign(df) # Simulate the financial impact of retention incentives

    print(df[['churn_probability', 'risk_segment', 'CLV', 'revenue_at_risk', 'retention_strategy', 'roi']].head()) # Preview top results
    df.to_csv("../output/results.csv", index=False) # Save final augmented dataset to results folder
    
    # Save a reference copy for drift detection
    df.to_csv("../data/reference_data.csv", index=False) # Store reference data for future monitoring/drift checks
    
    print("✅ Results saved to output/results.csv") # Success confirmation message


if __name__ == "__main__": # Entry point check
    main() # Execute the main pipeline