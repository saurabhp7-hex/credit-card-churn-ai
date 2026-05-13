from preprocessing import load_data, clean_data, encode_features
from model import train_model, evaluate_model, add_predictions
from business import calculate_clv, revenue_at_risk
from retention import assign_strategy, simulate_campaign
from model import save_model

def main():
    # Step 1: Load data
    df = load_data("../data/bank_churners.csv")

    # Step 2: Clean + preprocess
    df = clean_data(df)
    df = encode_features(df)

    # Step 3: Train model
    model, X_test, y_test, X = train_model(df)
    evaluate_model(model, X_test, y_test)

    save_model(model)

    # Step 4: Predictions
    df = add_predictions(df, model, X)

    # Step 5: Business layer
    df = calculate_clv(df)
    df = revenue_at_risk(df)

    # Step 6: Retention engine
    df = assign_strategy(df)
    df = simulate_campaign(df)

    print(df[['churn_probability', 'CLV', 'revenue_at_risk', 'retention_strategy', 'roi']].head())
    df.to_csv("../output/results.csv", index=False)
    print("✅ Results saved to output/results.csv")


if __name__ == "__main__":
    main()