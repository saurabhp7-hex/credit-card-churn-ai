import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Read from .env
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_API_VERSION")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# Create client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)


def generate_explanation(customer_row):

    prompt = f"""
    You are a banking AI assistant.

    Analyze the following credit card customer:

    - Churn Probability: {round(customer_row['churn_probability'], 2)}
    - Transactions Count: {customer_row['Total_Trans_Ct']}
    - Inactive Months: {customer_row['Months_Inactive_12_mon']}
    - Credit Utilization: {round(customer_row['Avg_Utilization_Ratio'], 2)}

    Your task:

    1. Classify the customer as:
    - High Risk / Medium Risk / Low Risk

    2. Provide a SHORT summary (1–2 lines)

    3. Provide 3–4 bullet points explaining key behavior

    4. Suggest why the retention strategy makes sense

    Format strictly like:

    Risk Level: <value>

    Summary:
    <short explanation>

    Key Drivers:
    - ...
    - ...
    - ...

    Business Interpretation:
    <one short line>
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content