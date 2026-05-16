import os # OS module for accessing environment variables
from dotenv import load_dotenv # Load variables from .env file
from openai import AzureOpenAI # Azure OpenAI client for LLM interaction

# Initialize environment configuration
load_dotenv()

# Extract Azure OpenAI configuration from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_API_VERSION")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# Initialize the Azure OpenAI client with security credentials
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)


def generate_explanation(customer_row): # Function to generate narrative insights for a specific customer
    """
    Constructs a prompt for the LLM using specific customer data points
    (Churn probability, activity levels, and utilization).
    The model generates a human-readable explanation of why the customer is at risk.
    """

    # Dynamic prompt construction with customer context
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

    # Call the LLM to generate the narrative summary
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    # Return the text-based response from the AI assistant
    return response.choices[0].message.content