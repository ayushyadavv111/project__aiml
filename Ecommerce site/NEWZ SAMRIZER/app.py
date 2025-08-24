import os
import re
import json
import dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from groq import Groq

# Load environment variables
dotenv.load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")  # Match the .env variable name
client = Groq(api_key=groq_api)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with something secure

def analyze_product(product_link, product_description):
    PROMPT_TEMPLATE = f"""
    Product Link: {product_link if product_link else '[User did not provide a link. Suggest an official or relevant product link.]'}
    Product Description: {product_description}

    You are an expert assistant helping users understand products. Explain the product in simple language, including its name, model, price, features, and any other important details. 
    Format your response like this:

    Product Name:
    ...

    Model:
    ...

    Price:
    ...

    Features:
    - ...
    - ...

    Details:
    ...

    Product Link:
    ...
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": PROMPT_TEMPLATE}],
        temperature=0.3,
        max_tokens=800
    )

    expected_response = response.choices[0].message.content
    if not expected_response or expected_response.strip() == "":
        return None, "No response from Groq API. Please check your API key, prompt, or try again."

    # Parse response
    try:
        product_name = re.search(r"Product Name:\s*(.*?)(?:Model:|$)", expected_response, re.DOTALL)
        model_match = re.search(r"Model:\s*(.*?)(?:Price:|$)", expected_response, re.DOTALL)
        price_match = re.search(r"Price:\s*(.*?)(?:Features:|$)", expected_response, re.DOTALL)
        features_match = re.search(r"Features:\s*(.*?)(?:Details:|$)", expected_response, re.DOTALL)
        details_match = re.search(r"Details:\s*(.*?)(?:Product Link:|$)", expected_response, re.DOTALL)
        product_link_match = re.search(r"Product Link:\s*(.*)", expected_response, re.DOTALL)

        result = {
            "product_name": product_name.group(1).strip() if product_name else "",
            "model": model_match.group(1).strip() if model_match else "",
            "price": price_match.group(1).strip() if price_match else "",
            "features": [line.strip('- ').strip() for line in features_match.group(1).split('\n') if line.strip()] if features_match else [],
            "details": details_match.group(1).strip() if details_match else "",
            "product_link": product_link_match.group(1).strip() if product_link_match else ""
        }
        return result, None
    except Exception as e:
        return None, f"Error parsing response: {e}. Raw response: {expected_response}"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        product_link = request.form.get('product_link', '').strip()
        product_description = request.form.get('product_description', '').strip()
        if product_link or product_description:
            result, error = analyze_product(product_link, product_description)
            session['result'] = result
            session['error'] = None if result else error
            return redirect(url_for('index'))
        else:
            session['result'] = None
            session['error'] = "Please provide at least a product link or a product description."
            return redirect(url_for('index'))
    else:
        result = session.pop('result', None)
        error = session.pop('error', None)
    return render_template('index.html', result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
