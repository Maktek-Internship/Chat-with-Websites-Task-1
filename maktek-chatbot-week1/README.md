# Website Interaction Chatbot

## Overview

This project is a Streamlit-based chatbot application that interacts with websites. It allows users to input a URL and ask questions about the website's content. The application leverages the Google Gemini API for natural language processing and advanced web scraping techniques to handle CAPTCHA challenges and retrieve text from websites. The code has been modularized for improved maintainability and scalability.

## Features

- **Interactive Chat Interface:** Users can input a website URL and ask questions via a Streamlit interface.
- **Google Gemini Integration:** Utilizes Google Gemini API for natural language understanding and response generation.
- **CAPTCHA Handling:** Includes functionality to solve CAPTCHAs using OCR.
- **Modular Code:** Refactored into distinct functions for better readability and maintainability.
- **Logging:** Integrated logging for tracking errors and system behavior.

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed. You will also need an API key for Google Gemini.

### Clone the Repository

```bash
git clone https://github.com/yourusername/website-interaction-chatbot.git
cd website-interaction-chatbot
```

## Install Dependencies
### Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Dependencies
```bash
streamlit
langchain_google_genai
langchain_core
requests
beautifulsoup4
pytesseract
Pillow
chromadb
```
## Configuration
Google Gemini API Key: Set up your Google Gemini API key in the environment. You can do this by adding GOOGLE_API_KEY to your environment variables or entering it in the Streamlit interface.

Tesseract Configuration: Ensure Tesseract OCR is installed and properly configured. Update the path in the code if necessary:
```bash
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

## Usage

### Run the Application
Before running the application, delete the db file if it exists.
```bash
streamlit run app.py
```

## Interact with the Application:

* Open the application in your browser (usually at http://localhost:8501).
* Enter your Google Gemini API key in the sidebar.
* Provide the URL of the website you want to interact with.
* Ask questions about the website's content.
* Code Structure
* app.py: Main application script with Streamlit UI and functionality.
* modules/: Contains modularized functions for CAPTCHA solving, web scraping, and text processing.
* requirements.txt: Lists all the dependencies required for the project.
