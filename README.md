# Congometrix FastAPI Application

This is a FastAPI application designed to provide AI services for a data accessibility platform. The application integrates with modern technologies and machine learning to deliver robust data processing capabilities.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install and set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/pacifiquerubasha/municonnect-ai
    cd municonnect-ai
    ```

2. Create a virtual environment:
    ```bash
    python -m venv env
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source env/bin/activate
        ```

4. Install the dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5. Create a `.env` file in the root of your project and add any required environment variables.

6. Run the application:
    ```bash
    python main.py
    ```

The application will be available at [http://localhost:8000](http://localhost:8000).

## Usage

To use the application, follow these steps:

1. Ensure the application is running by navigating to [http://localhost:8080](http://localhost:8080).
2. Explore the automatically generated API documentation at [http://localhost:8080/docs](http://localhost:8080/docs) or [http://localhost:8000/redoc](http://localhost:8080/redoc).
3. Utilize the provided endpoints for various data processing and interaction tasks.

## Environment Variables

Specify your environment variables in a `.env` file in the root of your project. These include:

```env
# Example environment variables
S3_ACCESS_KEY=
S3_SECRET_KEY=
OPENAI_API_KEY=

