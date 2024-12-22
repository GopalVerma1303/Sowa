import os
import subprocess


def setup_project():
    # Create necessary directories
    os.makedirs("textbooks", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Install requirements
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

    # Set up environment variables
    os.environ["GROQ_API_KEY"] = input("Enter your Groq API key: ")

    print("Project setup complete!")


if __name__ == "__main__":
    setup_project()
