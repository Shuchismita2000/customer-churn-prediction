import streamlit as st
import os

def data_collection_page():
    st.title("Git & VS Code Setup")

    st.write("""
    In this section, we will start by setting up our project environment. We'll create a folder for our project, set up a virtual environment, and connect our project to GitHub. This setup will ensure our code is organized, dependencies are managed, and version control is in place.
    """)

    st.header("Step 1: Create a Project Folder")
    st.write("""
    **Create a Project Folder:** First, create a folder for your project. Open your terminal and run the following command:
    ```bash
    mkdir customer-churn-prediction
    cd customer-churn-prediction
    ```
     """)
    
    image_path = os.path.join("Guideline", "images", "img1.png")
    st.image(image_path, 
             caption="Creating a Project Folder in Terminal", use_column_width=True)
    
    st.write("""
    **Initialize a Git Repository:** Initialize a Git repository in your project folder:
    ```bash
    git init
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img2.png")
    st.image(image_path, 
             caption="Initialize a Git Repository", use_column_width=True)

    st.header("Step 2: Set Up a Virtual Environment")
    st.write("""
    **Create a Virtual Environment:** Create a virtual environment using `venv`. This ensures that all dependencies are isolated from your global Python environment.
    ```bash
    python -m venv venv
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img3.png")
    st.image(image_path, 
             caption="Create a Virtual Environment", use_column_width=True)
             
    st.write("""         
    **Activate the Virtual Environment:** Activate the virtual environment:

    On Windows:
    ```bash
    .\\venv\\Scripts\\activate
    ```

    On macOS and Linux:
    ```bash
    source venv/bin/activate
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img4.png")
    st.image(image_path, 
             caption="Activate the Virtual Environment", use_column_width=True)

    st.write(""" 
    **Install Required Packages:** Install the necessary packages using pip. Start with pandas, numpy, matplotlib, seaborn, and scikit-learn. You can add more packages as needed later.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img5.png")
    st.image(image_path, 
             caption="Install Required Packages", use_column_width=True)
    image_path = os.path.join("Guideline", "images", "img6.png")
    st.image(image_path, 
             caption="Install Required Packages", use_column_width=True)
    
    st.write(""" 
    **Freeze Dependencies:** Create a `requirements.txt` file to track your project's dependencies. This file will help others recreate the environment.
    ```bash
    pip freeze > requirements.txt
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img7.png")
    st.image(image_path, 
             caption="Freeze Dependencies", use_column_width=True)
    
    st.header("Step 3: Set Up Git Ignore")
    st.write("""
    **Create a `.gitignore` File:** Create a `.gitignore` file to exclude certain files and directories from being tracked by Git. This is important to avoid committing unnecessary files, such as your virtual environment.
    ```bash
    touch .gitignore
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img8.png")
    st.image(image_path, 
             caption="Create a `.gitignore` File", use_column_width=True)

    st.write("""
    **Add Entries to `.gitignore`:** Open the `.gitignore` file in a text editor and add the following entries to exclude the virtual environment and other unnecessary files:
    ```bash
    # Virtual environment
    venv/

    # Python cache
    __pycache__/

    # Jupyter Notebook checkpoints
    .ipynb_checkpoints/

    # Other
    .DS_Store
    *.pyc
    .env
    ```
    """)

    image_path = os.path.join("Guideline", "images", "img9.png")
    st.image(image_path, 
             caption="Add Entries to `.gitignore`", use_column_width=True)


    st.header("Step 4: Connect to GitHub")
    st.write("""
    **Create a New Repository on GitHub:** Go to GitHub and create a new repository. Do not initialize it with a README, .gitignore, or license as we have already set up our project locally.
    """)
    image_path = os.path.join("Guideline", "images", "img10.png")
    st.image(image_path, 
             caption="Create a New Repository on GitHub", use_column_width=True)

    st.write("""
    **Link Local Repository to GitHub:** Link your local repository to the remote GitHub repository. Replace `YOUR_GITHUB_USERNAME` and `REPOSITORY_NAME` with your actual GitHub username and repository name.
    ```bash
    git remote add origin https://github.com/YOUR_GITHUB_USERNAME/REPOSITORY_NAME.git
    ```
    """)
    image_path = os.path.join("Guideline", "images", "img11.png")
    st.image(image_path, 
             caption="Create a New Repository on GitHub", use_column_width=True)
    image_path = os.path.join("Guideline", "images", "img12.png")
    st.image(image_path, 
             caption="Create a New Repository on GitHub", use_column_width=True)
    image_path = os.path.join("Guideline", "images", "img13.png")
    st.image(image_path, 
             caption="Create a New Repository on GitHub", use_column_width=True)
    

    st.write("""
    Complete the authentication to connect VS Code with your GitHub account.

    **Push Initial Commit:** Add all files to the staging area, commit the changes, and push them to GitHub.
    ```bash
    git add .
    git commit -m "Initial commit"
    git push -u origin master
    ```
    """)

    st.image("../Guideline/images/image14.png", 
             caption="Push Initial Commit", use_column_width=True)
    
    
    st.header("Step 5: Open the Project in Visual Studio Code (VS Code)")
    st.write("""
    **Open VS Code:** Open Visual Studio Code and navigate to your project folder:
    ```bash
    code .
    ```

    **Set Up Python Interpreter:** In VS Code, select the Python interpreter associated with your virtual environment:
    1. Open the Command Palette (Ctrl+Shift+P on Windows/Linux or Cmd+Shift+P on macOS).
    2. Type `Python: Select Interpreter` and choose the interpreter located in your virtual environment folder (e.g., `venv/bin/python`).

    **Install Python Extensions:** Ensure you have the Python extension installed in VS Code for better support for Python development.

    With the environment set up and connected to GitHub, you are ready to start working on the project. In the next section, we will load the Telco Customer Churn dataset into our environment and begin our exploratory data analysis (EDA).
    """)


