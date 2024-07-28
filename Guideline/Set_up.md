In this section, we will start by setting up our project environment. We'll create a folder for our project, set up a virtual environment, and connect our project to GitHub. This setup will ensure our code is organized, dependencies are managed, and version control is in place.

#### **Step 1: Create a Project Folder**

* **Create a Project Folder**: First, create a folder for your project. Open your terminal and run the following command:

		```
        bash
		mkdir customer-churn-prediction
        cd customer-churn-prediction
        ```
![][image1]

* **Initialize a Git Repository**: Initialize a Git repository in your project folder:
    *`bash`*  
    `git init`

![][image2]

**Step 2: Set Up a Virtual Environment**

* **`Create a Virtual Environment`**`: Create a virtual environment using venv. This ensures that all dependencies are isolated from your global Python environment.`  
  *`bash`*  
  `python -m venv venv`


`![][image3]`

* **`Activate the Virtual Environment`**`: Activate the virtual environment:`  
  1. `On Windows:`  
     *`bash`*  
     `.\venv\Scripts\activate`

  `![][image4]`

     1. `On macOS and Linux:`  
        *`bash`*  
        `source venv/bin/activate`  
          
  * **`Install Required Packages`**`: Install the necessary packages using pip. Start with pandas, numpy, matplotlib, seaborn, and scikit-learn. You can add more packages as needed later.`  
    *`bash`*  
    `pip install pandas numpy matplotlib seaborn scikit-learn`

`![][image5]`

`![][image6]`

* **`Freeze Dependencies`**`: Create a requirements.txt file to track your project's dependencies. This file will help others recreate the environment.`  
  *`bash`*  
  `pip freeze > requirements.txt`

`![][image7]`

#### **`Step 3: Set Up Git Ignore`**

* **`Create a .gitignore File`**`: Create a .gitignore file to exclude certain files and directories from being tracked by Git. This is important to avoid committing unnecessary files, such as your virtual environment.`  
  *`bash`*  
  `New-Item .gitignore -ItemType File`

`![][image8]`

* **`Add Entries to .gitignore`**`: Open the .gitignore file in a text editor and add the following entries to exclude the virtual environment and other unnecessary files:`  
  `# Virtual environment`  
  `venv/`

  `# Python cache`

  `__pycache__/`

  `# Jupyter Notebook checkpoints`

  `.ipynb_checkpoints/`

  `# Other`

  `.DS_Store`

  `*.pyc`

  `.env`

  `![][image9]`

#### **`Step 4: Connect to GitHub`**

* **`Create a New Repository on GitHub`**`: Go to GitHub and create a new repository. Do not initialize it with a README, .gitignore, or license as we have already set up our project locally.`

	`	![][image10]`

* **`Link Local Repository to GitHub`**`: Link your local repository to the remote GitHub repository. Replace YOUR_GITHUB_USERNAME and REPOSITORY_NAME with your actual GitHub username and repository name.`  
  *`bash`*  
  `git remote add origin https://github.com/YOUR_GITHUB_USERNAME/REPOSITORY_NAME.git`

  `![][image11]`


  `![][image12]`

#### **`![][image13]`**

`Complete the authentication to connect VS code with your github account.` 

* **`Push Initial Commit`**`: Add all files to the staging area, commit the changes, and push them to GitHub.`  
  *`bash`*  
  `git add .`

  `git commit -m "Initial commit"`

  `git push -u origin master`


`![][image14]`

#### **`Step 5: Open the Project in Visual Studio Code (VS Code)`**

* **`Open VS Code`**`: Open Visual Studio Code and navigate to your project folder:`  
  `bash`  
  `code .`  
  * **`Set Up Python Interpreter`**`: In VS Code, select the Python interpreter associated with your virtual environment:`  
    1. `Open the Command Palette (Ctrl+Shift+P on Windows/Linux or Cmd+Shift+P on macOS).`  
    1. `Type Python: Select Interpreter and choose the interpreter located in your virtual environment folder (e.g., venv/bin/python).`  
  * **`Install Python Extensions`**`: Ensure you have the Python extension installed in VS Code for better support for Python development.`

`With the environment set up and connected to GitHub, you are ready to start working on the project. In the next section, we will load the Telco Customer Churn dataset into our environment and begin our exploratory data analysis (EDA).`

### 
