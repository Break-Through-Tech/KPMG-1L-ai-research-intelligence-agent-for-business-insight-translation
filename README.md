# Krystal - your 

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Brando Budhan    | @Bran7astic  | Ingestion, Frontend             |
| Arushi Agarwal   | @arushi-a1    | Validation  |
| Michelle Hong     | @seashello  | Retrieval                |

---

## 🎯 **Project Highlights**

- Developed a Retrieval Augmented Generation (RAG) chatbot, turning `advanced AI Research into relevant business insights` for `KPMG`.
- Locally-ran application and easy-to-use interface for any user.

---

## 👩🏽‍💻 **Setup and Installation**

<!-- **Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts -->

Before installing, ensure you have the following tools installed: 
* **Conda:** https://www.anaconda.com/download/success?reg=auth

---
1. Clone the repository to your machine by running

    ```
    git clone https://github.com/Break-Through-Tech/KPMG-1L-ai-research-intelligence-agent-for-business-insight-translation.git
    ```
    ```
    cd KPMG-1L-ai-research-intelligence-agent-for-business-insight-translation
    ```

2. Create and activate Conda environment from environment.yml by running 
    ```
    conda env create -f environment.yml
    ```
    ```
    conda activate KPMG_1L
    ```

3. To install new packages:

    ```
    conda install <package name>
    conda env export --from-history > environment.yml
    ```
4. Everytime you pull new code, update your conda environment by running
    ```
    conda activate KPMG_1L

    conda env update --file environment.yml --prune
    ```

5. To use modules from the conda environment in a `.ipynb` jupyter notebook file:
    1. Open your `.ipynb` file in VS Code.
    2. Look at the top-right corner of the notebook window and click on Select Kernel (or click the existing kernel name if one is already selected).
    3. In the dropdown palette that opens, choose Python Environments...
    4. Select your specific Conda environment from the list. It should look something like: 
    `KPMG_1L (Python 3.14.7)`

---

## 🏗️ **Project Overview**

- KPMG is a global leader in audit, tax, and advisory services, providing insights and expertise to a diverse clientele across various industries and countries
- With new research published every day, there is new information that can be useful and applied to any setting. However, with so much information, it is difficult to ingest everything at scale.
- This project enables efficiently monitoring and operationalizing the rapidly growing volume of AI research within a business setting. Turning technical knowledge into tangible solutions for non-technical leaders is extremely valuable.

---

## 📊 **Data Exploration**

* ArXiv is an open-source archive for millions of research papers. Hundreds of new papers are added every day in the growing field of Artificial Intelligence, which contains a
* include a visual (either data visualization about how much AI research there is / why this is important) 

---

## 🧠 **Rag Process**

* Include a chart about what RAG is and how our app works
* Query --> search for relevant papers directly from ArXiv --> store paper locally --> choose 1 paper if it reaches threshold --> summarize paper

---

## 📈 **Results & Key Findings**

* include runtime (how long it takes for a new + existing query)
* how many calls we can make with our key
* etc.

---

## 🚀 **Next Steps**

* What are some of the limitations of your model?
* deploying
* new ideas

---

## 📝 **License** (ask Deanna!)

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank you to Deanna DiMonte and Lauren Fang for their generous help and support!
