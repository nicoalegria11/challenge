# Software Engineer (ML & LLMs) Challenge

## Overview

Welcome to the **Presentation**  of explanations or assumptions regarding the work carried out. In this presentation, we will provide the rationale for the choice of the model, assumptions, and considerations.

## Part I

**Model Selection**: In selecting the model between Logistic Regression with Feature Importance and Class Balancing and XGBoost with Feature Importance and Class Balancing, we need to consider several factors to make an informed decision. These factors are Performance Similarity, Simplicity and Interpretability, Training and Inference Speed, Deployment and Maintenance, among others.

I choose **Logistic Regression with Feature Importance and Class Balancing** for the following reasons:

- Both Logistic Regression and XGBoost provide similar predictive performance, so opting for the simpler model makes sense to keep the solution straightforward

- Logistic Regression is more interpretable, which is crucial when explaining model predictions to non-technical stakeholders or investigating the reasons behind specific predictions

- Logistic Regression is faster to train, which is advantageous for real-time or near-real-time prediction scenarios. This is important because the airline industry have to manage a big amount of flights everyday.

Ultimately, the choice of model depends on the specific requirements and constraints of the each scenario.

**Assumptions**: An attempt was made to create a scalable model with a set of adjustable parameters for future modifications, allowing us to fine-tune the model depending on the scenarios encountered. For instance, I found that obtaining the top 10 features through data analysis is a good practice compared to manual selection. This approach enables our model to adapt to changes in the industry.

### Part II

It was executed smoothly, achieving accurate predictions at least for the data presented in the tests and some test data created by the author of this code

### Part III

**URL**: https://challengebynar.rj.r.appspot.com/

### Part IV

I was unable to complete this part because, even though we had 5 days to complete the challenge, I had a weekend filled with responsibilities and couldn't allocate enough time to the task. I managed to work on the branch, but I haven't merged it into dev because I didn't have a chance to test it (5-partIV-CI/CD-integration)