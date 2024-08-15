import streamlit as st
import plotly.express as px
import pandas as pd


# Sample DataFrame (Replace this with your actual DataFrame)
df = pd.read_excel(r'D:\Portfolio Github\customer-churn-prediction\Telco_customer_churn.xlsx')

def explore_page():
    st.title("Explore Customer Churn Past Data")

    # Accordion-style layout for all scenarios
    st.subheader("FAQs")

    scenarios = [
        {"title": "What is the overall distribution of churn?",
         "question": "#1 Can you provide an overview of the churn distribution in our customer base?",
         "answer": "The plot below shows the count of customers who have churned versus those who have not.This gives us a baseline understanding of the churn rate.",
         "insights": """
            **Insights from Churn Distribution:**
            
            - **Churn Rate:** A significant portion of customers, around 26%, have churned (Yes). The remaining 74% have stayed with the company (No).
            - **Imbalance:** There is a clear imbalance in the data, with far fewer customers churning compared to those who remain. This suggests that most customers are generally satisfied, but the churned segment is still sizable and needs attention.
            - **Business Impact:** Understanding the reasons behind the 26% churn can help the business focus on improving customer retention strategies to reduce potential revenue loss.
            """},
        {"title": " What are the demographic characteristics of customers who churn?",
         "question": "#2 What are the demographic characteristics (e.g., gender, senior citizen status) of customers who are more likely to churn?",
          "answer": "The plots below will help us understand if gender or senior citizen status has any influence on churn. We can compare the distribution of churn across these demographic groups.",
          "insights": 
              """
            **Insights from Gender vs Churn:**
            - **Gender Neutrality:** Both males and females exhibit similar churn patterns, indicating that gender does not significantly influence customer churn in this dataset.
            - **Balanced Churn Rates:** The churn rate is almost identical for both genders, suggesting that retention strategies do not need to be gender-specific.
            - **Business Focus:** Since gender does not appear to be a differentiating factor, the company should focus on other variables to better understand and address the reasons behind customer churn.
            
            **Insights from Senior Citizen vs Churn:**

            - **Higher Churn Rate Among Seniors:** Senior citizens show a higher churn rate compared to non-senior citizens. Although fewer seniors are customers, a larger proportion of them tend to leave.
            - **Non-Seniors Retain More:** Non-senior citizens are more likely to stay with the company, with a much lower churn rate relative to their population size.
            - **Targeted Retention:** The company should consider developing tailored retention strategies for senior citizens, who appear more likely to churn, possibly addressing specific needs or concerns unique to this group.

            """
          },
        {"title": "How does tenure affect churn?",
         "question": "#3 Is there a relationship between the length of tenure with the company and churn?",
          "answer":  "The plot below shows how tenure is distributed among churned and non-churned customers. We can observe if longer-tenure customers are less likely to churn.",
          "insights": """
            """},
        {"title": "What is the impact of monthly charges on churn?",
         "question": "#4 Do customers with higher monthly charges tend to churn more?",
         "answer":  " The plot below will help us see if there is a significant difference in monthly charges between customers who churn and those who don't.",
         "insights": """
            **Insights from Monthly Charges Distribution by Churn Status**
            - **Higher Monthly Charges Associated with Churn:** Customers paying higher monthly charges are significantly more likely to churn. The median monthly charge for churned customers is substantially higher than that of retained customers.
            - **Wider Range of Charges Among Churners:** The distribution of monthly charges for churned customers is more spread out, indicating a greater variability in pricing for those who leave compared to those who stay.
            - ** Potential Pricing Sensitivity:** The relationship between higher monthly charges and churn suggests that pricing may be a factor influencing customer satisfaction and retention. It's possible that customers with higher bills feel less value for their money or are more susceptible to competitive offers.
         """},
        {"title": "How does contract type affect churn?",
         "question": "#5 Does the type of contract (month-to-month, one year, two years) influence the likelihood of churn?",
         "answer":  " The plot below shows the distribution of churn across different contract types. It helps us understand if longer contracts are associated with lower churn rates.",
         "insights": """ 
            **Insights from Contract Type Distribution by Churn Status**
            - **Month-to-Month Contracts Have Highest Churn:** Customers with month-to-month contracts have the highest churn rate compared to those on one-year or two-year contracts. This suggests that the lack of a long-term commitment is a significant factor influencing customer churn.
            - **Longer Contracts Reduce Churn:** Both one-year and two-year contracts demonstrate lower churn rates than month-to-month contracts. This indicates that longer-term commitments can help improve customer retention.
            - **Two-Year Contracts Offer Strongest Retention:** Among the contract types, two-year contracts have the lowest churn rate, suggesting that they provide the strongest retention benefits. This could be attributed to factors such as potential discounts or incentives associated with longer-term commitments.
        """},
        {"title": "What is the relationship between different services and churn?",
         "question": "#6 Are certain services (e.g., internet service, phone service) associated with higher churn rates?",
         "answer":  "The plots below help us see if there are higher churn rates associated with specific services. This insight can guide service improvements or targeted retention efforts. ",
         "insights": """
            **Insights from Internet and Phone Service Distribution by Churn Status**
                *Internet Service:*
            - **Fiber Optic Customers Less Likely to Churn:** Customers with fiber optic internet service have a significantly lower churn rate compared to those with DSL or no internet service.
            - **DSL Customers at Higher Risk:** DSL customers have a higher churn rate than both fiber optic customers and those without internet service.
            - **No Internet Service and Churn:** The churn rate for customers without internet service is higher than for DSL customers but lower than for fiber optic customers.
            *Phone Service:*
            - **No Phone Service Linked to Higher Churn:* Customers without phone service have a higher churn rate than those with phone service.
            - **Phone Service Retention:* Having a phone service appears to be a factor in retaining customers, as those with phone service have a lower churn rate overall.
            *Overall:* Fiber optic internet service seems to be a factor in customer retention, while the presence of phone service also contributes to lower churn rates. DSL customers and those without either internet or phone service are at a higher risk of churning.
             """},
        {"title": "What is the relationship between payment methods and churn?",
         "question": "#7 Are certain payment methods associated with higher churn rates?",
         "answer":  "The plot below will help us see if customers using certain payment methods are more likely to churn. This insight can help in understanding if payment convenience or security impacts customer retention.",
         "insights": """ 
            **Insights from Payment Method Distribution by Churn Status**
            - ** Automatic Payments Linked to Lower Churn:** Customers using automatic payment methods (bank transfer or credit card) have significantly lower churn rates compared to those using mailed checks or electronic checks.
            - **Mailed Check Customers at Highest Risk:** Customers who pay by mailed check have the highest churn rate among all payment methods. This suggests that the inconvenience or potential issues associated with mailed checks may contribute to customer dissatisfaction and churn.
            - **Electronic Checks and Churn:** While electronic checks have a lower churn rate than mailed checks, they still have a higher churn rate than automatic payment methods. This indicates that the convenience and reliability of automatic payments may play a role in customer retention.
            - **Potential for Improved Retention:** By encouraging customers to switch to automatic payment methods, the company could potentially improve overall customer retention and reduce churn rates. This could be achieved through incentives, promotions, or simplified enrollment processes.
            """},
        {"title": "How do additional services affect churn?",
         "question": "#8 Do additional services like streaming TV, streaming movies, or online security influence customer churn?",
         "answer":  "The plots below will help us determine if customers who subscribe to additional services are more or less likely to churn. This can guide decisions on bundling services or improving specific offerings.",
         "insights": """
            **Insights from Streaming Services and Online Security by Churn Status**
                *Streaming Services:*
            - ** Streaming TV and Movies: Customers who subscribe to either streaming TV or movies have significantly lower churn rates compared to those without either service. This suggests that these services are effective in retaining customers.
            - ** No Internet Service and Churn: As expected, customers without internet service (and therefore no access to streaming services) have the highest churn rate.
                *Online Security:*
            - ** Online Security and Churn: There is no significant difference in churn rates between customers who subscribe to online security and those who do not. This indicates that online security services do not appear to be a major factor influencing customer retention.
                *Overall:* Streaming TV and movies are strongly associated with lower churn rates, suggesting that these services play a crucial role in customer satisfaction and loyalty. Online security, however, does not seem to have a significant impact on customer retention.
                """},
    ]

    for scenario in scenarios:
        with st.expander(scenario["question"]):
            #st.write(scenario["question"])
            st.markdown(scenario["answer"])
            # Add a 'Get Started' button for each scenario
            if st.button(scenario["title"]):
                # Trigger the corresponding visualization based on the scenario title
                if scenario['question'].startswith("#1"):
                    plot_scenario_1()
                    st.markdown(scenario["insights"])
                elif scenario['question'].startswith("#2"):
                    plot_scenario_2()
                    st.markdown(scenario["insights"])
                #elif scenario['question'].startswith("#3"):
                    #plot_scenario_3(df)
                elif scenario['question'].startswith("#4"):
                    plot_scenario_4()
                    st.markdown(scenario["insights"])
                elif scenario['question'].startswith("#5"):
                    plot_scenario_5()
                    st.markdown(scenario["insights"])
                elif scenario['question'].startswith("#6"):
                    plot_scenario_6()
                    st.markdown(scenario["insights"])
                elif scenario['question'].startswith("#7"):
                    plot_scenario_7()
                    st.markdown(scenario["insights"])
                elif scenario['question'].startswith("#8"):
                    plot_scenario_8()
                    st.markdown(scenario["insights"])
                
def plot_scenario_1():
    # Example for Scenario 1 visualization code
    st.subheader("Churn Distribution")
    # Plot the distribution of churn using Plotly
    fig1 = px.histogram(df, x='Churn Label', color='Churn Label', 
                        title='Churn Distribution', 
                        labels={'Churn Label': 'Churn'}, 
                        color_discrete_sequence=['#636EFA', '#EF553B'])

    fig1.update_layout(xaxis_title='Churn', yaxis_title='Count', 
                       template='plotly_white', 
                       showlegend=False)

    st.plotly_chart(fig1)


def plot_scenario_2():
    # Gender vs Churn using Plotly
    fig2 = px.histogram(df, x='Gender', color='Churn Label', 
                        title='Gender vs Churn', 
                        labels={'Churn Label': 'Churn'}, 
                        color_discrete_sequence=['#636EFA', '#EF553B'])

    fig2.update_layout(xaxis_title='Gender', yaxis_title='Count', 
                       template='plotly_white')

    st.plotly_chart(fig2)

    # Senior Citizen vs Churn using Plotly
    fig3 = px.histogram(df, x='Senior Citizen', color='Churn Label', 
                        title='Senior Citizen vs Churn', 
                        labels={'Churn Label': 'Churn', 'Senior Citizen': 'Senior Citizen Status'}, 
                        color_discrete_sequence=['#636EFA', '#EF553B'])

    fig3.update_layout(xaxis_title='Senior Citizen Status', yaxis_title='Count', 
                       template='plotly_white')

    st.plotly_chart(fig3)

#def plot_scenario_3():
    # Check if the required columns are present and if there are any null values
    #if 'Tenure Months' in df.columns and 'Churn Label' in df.columns:
    #    df_filtered = df.dropna(subset=['Tenure Months', 'Churn Label'])

    #    # Tenure vs Churn using Plotly
    #    fig4 = px.histogram(df_filtered, x='Tenure Months', color='Churn Label', 
    #                        title='Tenure Distribution by Churn Status', 
    #                        labels={'Tenure Months': 'Tenure (Months)', 'Churn Label': 'Churn'}, 
    #                        marginal='kde', 
    #                        color_discrete_sequence=['#636EFA', '#EF553B'], 
    #                        histnorm='density')
#
    #    fig4.update_layout(xaxis_title='Tenure (Months)', yaxis_title='Density', 
    #                       template='plotly_white')

    #    st.plotly_chart(fig4)
    #else:
    #    st.error("The dataset does not contain the required columns: 'Tenure Months' and 'Churn Label'.")

def plot_scenario_4():
    # Check if the required columns are present and if there are any null values
    if 'Monthly Charges' in df.columns and 'Churn Label' in df.columns:
        df_filtered = df.dropna(subset=['Monthly Charges', 'Churn Label'])

        # Monthly Charges vs Churn using Plotly
        fig5 = px.box(df_filtered, x='Churn Label', y='Monthly Charges',
                      title='Monthly Charges Distribution by Churn Status',
                      labels={'Churn Label': 'Churn', 'Monthly Charges': 'Monthly Charges'},
                      color='Churn Label', 
                      color_discrete_sequence=['#636EFA', '#EF553B'])

        fig5.update_layout(xaxis_title='Churn', yaxis_title='Monthly Charges',
                           template='plotly_white')

        st.plotly_chart(fig5)

def plot_scenario_5():
    # Check if the required columns are present and if there are any null values
    if 'Contract' in df.columns and 'Churn Label' in df.columns:
        df_filtered = df.dropna(subset=['Contract', 'Churn Label'])

        # Contract vs Churn using Plotly
        fig6 = px.histogram(df_filtered, x='Contract', color='Churn Label',
                            title='Contract Type Distribution by Churn Status',
                            labels={'Contract': 'Contract Type', 'Churn Label': 'Churn'},
                            barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])

        fig6.update_layout(xaxis_title='Contract Type', yaxis_title='Count',
                           template='plotly_white')

        st.plotly_chart(fig6)
    else:
        st.error("The dataset does not contain the required columns: 'Contract' and 'Churn Label'.")


def plot_scenario_6():
    # Check if the required columns are present and if there are any null values
    if 'Internet Service' in df.columns and 'Phone Service' in df.columns and 'Churn Label' in df.columns:
        df_filtered = df.dropna(subset=['Internet Service', 'Phone Service', 'Churn Label'])

        # Plot Internet Service vs Churn using Plotly
        fig7 = px.histogram(df_filtered, x='Internet Service', color='Churn Label',
                            title='Internet Service Distribution by Churn Status',
                            labels={'Internet Service': 'Internet Service', 'Churn Label': 'Churn'},
                            barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])

        fig7.update_layout(xaxis_title='Internet Service', yaxis_title='Count',
                           template='plotly_white')

        # Plot Phone Service vs Churn using Plotly
        fig8 = px.histogram(df_filtered, x='Phone Service', color='Churn Label',
                            title='Phone Service Distribution by Churn Status',
                            labels={'Phone Service': 'Phone Service', 'Churn Label': 'Churn'},
                            barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])

        fig8.update_layout(xaxis_title='Phone Service', yaxis_title='Count',
                           template='plotly_white')

        # Display the plots side by side in Streamlit
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig7)
        with col2:
            st.plotly_chart(fig8)
    else:
        st.error("The dataset does not contain the required columns: 'Internet Service', 'Phone Service', and 'Churn Label'.")

def plot_scenario_7():
    # Check if the required columns are present and if there are any null values
    if 'Payment Method' in df.columns and 'Churn Label' in df.columns:
        df_filtered = df.dropna(subset=['Payment Method', 'Churn Label'])

        # Plot Payment Method vs Churn using Plotly
        fig9 = px.histogram(df_filtered, x='Payment Method', color='Churn Label',
                            title='Payment Method Distribution by Churn Status',
                            labels={'Payment Method': 'Payment Method', 'Churn Label': 'Churn'},
                            barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])

        fig9.update_layout(xaxis_title='Payment Method', yaxis_title='Count',
                           template='plotly_white', xaxis_tickangle=-45)

        # Display the plot in Streamlit
        st.plotly_chart(fig9)
    else:
        st.error("The dataset does not contain the required columns: 'Payment Method' and 'Churn Label'.")

def plot_scenario_8():
    # Plot Streaming TV vs Churn using Plotly
    if 'Streaming TV' in df.columns and 'Churn Label' in df.columns:
        fig10 = px.histogram(df, x='Streaming TV', color='Churn Label',
                             title='Streaming TV Distribution by Churn Status',
                             labels={'Streaming TV': 'Streaming TV', 'Churn Label': 'Churn'},
                             barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])
        fig10.update_layout(xaxis_title='Streaming TV', yaxis_title='Count', template='plotly_white')
        st.plotly_chart(fig10)
    else:
        st.error("The dataset does not contain the required columns: 'Streaming TV' and 'Churn Label'.")

    # Plot Streaming Movies vs Churn using Plotly
    if 'Streaming Movies' in df.columns and 'Churn Label' in df.columns:
        fig11 = px.histogram(df, x='Streaming Movies', color='Churn Label',
                             title='Streaming Movies Distribution by Churn Status',
                             labels={'Streaming Movies': 'Streaming Movies', 'Churn Label': 'Churn'},
                             barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])
        fig11.update_layout(xaxis_title='Streaming Movies', yaxis_title='Count', template='plotly_white')
        st.plotly_chart(fig11)
    else:
        st.error("The dataset does not contain the required columns: 'Streaming Movies' and 'Churn Label'.")

    # Plot Online Security vs Churn using Plotly
    if 'Online Security' in df.columns and 'Churn Label' in df.columns:
        fig12 = px.histogram(df, x='Online Security', color='Churn Label',
                             title='Online Security Distribution by Churn Status',
                             labels={'Online Security': 'Online Security', 'Churn Label': 'Churn'},
                             barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'])
        fig12.update_layout(xaxis_title='Online Security', yaxis_title='Count', template='plotly_white')
        st.plotly_chart(fig12)
    else:
        st.error("The dataset does not contain the required columns: 'Online Security' and 'Churn Label'.")