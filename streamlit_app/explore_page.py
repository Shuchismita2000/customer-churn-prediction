import streamlit as st
import plotly.express as px
import pandas as pd

# Sample DataFrame (Replace this with your actual DataFrame)
df = pd.read_excel(r'D:\Portfolio Github\customer-churn-prediction\Telco_customer_churn.xlsx')

def show_explore_page():
    st.title("Explore Customer Churn Data")

    # Scenario 1: What is the overall distribution of churn?
    st.header("Scenario 1: What is the overall distribution of churn?")
    st.markdown("""
    **Question:** "Can you provide an overview of the churn distribution in our customer base?"
    
    The plot below shows the count of customers who have churned versus those who have not. 
    This gives us a baseline understanding of the churn rate.
    """)

    # Plot the distribution of churn using Plotly
    fig1 = px.histogram(df, x='Churn Label', color='Churn Label', 
                        title='Churn Distribution', 
                        labels={'Churn Label': 'Churn'}, 
                        color_discrete_sequence=['#636EFA', '#EF553B'])

    fig1.update_layout(xaxis_title='Churn', yaxis_title='Count', 
                       template='plotly_white', 
                       showlegend=False)

    st.plotly_chart(fig1)

    # Scenario 2: What are the demographic characteristics of customers who churn?
    st.header("Scenario 2: What are the demographic characteristics of customers who churn?")
    st.markdown("""
    **Question:** "What are the demographic characteristics (e.g., gender, senior citizen status) of customers who are more likely to churn?"
    
    The plots below will help us understand if gender or senior citizen status has any influence on churn. We can compare the distribution of churn across these demographic groups.
    """)

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

    # Scenario 3: How does tenure affect churn?
    st.header("Scenario 3: How does tenure affect churn?")
    st.markdown("""
    **Question:** "Is there a relationship between the length of tenure with the company and churn?"
    
    The plot below shows how tenure is distributed among churned and non-churned customers. 
    We can observe if longer-tenure customers are less likely to churn.
    """)

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

    # Scenario 4: What is the impact of monthly charges on churn?
    st.header("Scenario 4: What is the impact of monthly charges on churn?")
    st.markdown("""
    **Question:** "Do customers with higher monthly charges tend to churn more?"

    The plot below will help us see if there is a significant difference in monthly charges between customers who churn and those who don't.
    """)

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

    # Scenario 5: How does contract type affect churn?
    st.header("Scenario 5: How does contract type affect churn?")
    st.markdown("""
    **Question:** "Does the type of contract (month-to-month, one year, two years) influence the likelihood of churn?"

    The plot below shows the distribution of churn across different contract types. It helps us understand if longer contracts are associated with lower churn rates.
    """)

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

    # Scenario 6: What is the relationship between different services and churn?
    st.header("Scenario 6: What is the relationship between different services and churn?")
    st.markdown("""
    **Question:** "Are certain services (e.g., internet service, phone service) associated with higher churn rates?"

    The plots below help us see if there are higher churn rates associated with specific services. This insight can guide service improvements or targeted retention efforts.
    """)

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

    # Scenario 7: What is the relationship between payment methods and churn?
    st.header("Scenario 7: What is the relationship between payment methods and churn?")
    st.markdown("""
    **Question:** "Are certain payment methods associated with higher churn rates?"

    The plot below will help us see if customers using certain payment methods are more likely to churn. This insight can help in understanding if payment convenience or security impacts customer retention.
    """)

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

    # Scenario 8: How do additional services affect churn?
    st.header("Scenario 8: How do additional services affect churn?")
    st.markdown("""
    **Question:** "Do additional services like streaming TV, streaming movies, or online security influence customer churn?"

    The plots below will help us determine if customers who subscribe to additional services are more or less likely to churn. This can guide decisions on bundling services or improving specific offerings.
    """)

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