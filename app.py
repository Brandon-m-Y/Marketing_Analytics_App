import streamlit as st

st.set_page_config(
    page_title="BMY Analytics - Privacy-First Marketing Analytics for Small Businesses",
    page_icon="🏠", 
    layout="wide"
)

# Hero Section
st.title("🎯 BMY Analytics")
st.subheader("Privacy-First Marketing Analytics for Small Businesses")

# Value proposition with emphasis on privacy
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### Understand Your Customers Without Compromising Their Privacy
        
        Built for **small businesses** who want to segment customers, forecast sales, 
        and optimize marketing—**without paying enterprise prices or sending sensitive 
        customer data to big tech companies**.
        
        **Perfect for:**
        - 🎯 **Customer Segmentation** - Find your best customers and tailor your marketing
        - 📈 **Sales Forecasting** - Predict trends and plan inventory
        - 💡 **Marketing Optimization** - Understand what drives purchases
        - 📊 **General Analytics** - Explore any business data you have
        
        **Why small businesses choose us:**
        - ✅ **100% Local Processing** - Customer data never leaves your computer
        - ✅ **No Subscriptions** - No monthly fees like enterprise platforms
        - ✅ **Open Source** - [Verify our privacy promise yourself](https://github.com/Brandon-m-Y/Marketing_Analytics_App)
        - ✅ **No Data Mining** - We don't collect, sell, or train on your customer data
        """
    )

with col2:
    # Privacy badge/emphasis
    st.markdown(
        """
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;'>
            <h3 style='color: #2e7d32; margin-top: 0;'>🔐 Privacy Promise</h3>
            <p style='color: #1b5e20; margin-bottom: 0;'>
                <strong>Your customer data is YOUR data.</strong><br><br>
                ❌ No cloud uploads<br>
                ❌ No tracking pixels<br>
                ❌ No data harvesting<br>
                ❌ No AI training on your data<br><br>
                ✅ Everything runs locally<br>
                ✅ You control what gets shared<br><br>
                <em>Don't take our word for it—<br><a href='https://github.com/Brandon-m-Y/Marketing_Analytics_App' target='_blank'>review our open source code</a>.</em>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.divider()

# Marketing-focused use cases
st.subheader("🎯 Marketing Segmentation Made Simple")

st.markdown(
    """
    The #1 use case: **Understand who your customers are and how to reach them.**
    
    Stop treating all customers the same. Use data to find:
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        #### 👑 Your VIPs
        **High-value customers**
        - Who spends the most?
        - Who buys most frequently?
        - Who's most loyal?
        
        *Target with premium offers*
        """
    )

with col2:
    st.markdown(
        """
        #### 💤 At-Risk Customers
        **Churning or dormant**
        - Who used to buy but stopped?
        - Who's reducing spend?
        - Who hasn't returned?
        
        *Win them back with incentives*
        """
    )

with col3:
    st.markdown(
        """
        #### 🌱 Growth Opportunities
        **High-potential segments**
        - Who's trending up?
        - Which products sell together?
        - What drives repeat purchases?
        
        *Optimize your marketing spend*
        """
    )

st.divider()

# What you can do section
st.subheader("📊 Beyond Marketing: General Analytics Too")

st.markdown(
    """
    While **marketing segmentation** is our sweet spot, BMY Analytics works for any business data analysis:
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        #### 🧹 Clean Your Data
        - Remove duplicates
        - Handle missing values
        - Fix formatting issues
        - Create custom metrics
        
        *Turn messy spreadsheets into clean datasets*
        """
    )

with col2:
    st.markdown(
        """
        #### 🔍 Explore & Visualize
        - Charts and graphs
        - Find correlations
        - Spot trends
        - Identify outliers
        
        *See patterns you'd miss in Excel*
        """
    )

with col3:
    st.markdown(
        """
        #### 🤖 AI-Powered Insights*
        - Explain segments in plain English
        - Get marketing recommendations
        - Understand "why" behind the numbers
        
        *Let AI translate data into action*
        """
    )

st.caption("*Optional feature: Only anonymized charts are sent to AI (using your own API key), never your customer data")

st.divider()

# How it works section
st.subheader("🚀 How It Works (4 Simple Steps)")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown(
        """
        **1️⃣ Upload**
        
        Your customer list, sales history, or any CSV/Excel file
        
        *(Stays on your computer)*
        """
    )

with step2:
    st.markdown(
        """
        **2️⃣ Clean**
        
        Fix errors, remove junk, create new columns
        
        *(We guide you through it)*
        """
    )

with step3:
    st.markdown(
        """
        **3️⃣ Analyze**
        
        Generate charts, find segments, forecast trends
        
        *(Automatic insights)*
        """
    )

with step4:
    st.markdown(
        """
        **4️⃣ Act**
        
        Export results, get AI recommendations, improve marketing
        
        *(Make data-driven decisions)*
        """
    )

st.divider()

# Real-world examples
st.subheader("💼 Real Small Business Examples")

use_case_col1, use_case_col2 = st.columns(2)

with use_case_col1:
    st.markdown(
        """
        **📧 Email Marketing:**
        - Segment customers by engagement level
        - Find who responds to discounts vs. new products
        - Identify best time to send campaigns
        - Reduce unsubscribes by targeting relevantly
        
        **🛍️ E-commerce & Retail:**
        - Group customers by purchase behavior
        - Predict which products will sell next month
        - Find cross-sell opportunities
        - Identify seasonal trends
        """
    )

with use_case_col2:
    st.markdown(
        """
        **🏪 Local Businesses:**
        - Understand your repeat customer base
        - Find your most profitable customer types
        - Optimize loyalty program targeting
        - Forecast busy periods for staffing
        
        **📦 General Analytics:**
        - Inventory forecasting
        - Service usage patterns
        - Operational metrics
        - Any data you want to understand better
        """
    )

st.divider()

# Privacy deep dive
st.subheader("🔒 Why Privacy Matters for Small Businesses")

privacy_col1, privacy_col2 = st.columns(2)

with privacy_col1:
    st.markdown(
        """
        **The Problem with Big Platforms:**
        
        When you upload customer data to Google Analytics, Facebook Ads, 
        or enterprise marketing platforms:
        
        - 🚨 You're giving away YOUR customer insights
        - 🚨 They use it to train AI that helps your competitors
        - 🚨 They track your customers across the web
        - 🚨 You often can't delete the data even if you want to
        - 🚨 You're paying $$$$ per month for the privilege
        """
    )

with privacy_col2:
    st.markdown(
        """
        **The BMY Analytics Difference:**
        
        Everything happens on YOUR computer:
        
        - ✅ Your customer list never touches our servers (we don't have servers!)
        - ✅ No cookies, no tracking, no data collection
        - ✅ You can verify this—our code is open source
        - ✅ Works offline—no internet required after install
        - ✅ Free forever—no monthly subscriptions
        - ✅ You maintain full control and ownership
        """
    )

st.info(
    """
    **🔍 Trust but Verify:** We're open source because privacy claims mean nothing without proof. 
    [Review our code on GitHub](https://github.com/Brandon-m-Y/Marketing_Analytics_App) to see exactly what we do (and don't do) with your data.
    """
)

st.divider()

# Call to action
st.subheader("👉 Ready to Segment Your Customers?")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🎯 Start Analyzing", type="primary", use_container_width=True):
        st.switch_page("pages/1_📥_Data_Intake.py")

with col2:
    st.markdown(
        """
        <a href='https://github.com/Brandon-m-Y/Marketing_Analytics_App' target='_blank'>
            <button style='width: 100%; padding: 0.5rem; background-color: None; border: 1px solid #d0d0d0; border-radius: 0.5rem; cursor: pointer;'>
                ⭐ Star on GitHub
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <a href='https://github.com/Brandon-m-Y/Marketing_Analytics_App/blob/main/readme.md' target='_blank'>
            <button style='width: 100%; padding: 0.5rem; background-color: None; border: 1px solid #d0d0d0; border-radius: 0.5rem; cursor: pointer;'>
                📖 Installation Guide
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

st.caption("💡 **Tip:** Have a customer email list or sales CSV? That's all you need to get started.")

st.divider()

# Footer with creator info and privacy reminder
footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:
    st.caption(
        """
        **Built by Brandon Ytuarte** | Data Analyst & Aspiring Data Scientist  
        [LinkedIn](https://www.linkedin.com/in/brandon-m-ytuarte/) • [GitHub](https://github.com/Brandon-m-Y)  
        
        Questions? Feedback? [Open an issue](https://github.com/Brandon-m-Y/Marketing_Analytics_App/issues) or DM me on LinkedIn.  
        This is a living project—your input shapes what gets built next.
        """
    )

with footer_col2:
    st.caption(
        """
        **🔓 100% Open Source**  
        **🔒 Privacy-First Architecture**  
        **💯 Free Forever**
        
        [MIT License](https://github.com/Brandon-m-Y/Marketing_Analytics_App/tree/main) • [Changelog](https://github.com/Brandon-m-Y/Marketing_Analytics_App/releases)
        """
    )