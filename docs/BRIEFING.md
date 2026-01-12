# Bluerose Project: Discount & Sales Prediction Briefing

## Executive Summary

This project implements machine learning models to optimize discount allocation and predict sales within IFFCO's consumer goods business. Using 2+ years of historical contract, Trade Display Agreement (TDA), and sales data, we deliver predictive analytics for two core business functions: discount optimization and sales forecasting.


---

## Organizational Context

### IFFCO Business Model
IFFCO operates as a consumer goods manufacturer with a complex multi-channel distribution system. The organization manages relationships with major retailers (Dmart, Lulu, etc.) and implements sophisticated promotional strategies to drive market competitiveness. Key aspects:

- **Products**: Oils (1L, 2L, 500mL sizes), biscuits, dairy, ice cream, soaps, chocolates, and branded goods (e.g., Tiffany brand ice cream)
- **Product Sizing**: Within product groups, multiple package sizes (e.g., 100g, 200g snack packets, 1L/2L/500mL oils) identified by unique product numbers
- **Channels**: Direct sales to large retail chains (Dmart, Lulu, Majdar, Quick Supermarket, City Market), wholesale distributors, location-based operations (UAE, specific geographic markets, malls)
- **Organization Structure**:
  - **Brand Organization** (Main IFFCO, e.g., IDS—IFFCO Distribution Services): Official legal entity for contracts
  - **Manufacturing Organizations** (e.g., UPX, UX): Specific units manufacturing/selling products; may differ from brand org if product outsourced to third party
  - For contracts: brand organization mandatory; manufacturing organization specified per product group (as different products may come from different manufacturers)
  - For TDA/SIP: brand and manufacturing orgs defined at plan level (same for all products in plan)

### Siebel CRM System
Siebel is the primary enterprise system used by IFFCO for:
- Contract and promotional deal management (definition, approval workflows)
- Claims processing and financial reconciliation (generating invoices to customers for payment recovery)
- Real-time sales and distribution tracking (sales team data entry)
- Promotional plan execution (contracts, TDAs, normal functionality)
- Integration with Oracle EBS for live sales data

**Why Siebel?** The system bridges contract/promotional management with actual market sales data, enabling claims generation for payment recovery from customers based on negotiated promotional terms.

---

## PROJECT CONTACTS & SUPPORT

### Primary Contacts
- **Vijay**: Main point of contact for data, technical questions, and project guidance
- **Support Team**: Available for Python, machine learning, and implementation help

### Support Model
- **Self-Learning Phase** (Week 1-2): Explore data independently, read documentation, try hands-on exercises
- **Weekly Check-ins**: Discuss progress and blockers
- **Expert Consultation**: Get Python/ML expert walkthroughs if needed (arrange Friday sessions if stuck)
- **Continuous Mentorship**: Daily basis questions welcome after initial self-exploration

### Learning Philosophy
> "This is a playground. You are not exposed to a production environment yet. Get your fundamentals done, grow some muscle, and then come hands-up after practice."

---

## DATA OVERVIEW

### Data Sources & Types

#### 1. **TDA Data** (Trade Display Allowances)
- Funding for product displays/gondolas in retail stores
- Promotional support for point-of-sale visibility

#### 2. **Normal Functionality Data** (Budget-Based)
- Standard promotional budget allocation
- Used for monthly/quarterly planning
- More voluminous and reliable than contract data

#### 3. **Contractual Data**
- Long-term promotional agreements with retailers
- Less reliable/complete for initial analysis
- May require special handling or alternative sources

### Data Structure & Dimensions

**Customer Dimension**:
- Retail chain name (e.g., D-Mart, major supermarkets)
- Channel type (HORECA, Modern Trade, Traditional Trade, E-commerce)
- Customer segment and tier

**Product Dimension**:
- Product group (e.g., Biscuits, Ice Creams, Oils, Soaps)
- Product subgroup
- Brand organization (could differ from manufacturing org)
- Manufacturing organization

**Promotional Dimension**:
- Discount/Rebate percentage
- Tactic type:
  - **OSA Rentals**: Shelf space rental for display/visibility (often 100% marketing credit)
  - **Trade Display Allowances (TDA)**: Funding for in-store displays
  - **BOGO Promotions**: Buy One Get One offers (in-store + online)
  - **Contractual Fixed Rebates**: Negotiated volume discounts (typically 5-15%)
  - **Progressive Rebates**: Volume-based variable discounts
  - **Fixed Marketing Credits**: Marketing support budgets

**Time Dimension**:
- Month / Time period
- **Seasonality**: Critical factor
  - Cold drinks consumption ↑ in summer
  - Ice cream consumption ↑ in summer, ↓ in winter
  - Seasonal demand patterns vary by product

**Transaction Data**:
- Sales amounts (Sell-In: products sold to customer, Sell-Out: end consumer sales)
- Returns and reverse transactions
- Claimed amounts vs. budgeted amounts

### Data Characteristics
- **Historical Period**: 6 months of data provided initially
- **Types**: Mix of Sell-In (B2B) and Sell-Out (B2C) data
- **Granularity**: Product-Customer-Month level
- **Limitations**:
  - Limited historical data (6 months may give unreliable predictions)
  - Single brand focus (competitive data not available)
  - Data sensitivity (competitive brand data cannot be shared)

---

## PHASE 1: EXPLORATORY ANALYSIS & DASHBOARDING (Power BI)

### Objectives
Create business intelligence dashboards to understand promotional effectiveness, market trends, and seasonal patterns.

### Access
- **Tool**: Power BI (Basic access provided via team account)
- **Scope**: Basic Power BI functionality (some advanced features restricted to specialized teams)

### Required Dashboards & Analysis

#### Dashboard 1: Product Consumption Trends
**Question**: Which products are consumed most in the last 6 months?
- **Visualization**: Trend charts showing consumption volume over time
- **Breakdown**: By product group, seasonality
- **Insight**: Identify top-performing products and demand shifts

#### Dashboard 2: Seasonality & Seasonal Patterns
**Question**: How do products perform across different seasons?
- **Seasonal Analysis**:
  - Summer season: Cold drinks, ice creams peak
  - Winter season: Ice cream consumption drops, winter products peak
  - Monsoon/Spring: Different consumption patterns
- **Visualizations**: Line charts, heatmaps by season and product group
- **Metrics**: Sales volume, pricing, promotional intensity by season

#### Dashboard 3: Geographic/Channel Performance
**Question**: Which supermarket/retail chain has highest sales? Which channel performs best?
- **Breakdowns**:
  - By customer (retail chain)
  - By channel (HORECA, Modern Trade, Traditional Trade, E-commerce)
  - By region/geography
- **Metrics**: Total sales, growth rate, market share

#### Dashboard 4: Promotional Impact Analysis
**Question**: What is the effect of different promotional tactics?
- **Comparison**:
  - Display allowances vs. contractual rebates
  - BOGO campaigns vs. fixed discounts
  - Promotional spend vs. incremental sales
- **Metrics**: ROI, sales lift, discount effectiveness by tactic type

#### Dashboard 5: Budget vs. Actual
**Question**: How is promotional budget being spent vs. planned?
- **Analysis**: Budget allocation by product group, channel, time
- **Variance**: Planned vs. actual spend, efficiency

---

## PHASE 2: PREDICTIVE MODELING (Python/Machine Learning)

### Objectives
Build machine learning models to predict promotional strategy outcomes and optimize decision-making.

### Technology Stack
- **Language**: Python
- **ML Algorithms**: Random Forest, Gradient Boosting, or similar ensemble methods
- **Key Libraries**: 
  - pandas, numpy (data processing)
  - scikit-learn (machine learning)
  - matplotlib, seaborn (visualization)

### Prediction Tasks

#### Task 1: Discount Optimization (PRIMARY)
**Business Question**: What discount percentage should we give to each supermarket/customer?

**Details**:
- **Current State**: E.g., currently giving 7% discount to customer X
- **Future State**: Predict optimal discount for next year
- **Factors**:
  - Customer tier and bargaining power
  - Product group and profitability
  - Seasonal demand
  - Competitive landscape
  - Historical response to discounts
  
**Output**: Discount percentage recommendation by customer, product, season

**Complexity**: Limited data, single brand, may result in lower accuracy initially

#### Task 2: Sales Forecasting (PRIMARY)
**Business Question**: What sales value will a product achieve in a specific market/season?

**Details**:
- **Example**: How much sales will Ice Cream generate in Delhi in Summer 2026?
- **Factors**:
  - Seasonality (critical)
  - Customer/channel type
  - Product group characteristics
  - Promotional intensity
  - Historical sales patterns
  - Regional demographics

**Output**: Sales value predictions by product, location, season, customer

#### Task 3: Custom Prediction (TEAM-DEFINED)
**Business Question**: You decide what to predict!

**Requirements**:
- Must be relevant to consumer goods/promotional industry
- Use current domain knowledge or research (e.g., GPT, case studies)
- Team consensus on the third prediction
- Should be actionable and measurable

**Examples** (team decides):
- Optimal promotional timing for maximum impact
- Cannibalization risk (discount impacting full-price sales)
- Customer churn prediction based on promotional changes
- Optimal display/shelf space allocation
- Market expansion opportunity identification

---

## BUDGET LOAD PROCESS (Business Context)

### Annual Planning Cycle
1. **Data Collection**: Historical sales and promotional effectiveness analyzed
2. **Model Predictions**: Use prediction models to forecast scenarios
3. **Budget Allocation**: Based on predictions, determine promotional budget per product/customer
4. **Budget Load**: Annual budgets loaded into Siebel system
5. **Execution**: Monthly/quarterly execution against allocated budget
6. **Monitoring**: Track actual vs. planned spend and ROI

### Data Flow
- **Input**: Historical sales, product data, customer data
- **Processing**: Budget analysis and optimization
- **Output**: Annual promotional budget allocation
- **External**: Budget loaded into separate system, returned to Siebel with derived metrics

---

## PROJECT WORKFLOW & TIMELINE

### Week 1-2: Data Exploration & Learning
- [ ] Receive complete data breakup from Vijay
- [ ] Explore Power BI dashboards basics
- [ ] Understand promotional tactics and business logic
- [ ] Begin Python setup and library familiarization
- [ ] Create initial exploratory dashboards

### Week 3-4: Dashboard Development
- [ ] Complete all 5 Power BI dashboards
- [ ] Analyze seasonality patterns
- [ ] Identify key drivers of sales
- [ ] Document insights and anomalies

### Week 5-6: Model Development
- [ ] Develop discount prediction model
- [ ] Train sales forecasting model
- [ ] Define custom third prediction
- [ ] Initial validation and testing

### Week 7-8: Refinement & Testing
- [ ] Test models on new data (provided at end of exercise)
- [ ] Tune hyperparameters
- [ ] Validate predictions against ground truth
- [ ] Document model performance metrics

### Week 9-10: Project Submission
- [ ] Final predictions on new data
- [ ] Project report and documentation
- [ ] Presentation of findings
- [ ] Submit complete deliverables

**Note**: Timeline is flexible. Focus on quality and learning; exact deadlines set by institute requirements and team capacity.

---

## DELIVERABLES

### Final Project Submission
1. **Power BI Dashboards**
   - 5 fully functional dashboards with insights
   - Clear visualizations and KPIs
   - Actionable business recommendations

2. **Predictive Models**
   - Discount prediction model (trained and validated)
   - Sales forecasting model (trained and validated)
   - Custom prediction model (team-defined)

3. **Documentation**
   - Model architecture and feature engineering approach
   - Performance metrics (MAE, RMSE, R², accuracy)
   - Business interpretation of results
   - Recommendations for implementation

4. **Predictions on New Data**
   - Apply models to new unseen data
   - Provide discount and sales predictions
   - Generate business recommendations

5. **Project Report/Synopsis**
   - Executive summary
   - Data exploration findings
   - Model development process
   - Results and insights
   - Limitations and future improvements

---

## KEY SUCCESS FACTORS

✓ **Understand Dimensions**: Clearly understand product, customer, season, and promotion types
✓ **Seasonality Focus**: Recognize seasonal patterns are critical for accurate predictions
✓ **Domain Context**: Learn why IFFCO uses Siebel, how budget works, what "budget load" means
✓ **Independent Learning**: Invest time understanding data before building models
✓ **Business Thinking**: Translate technical metrics into business recommendations
✓ **Edge Case Handling**: Consider how models handle unknown customers/products/seasons
✓ **Validation**: Test predictions make business sense, not just statistical accuracy

---

## COMMON CHALLENGES & SOLUTIONS

### Challenge 1: Limited Historical Data
- **Issue**: 6 months of data may not capture full seasonality
- **Solution**: Use external seasonal indices, combine with domain knowledge, acknowledge uncertainty

### Challenge 2: Single Brand Focus
- **Issue**: No competitive data available for comparison
- **Solution**: Focus on internal trends, comparative advantage analysis, predictive accuracy

### Challenge 3: Data Quality Issues
- **Issue**: Missing values, outliers, inconsistent categorization
- **Solution**: Data cleaning, outlier investigation, imputation strategies

### Challenge 4: Model Generalization
- **Issue**: Models trained on specific customers/products struggle with new combinations
- **Solution**: Robust feature engineering, handle unknown categories, regularization

---

## LEARNING PHILOSOPHY & EXPECTATIONS

### What This Exercise IS:
- ✓ A learning playground to build foundational skills
- ✓ Opportunity to practice end-to-end data science workflow
- ✓ Introduction to real business problems and data
- ✓ Safe environment to try, fail, learn, and iterate

### What This Exercise IS NOT:
- ✗ Production environment with strict SLAs
- ✗ Expected to achieve perfect accuracy immediately
- ✗ Demonstration of all advanced ML techniques
- ✗ High-stakes decision-making system

### Expected Behavior:
1. **Weeks 1-2**: Explore, understand, ask questions freely
2. **Weeks 3-6**: Develop models, iterate, learn from mistakes
3. **Weeks 7-10**: Polish, validate, prepare for submission

### Support & Help:
- Struggling with Python? → Ask for expert help
- Unclear business logic? → Reach out to Vijay
- Stuck on modeling approach? → Discuss options with team
- Friday expert sessions available if thoroughly stuck

---

## NEXT IMMEDIATE STEPS

1. **Connect with Vijay** for complete data breakup and clarifications
2. **Set up Power BI** access and familiarize yourself with basic features
3. **Download Python libraries**: pandas, scikit-learn, matplotlib
4. **Explore the data**: Load CSV files, understand structure, compute basic statistics
5. **Prepare synopsis**: Document your understanding of the problem and approach
6. **Weekly reporting**: Share progress even if incomplete (learning > perfection)

---

## CONCLUSION

This is an excellent opportunity to apply real-world data science to a genuine business problem in the consumer goods industry. The skills you develop—data exploration, business analysis, machine learning, and problem-solving—are directly transferable to your future career.

**Remember**: The goal is to **learn and grow**, not to achieve perfect results. Ask questions, make mistakes, iterate, and document your learning journey.

---

**Last Updated**: December 15, 2025
**Contact**: Vijay (Primary) | Support Team (Technical)
