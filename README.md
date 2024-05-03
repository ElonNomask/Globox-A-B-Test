# Globox-A-B-Test
In this project, we analyze the results of an A/B test and create a report of data-driven recommendations based on the findings
# A/B Test Report: Food and Drink Banner

## Purpose
The purpose of the A/B test was to assess the impact of introducing a new banner on the GloBox website's mobile version, specifically related to user conversion rates and average amounts spent.

## Hypotheses

# Null Hypothesis (H0):
H0: The new banner has no significant impact on user conversion rates and average amounts spent.

# Alternative Hypothesis (H1):
H1: The new banner leads to a significant increase in user conversion rates and average amounts spent.


## Methodology
### Test Design
- **Population:** The experiment was conducted on 48943 users of the mobile website. The control group had 24343 users while the treatment group had 24600.

- **Duration:** The test ran from 2023-01-25 to 2023-02-06.

- **Success Metrics:**  User conversion rates and average amounts spent were used to measure success.

## Results
### Data Analysis
- **Pre-Processing Steps:** 

```sql

-- Select all data from the 'users' table
SELECT * FROM users;

-- Select all data from the 'groups' table
SELECT * FROM groups;

-- Select all data from the 'activity' table
SELECT * FROM activity;

```

```sql
--This is the SQL code that we used to get the user ID, the user’s country, the user’s gender, the user’s device type, the user’s test group, whether or not they converted (spent > $0), and how much they spent in total ($0+). 

SELECT
    u.id AS user_id,
    COALESCE(u.country, 'N/A') AS country,
    COALESCE(u.gender, 'N/A') AS gender,
    COALESCE(g.device, 'N/A') AS device_type,
    COALESCE(g.group, 'N/A') AS test_group,
    CASE WHEN a.spent > 0 THEN 'Yes' ELSE 'No' END AS converted,
    COALESCE(SUM(COALESCE(a.spent, 0)), 0) AS total_spent
FROM
    users u
LEFT JOIN
    groups g ON u.id = g.uid
LEFT JOIN
    activity a ON u.id = a.uid
GROUP BY
    u.id, u.country, u.gender, g.device, g.group, a.spent;

--In this query, COALESCE is used for each column (except for user_id) to replace NULL values with the default value ('N/A' for strings, 0 for numeric values). The CASE statement and SUM function remain unchanged.

```



- **Statistical Tests Used:** 

## Hypothesis test to see whether there is a difference in the conversion rate between the two groups. The following code resulted in a p-value and conclusion ?


```python
import pandas as pd
import numpy as np
from scipy.stats import norm


# Load the data from the CSV file
data = pd.read_csv("Globoxdata.csv")


# Convert the 'converted' column to lowercase
data['converted'] = data['converted'].str.lower()


# Separate data for groups A and B
group_A = data[data['test_group'] == 'A']
group_B = data[data['test_group'] == 'B']


# Calculate the sample proportions for each group
p_A = group_A['converted'].value_counts(normalize=True).get('yes', 0)
p_B = group_B['converted'].value_counts(normalize=True).get('yes', 0)


# Calculate the pooled proportion
p_pooled = (group_A['converted'].value_counts().get('yes', 0) + group_B['converted'].value_counts().get('yes', 0)) / (len(group_A) + len(group_B))


# Calculate the sample sizes for each group
n_A = len(group_A)
n_B = len(group_B)


# Calculate the test statistic
z_statistic = (p_A - p_B) / np.sqrt(p_pooled * (1 - p_pooled) * ((1 / n_A) + (1 / n_B)))


# Calculate the p-value
p_value = 2 * (1 - norm.cdf(abs(z_statistic)))


# Print the results
print("Test Statistic (z):", z_statistic)
print("P-value:", p_value)


# Draw a conclusion
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the conversion rate between groups A and B.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in the conversion rate between groups A and B.")

```
- **Output:** 

Test Statistic (z): -4.1664960105779585
P-value: 3.093173032975294e-05

Reject the null hypothesis. There is a significant difference in the conversion rate between groups A and B.

## The 95% confidence interval for the difference in the conversion rate between the treatment and control (treatment-control)

```python
import pandas as pd
import numpy as np
from scipy.stats import norm


# Load the data from the CSV file
data = pd.read_csv("Globoxdata.csv")


# Convert the 'converted' column to lowercase
data['converted'] = data['converted'].str.lower()


# Separate data for groups A and B
group_A = data[data['test_group'] == 'A']
group_B = data[data['test_group'] == 'B']


# Calculate the sample proportions for each group
p_A = group_A['converted'].value_counts(normalize=True).get('yes', 0)
p_B = group_B['converted'].value_counts(normalize=True).get('yes', 0)


# Calculate the sample sizes for each group
n_A = len(group_A)
n_B = len(group_B)


# Calculate the standard error
SE = np.sqrt((p_B * (1 - p_B) / n_B) + (p_A * (1 - p_A) / n_A))


# Calculate the critical value for a 95% confidence level
critical_value = norm.ppf(0.975)


# Calculate the sample statistic (difference in proportions)
sample_statistic = p_B - p_A


# Calculate the margin of error
margin_of_error = critical_value * SE


# Construct the confidence interval
confidence_interval = (sample_statistic - margin_of_error, sample_statistic + margin_of_error)


# Print the results
print("Sample Statistic (Difference in Proportions):", sample_statistic)
print("Standard Error:", SE)
print("Critical Value:", critical_value)
print("Margin of Error:", margin_of_error)
print("95% Confidence Interval:", confidence_interval)

```
**Output:** 
Sample Statistic (Difference in Proportions): 0.00783824943540909
Standard Error: 0.0018800496522399504
Critical Value: 1.959963984540054
Margin of Error: 0.003684829607537356
95% Confidence Interval: (0.004153419827871733, 0.011523079042946445)

## Hypothesis test to see whether there is a difference in the Average amount spent between the two groups. What are the resulting p-value and conclusion?

```python
import pandas as pd
import numpy as np
from scipy.stats import t


# Load the data from the CSV file
data = pd.read_csv("Globoxdata.csv")


# Separate data for groups A and B
group_A = data[data['test_group'] == 'A']
group_B = data[data['test_group'] == 'B']


# Calculate the mean and standard deviation for each group
mean_A = group_A['total_spent'].mean()
mean_B = group_B['total_spent'].mean()
std_A = group_A['total_spent'].std()
std_B = group_B['total_spent'].std()
n_A = len(group_A)
n_B = len(group_B)


# Calculate the test statistic
t_statistic = (mean_A - mean_B) / np.sqrt((std_A**2 / n_A) + (std_B**2 / n_B))


# Calculate the degrees of freedom
degrees_of_freedom = np.sqrt(((std_A**2 / n_A) + (std_B**2 / n_B))**2 / (((std_A**2 / n_A)**2) / (n_A - 1) + ((std_B**2 / n_B)**2) / (n_B - 1)))


# Calculate the p-value
p_value = 2 * (1 - t.cdf(abs(t_statistic), degrees_of_freedom))


# Print the results
print("Test Statistic:", t_statistic)
print("P-value:", p_value)


# Draw a conclusion
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the average amount spent per user between groups A and B.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in the average amount spent per user between groups A and B.")

```
**Output:** 

Test Statistic: -0.05968867409877706
P-value: 0.9524574314699203
Fail to reject the null hypothesis. There is no significant difference in the average amount spent per user between groups A and B.

## The 95% confidence interval for the difference in the average amount spent per user between the treatment and the control (treatment-control)?

```python
import pandas as pd
import numpy as np
from scipy.stats import t


# Load the data from the CSV file
data = pd.read_csv("Globoxdata.csv")


# Separate data for groups A and B
group_A = data[data['test_group'] == 'A']
group_B = data[data['test_group'] == 'B']


# Calculate the mean and standard deviation for each group
mean_A = group_A['total_spent'].mean()
mean_B = group_B['total_spent'].mean()
std_A = group_A['total_spent'].std()
std_B = group_B['total_spent'].std()
n_A = len(group_A)
n_B = len(group_B)


# Calculate the sample statistic (difference in means)
sample_statistic = mean_B - mean_A


# Calculate the standard error
SE = np.sqrt((std_B**2 / n_B) + (std_A**2 / n_A))


# Calculate the degrees of freedom
degrees_of_freedom = np.sqrt(((std_A**2 / n_A) + (std_B**2 / n_B))**2 / (((std_A**2 / n_A)**2) / (n_A - 1) + ((std_B**2 / n_B)**2) / (n_B - 1)))


# Calculate the critical value
critical_value = t.ppf(0.975, degrees_of_freedom)


# Calculate the margin of error
margin_of_error = critical_value * SE


# Construct the confidence interval
confidence_interval = (sample_statistic - margin_of_error, sample_statistic + margin_of_error)


# Print the results
print("Sample Statistic (Difference in Means):", sample_statistic)
print("Standard Error:", SE)
print("Critical Value:", critical_value)
print("Margin of Error:", margin_of_error)
print("95% Confidence Interval:", confidence_interval)

```
**Output:** 

Sample Statistic (Difference in Means): 0.013516040501036386
Standard Error: 0.22644229755663664
Critical Value: 1.970736153450424
Margin of Error: 0.44625802246524243
95% Confidence Interval: (-0.43274198196420605, 0.4597740629662788)





### Findings

## Interpretation
**Results Overview:**

# Conversion Rate Difference Hypothesis Test
Test Statistic (z): -4.1665
P-value: 3.09e-05
Conclusion: Reject the null hypothesis. There is a significant difference in the conversion rate between groups A and B (p < 0.05).
# Mathematically:
H_0: p_A = p_B
H_1: p_A ≠ p_B


# Conversion Rate Difference 95% Confidence Interval
Sample Statistic (Difference in Proportions): 0.0078
Standard Error: 0.00188
Margin of Error: 0.00368
95% Confidence Interval: (0.00415, 0.01152)
Interpretation: We are 95% confident that the true difference in conversion rates between groups A and B falls between 0.00415 and 0.01152.

# Average Amount Spent Difference Hypothesis Test
Test Statistic: -0.0597
P-value: 0.9525
Conclusion: Fail to reject the null hypothesis. There is no significant difference in the average amount spent per user between groups A and B (p > 0.05).
# Mathematically:
H_0: μ_A = μ_B
H_1: μ_A ≠ μ_B


# Average Amount Spent Difference 95% Confidence Interval
Sample Statistic (Difference in Means): 0.0135
Standard Error: 0.2264
Margin of Error: 0.4463
95% Confidence Interval: (-0.4327, 0.4598)
Interpretation: We are 95% confident that the true difference in the average amount spent per user between groups A and B falls between -0.4327 and 0.4598.




## Conclusion:

# Key Takeaways:

There is a significant difference in the conversion rate between groups A and B, with group B exhibiting a higher conversion rate. This suggests that the changes made in group B (treatment group) may have positively impacted user conversion. However, there is no significant difference in the average amount spent per user between groups A and B. Both groups have similar spending behaviors, indicating that the changes implemented in group B did not significantly impact user spending habits compared to group A.

# Limitations/Considerations:

While the difference in conversion rates is statistically significant, other factors not accounted for in this analysis may also influence user behavior. The observed difference may be influenced by sample size, timing of the experiment, or external factors not controlled for in the study. Additionally, while the lack of significant difference in spending suggests similarity between the groups, there may still be subtle differences in user behavior that were not captured in this analysis. External factors such as seasonal trends, marketing campaigns, or economic conditions may also influence user spending patterns.


## Recommendation:

Based on the analysis and findings, I recommend that we proceed with caution and do not launch the changes to all users at this time. While there is evidence of a significant increase in the conversion rate among users exposed to the new experience (group B), the lack of a significant difference in the average amount spent per user between groups A and B suggests that the changes may not have a substantial impact on overall user spending habits. Additionally, there are limitations and considerations regarding the observed effects, including potential external factors and unaccounted variables. Therefore, further investigation and analysis are warranted to better understand the implications of the observed differences before deciding to launch.



## Visualizing Confidence Intervals

To visualize the confidence intervals for the difference in conversion rate and the difference in the average amount spent between the two groups, we can plot them using Matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample statistics for conversion rate
conversion_rate_A = 0.041655
conversion_rate_B = 0.049553
diff_conversion_rate = conversion_rate_B - conversion_rate_A
# Standard errors for conversion rate (assuming pooled proportion)
n_A = 24343
n_B = 24600
pooled_proportion = (conversion_rate_A * n_A + conversion_rate_B * n_B) / (n_A + n_B)
std_error_conversion = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n_A + 1/n_B))
margin_of_error_conversion = 1.96 * std_error_conversion
confidence_interval_conversion = (diff_conversion_rate - margin_of_error_conversion, diff_conversion_rate + margin_of_error_conversion)

# Sample statistics for average amount spent
mean_amount_spent_A = 3.37452
mean_amount_spent_B = 3.39087
diff_mean_amount_spent = mean_amount_spent_B - mean_amount_spent_A
# Standard errors for average amount spent
std_dev_A = 0.22644229755663664
std_dev_B = 0.22644229755663664  # Assuming unequal variances
std_error_amount_spent = np.sqrt(std_dev_A**2/n_A + std_dev_B**2/n_B)
margin_of_error_amount_spent = 1.96 * std_error_amount_spent
confidence_interval_amount_spent = (diff_mean_amount_spent - margin_of_error_amount_spent, diff_mean_amount_spent + margin_of_error_amount_spent)

# Plotting
plt.errorbar(x=0, y=diff_conversion_rate, yerr=margin_of_error_conversion, fmt='o', color='blue', label='Diff. in Conversion Rate')
plt.errorbar(x=1, y=diff_mean_amount_spent, yerr=margin_of_error_amount_spent, fmt='o', color='green', label='Diff. in Avg. Amount Spent')

# Add labels and title
plt.xticks([0, 1], ['Conversion Rate', 'Avg. Amount Spent'])
plt.ylabel('Difference')
plt.title('95% Confidence Intervals')

# Add confidence interval text
plt.text(0, diff_conversion_rate - 0.003, f'CI: {confidence_interval_conversion}', ha='center', color='blue')
plt.text(1, diff_mean_amount_spent - 0.2, f'CI: {confidence_interval_amount_spent}', ha='center', color='green')

# Add legend
plt.legend()

# Show plot
plt.show()


```

## Checking for Novelty Effects
The following SQL code was used to get the data to visually inspect the possibiity of a novel effect.

```sql

SELECT 
    a.dt AS date,
    SUM(CASE WHEN g.group = 'A' THEN a.spent ELSE 0 END) AS group_A_spent,
    SUM(CASE WHEN g.group = 'B' THEN a.spent ELSE 0 END) AS group_B_spent,
    COUNT(DISTINCT CASE WHEN g.group = 'A' THEN g.uid ELSE NULL END) AS group_A_users,
    COUNT(DISTINCT CASE WHEN g.group = 'B' THEN g.uid ELSE NULL END) AS group_B_users
FROM 
    activity AS a
JOIN 
    groups AS g ON a.uid = g.uid
GROUP BY 
    a.dt
ORDER BY 
    a.dt;
    
```
This code was used to visualize the presense of a novelty effect.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('novelty.csv')

# Plot metrics over time
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['group_a_spent'], label='Group A')
plt.plot(data['date'], data['group_b_spent'], label='Group B')
plt.xlabel('Date')
plt.ylabel('Spending')
plt.title('Spending Over Time')
plt.legend()
plt.show()

# Compare group trends
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['group_a_users'], label='Group A')
plt.plot(data['date'], data['group_b_users'], label='Group B')
plt.xlabel('Date')
plt.ylabel('Users')
plt.title('Users Over Time')
plt.legend()
plt.show()

```
