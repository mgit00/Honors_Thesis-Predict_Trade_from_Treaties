Information about dataframes: 

- COW-country-codes.csv: country codes 
- DCAD-v1.0-main.csv & National_COW_4.0.csv: not used
- DCAD-v1.0-dyadic.csv & Dyadic_COW_4.0.csv: used
 
# Data 
When the DCA and Trade datasets are cleaned and joined, there are 366520 entries. 
The year spans from 1980 to 2010. The year is mostly evenly spread throughout and no outliers. 
- This means that when it comes to scaling, use min-max scaling rather than log-scaling since there are no outliers to worry about and it is not a normal distribution.
When looking at the trade levels over the years, it is easy to see that DCAs increase trade levels on average. Interestingly, we see that DCAs have the highest correlation with more trade during 1985- 1995 than other time periods. 

