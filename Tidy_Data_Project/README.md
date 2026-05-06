# Tidy Data Project Overview 

## Overview 

This project is meant to transform a provided data set to follow the "tidy data principles" presented in Hadley Wickham's paper __*Tidy Data*__. 

This project uses an old and simplifed verision of the *Mutant Moneyball Data* dataset. This data is related to the article entitled Mutant Moneyball on Rally. Github origin of this data can be found here: https://github.com/EliCash82/mutantmoneyball/tree/main

For the adapted version that was used for this project, see mutant_moneyball.csv in this repository.


---
## The Data : `mutant_moneyball.csv`

__First Impressions__: 
The value related columns appear to be a bit hectic. Each one following the format of `TotalValue[Decade]_[Source]`, acting as a sort of combo variable For instance, TotalValue90s_ebay would signify the total value of the commic in the 90s on ebay. This format comes off as a bit redundant when one considers that the altered dataset this project is using, only includes the TotalValue... variables and their corresponding X-Men member. So, one of the first things this project will address is turning those `TotalValue[Decade]_[Source]` columns into three tidy and searchable collumns: `value`, `decade`, and `source`. 

There is also a significant amount of missing data and distribution concerns that this project will delve into. 

__Shape:__ 26rows by 17 columns 

* Each row contains observations of total value per decade associated with an Xman
* Associated Xman member can be found in the first column labeled `Member`
* Price sources: *Heritage Auctions, Ebay, Wizard Magazine, and Old Street*
* Decades observed: 60s, 70s, 80s, and 90s
