# part 2
# pretrend sensitivity analysis

#%% libraries

library(did)
library(haven)
library(dplyr)
library(rlang)
library(ggplot2)
library(fixest)
library(bacondecomp)
library(HonestDiD)

setwd("E:/For Git/AppliedMetrics/Problems/HW06")

# part 1
#%% reading data

d <- read_dta("ehec_data.dta")


d <- d %>%
  filter(year <= 2015) 

states2015 <- d %>%
  filter(yexp2 == 2015) %>%
  distinct(stfips) %>%
  pull(stfips)

d <- d %>%
  filter(!(stfips %in% (states2015))) 

d <- d %>%
  mutate(Di = ifelse(yexp2 == 2014, 1, 0))

d <- d %>%
  mutate(
    D2008 = Di * (year == 2008),
    D2009 = Di * (year == 2009),
    D2010 = Di * (year == 2010),
    D2011 = Di * (year == 2011),
    D2012 = Di * (year == 2012),
    D2014 = Di * (year == 2014),
    D2015 = Di * (year == 2015)
  )

twfe_event <- feols(
  dins ~ D2008 + D2009 + D2010 + D2011 + D2012 + D2014 + D2015 | stfips + year,
  data = d,
  cluster = ~stfips
)

summary(twfe_event)

#%% sensitivity analysis using relative magnitudes

betahat <- coef(twfe_event)

# Extract clustered standard errors (sigma)
sigma <- vcov(twfe_event)

sensitivity <- createSensitivityResults_relativeMagnitudes(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = 5,         # 2008–2012 (relative to base year 2013)
  numPostPeriods = 2,        # 2014–2015
  Mbarvec = seq(0.2, 5, by = 0.5)
)

original_cs <- constructOriginalCS(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = 5,
  numPostPeriods = 2
)

createSensitivityPlot_relativeMagnitudes(sensitivity, original_cs)

