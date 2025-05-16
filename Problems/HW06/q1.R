

#%% libraries

library(did)
library(haven)
library(dplyr)
library(rlang)
library(ggplot2)
library(fixest)
library(bacondecomp)
setwd("E:/For Git/AppliedMetrics/Problems/HW06")

# part 1
#%% reading data

d <- read_dta("ehec_data.dta")

#%% Preprocessing: replace NA in yexp2 with 3000 for "never-treated"
d <- d %>%
  mutate(yexp2 = ifelse(is.na(yexp2), 3000, yexp2))

#%% Estimate group-time ATT using att_gt
# Estimate the ATT(g,t) using Callaway and Sant’Anna’s
att_gt_result <- att_gt(
  yname = "dins",           # outcome variable
  tname = "year",           # time variable
  idname = "stfips",            # unit identifier
  gname = "yexp2",          # first treatment period
  data = d,
  control_group = "notyettreated",  # control group
  allow_unbalanced_panel = TRUE
)

#%% Summary of ATT(g,t)
summary(att_gt_result)

#%%
# Compare to DiD estimates calculated by hand

d <- d %>%
  mutate(D = ifelse(yexp2 == 2014, 1, 0))

d_sub <- d %>%
  filter(year %in% c(2013, 2014))

group_means <- d_sub %>%
  group_by(D, year) %>%
  summarize(mean_dins = mean(dins), .groups = "drop")

print(group_means)

ATT_manual= 
  (
  group_means$mean_dins[group_means$D == 1 & group_means$year == 2014] - 
  group_means$mean_dins[group_means$D == 1 & group_means$year == 2013]
  )-
  (
    group_means$mean_dins[group_means$D == 0 & group_means$year == 2014] -
    group_means$mean_dins[group_means$D == 0 & group_means$year == 2013]
    
  )

ATT_function= att_gt_result$att[att_gt_result$group == 2014 & att_gt_result$t == 2014]

if(ATT_function== ATT_manual){
  print("equal!")
}


#%% bonus part for all g and T

compute_att_gt <- function(data, g, t) {
  D_var <- paste0("D_", g)
  
  data <- data %>%
    mutate(!!D_var := ifelse(yexp2 == g, 1, 0))
  
  sub_d <- data %>%
    filter(year %in% c(t - 1, t)) %>%
    filter((!!sym(D_var)) == 1 | yexp2 > t)
  
  group_means <- sub_d %>%
    group_by(!!sym(D_var), year) %>%
    summarize(mean_dins = mean(dins), .groups = "drop")
  
  if (nrow(group_means) < 4) {
    return(NA_real_)  # Return NA if not all 4 means are present
  }
  
  A <- group_means$mean_dins[group_means[[D_var]] == 0 & group_means$year == t - 1]
  B <- group_means$mean_dins[group_means[[D_var]] == 0 & group_means$year == t]
  C <- group_means$mean_dins[group_means[[D_var]] == 1 & group_means$year == t - 1]
  D <- group_means$mean_dins[group_means[[D_var]] == 1 & group_means$year == t]
  
  # Check if any of them are missing
  if (length(c(A, B, C, D)) < 4) {
    return(NA_real_)
  }
  
  return((D - C) - (B - A))
}



gt_pairs <- data.frame(
  g = d$yexp2,
  t = d$year
) %>%
  filter(g != 3000) %>%
  distinct()

att_results <- gt_pairs %>%
  rowwise() %>%
  mutate(att = compute_att_gt(d, g, t)) %>%
  ungroup()

# we can see that averages are equal!

# Estimate dynamic effects using event time (g - t)
dynamic_att <- aggte(att_gt_result, type = "dynamic")

# View event-time ATT estimates
summary(dynamic_att)


# Extract event-time, estimate, and CI from the aggte object
event_study_df <- data.frame(
  event_time = dynamic_att$egt,
  att = dynamic_att$att.egt,
  se = dynamic_att$se.egt
)

# Calculate 95% CI
event_study_df <- event_study_df %>%
  mutate(
    ci_lower = att - 1.96 * se,
    ci_upper = att + 1.96 * se
  )

ggplot(event_study_df, aes(x = event_time, y = att)) +
  geom_line(color = "blue") +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Event Study Plot of ATT(g, g+k)",
    x = "Event Time (t - g)",
    y = "Average Treatment Effect"
  ) +
  theme_minimal()

#%%  Compare to TWFE estimates (part 1)


d <- d %>%
  mutate(D = ifelse(year >= yexp2 & yexp2 != 3000, 1, 0))

ols_result <- feols(
  dins ~ D | stfips + year,     # D as regressor, fixed effects for id and year
  data = d,
  cluster = ~stfips         # cluster SEs at state level
)

summary(ols_result)


att= event_study_df%>%
  filter(event_time>0)

print('From Callaway:')
mean(att$att)
mean(att$se)

print('TWFE:')
ols_result[["coefficients"]][["D"]]
ols_result[["se"]][["D"]]

#%% Bacon Decomposition
bacon_result <- bacon(
  formula = dins ~ D,
  data = d,
  id_var = "stfips",
  time_var = "year"
)


#%% removing untreated
d_treated <- d %>%
  filter(yexp2 != 3000)

att_gt_treated <- att_gt(
  yname = "dins",
  tname = "year",
  idname = "stfips",
  gname = "yexp2",
  data = d_treated,
  control_group = "notyettreated",
  allow_unbalanced_panel = TRUE
)

# Aggregate to dynamic (event-time) ATT
dynamic_att_treated <- aggte(att_gt_treated, type = "dynamic")
summary(dynamic_att_treated)

event_study_df <- data.frame(
  event_time = dynamic_att_treated$egt,
  att = dynamic_att_treated$att.egt,
  se = dynamic_att_treated$se.egt
)

d_treated <- d_treated %>%
  mutate(D = ifelse(year >= yexp2, 1, 0))

ols_result_treated <- feols(
  dins ~ D | stfips + year,
  data = d_treated,
  cluster = ~stfips
)

att_treat= event_study_df%>%
  filter(event_time>0)

print('From Callaway:')
mean(att_treat$att)
mean(att_treat$se)

print('TWFE:')
ols_result_treated[["coefficients"]][["D"]]
ols_result_treated[["se"]][["D"]]

bacon_result <- bacon(
  formula = dins ~ D,
  data = d_treated,
  id_var = "stfips",
  time_var = "year"
)

#%% Even bigger TWFE problems
# when we have trend in value
d <- d %>%
  mutate(
    relativeTime = ifelse(yexp2 != 3000, year - yexp2, NA),  # only for treated
    dins2 = ifelse(!is.na(relativeTime) & relativeTime >= 0, dins + 0.01 * relativeTime, dins)
  )

att_gt_result2 <- att_gt(
  yname = "dins2",
  tname = "year",
  idname = "stfips",
  gname = "yexp2",
  data = d,
  control_group = "notyettreated",
  allow_unbalanced_panel = TRUE
)

dynamic_att2 <- aggte(att_gt_result2, type = "dynamic")

# Recalculate D with same logic
d <- d %>%
  mutate(D = ifelse(year >= yexp2 & yexp2 != 3000, 1, 0))


ols_result2 <- feols(
  dins2 ~ D | stfips + year,
  data = d,
  cluster = ~stfips
)

event_study_df <- data.frame(
  event_time = dynamic_att2$egt,
  att = dynamic_att2$att.egt,
  se = dynamic_att2$se.egt
)

att2= event_study_df%>%
  filter(event_time>0)

print('From Callaway:')
mean(att2$att)
mean(att2$se)

print('TWFE:')
ols_result2[["coefficients"]][["D"]]
ols_result2[["se"]][["D"]]

bacon_result2 <- bacon(
  formula = dins2 ~ D,
  data = d,
  id_var = "stfips",
  time_var = "year"
)





