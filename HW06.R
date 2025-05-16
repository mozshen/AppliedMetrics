

#%% libraries

library(did)
library(haven)
library(dplyr)
library(rlang)
library(ggplot2)

setwd("E:/For Git/AppliedMetrics/Problems/HW06")

#%% reading data

d <- read_dta("ehec_data.dta")

#%% Preprocessing: replace NA in yexp2 with 3000 for "never-treated"
d <- d %>%
  mutate(yexp2 = ifelse(is.na(yexp2), 3000, yexp2))

#%% Estimate group-time ATT using att_gt
# Estimate the ATT(g,t) using Callaway and Sant’Anna’s
estimator
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
















