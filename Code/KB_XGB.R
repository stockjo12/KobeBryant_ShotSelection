### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(beepr)
library(vroom)
library(xgboost)

#Bringing in Data
setwd("~/Library/CloudStorage/OneDrive-BrighamYoungUniversity/STAT 348/Coding/Kobe Bryant Shot Selection")
train <- vroom("train.csv") |>
  select(shot_made_flag, everything()) |>
  mutate(shot_made_flag = as.factor(shot_made_flag))
test <- vroom("test.csv")

### FEATURE ENGINEERING ###
#Making Original Recipe
kobe_recipe <- recipe(shot_made_flag ~ ., data = train) |>
  #Remove irrelevant columns
  step_rm(c("shot_id", "game_event_id", "game_id", "game_date")) |>
  #Convert character columns to factors
  step_mutate(across(where(is.character), as.factor)) |>
  #New variables
  step_mutate(
    distance_from_hoop = sqrt(loc_x^2 + loc_y^2),
    angle_from_hoop = atan2(loc_y, loc_x),
    total_seconds_remaining = (minutes_remaining * 60) + seconds_remaining,
    is_clutch = as.factor(total_seconds_remaining <= 180),
    left_side = as.factor(if_else(loc_x < 0, 1, 0)),
    home_game = as.factor(if_else(str_detect(matchup, "vs"), 1, 0))
  ) |>
  #Interaction features
  step_mutate(
    shot_type_interaction = as.factor(paste(action_type, combined_shot_type, sep = "_")),
    time_period_interaction = as.factor(paste(period, total_seconds_remaining, sep = "_")),
    distance_angle_interaction = distance_from_hoop * angle_from_hoop,
    clutch_period_interaction = as.factor(paste(is_clutch, period, sep = "_"))
  ) |>
  #Handle new factor levels, dummies, and numeric imputation
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors())
# prep_kobe <- prep(kobe_recipe)
# baked <- bake(prep_kobe, new_data = train)

### WORKFLOW ###
#XGBoost
#Defining Model
xgb_model <- boost_tree(
  mode = "classification",
  trees = 750,
  tree_depth = 8,
  learn_rate = 0.01
) |>
  set_engine("xgboost")

#Creating a Workflow
xgb_wf <- workflow() |>
  add_recipe(kobe_recipe) |>
  add_model(xgb_model)

### FIT AND PREDICT ###
#Finalizing Workflow
final_xwf <- fit(xgb_wf, data = train)
beepr::beep()

#Making Predictions
xgb_pred <- predict(final_xwf, new_data = test, type = "prob")

### SUBMISSION ###
#Formatting Predictions for Kaggle
kaggle_xgb <- xgb_pred |>
  transmute(
    shot_id = test$shot_id,
    shot_made_flag = .pred_1
  )

#Saving CSV File
vroom_write(kaggle_xgb, file = "./XGB_Test20.csv", delim = ",")
