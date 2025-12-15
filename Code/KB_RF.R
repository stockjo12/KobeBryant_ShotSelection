### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(beepr)
library(vroom)
library(ranger)
library(dials)

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
  step_other(all_nominal_predictors(), threshold = 0.05) |>
  step_impute_median(all_numeric_predictors())
# prep_kobe <- prep(kobe_recipe)
# baked <- bake(prep_kobe, new_data = train)

### WORKFLOW ###
#Random Forest
#Defining Model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 500) |> #500 Trees
  set_engine("ranger") |>
  set_mode("classification")

#Creating a Workflow
forest_wf <- workflow() |>
  add_recipe(kobe_recipe) |>
  add_model(forest_model)

### CROSS VALIDATION ###
#Defining Grids of Values
forest_grid <- grid_regular(mtry(range = c(5,20)), 
                            min_n(range = c(5,20)),
                            levels = 3) #3 Levels

#Splitting Data
forest_folds <- vfold_cv(train,
                      v = 5,
                      repeats = 1, #1 for Testing; 3 for Results
                      strata = shot_made_flag)

#Run Cross Validations
forest_results <- forest_wf |>
  tune_grid(resamples = forest_folds,
            grid = forest_grid,
            metrics = metric_set(mn_log_loss),
            control = control_grid(parallel_over = "resamples"))
beepr::beep()

#Find Best Tuning Parameters
forest_best <- forest_results |>
  select_best(metric = "mn_log_loss")

### FIT AND PREDICT ###
#Finalizing Workflow
final_fwf <- forest_wf |>
  finalize_workflow(forest_best) |>
  fit(data = train)
beepr::beep()

#Making Predictions
forest_pred <- predict(final_fwf, new_data = test, type = "prob")

### SUBMISSION ###
#Formatting Predictions for Kaggle
kaggle_forest <- forest_pred |>
  transmute(
    shot_id = test$shot_id,
    shot_made_flag = .pred_1
  )

#Saving CSV File
vroom_write(kaggle_forest, file = "./RF_Test.csv", delim = ",")
