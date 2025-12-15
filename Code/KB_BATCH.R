### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(beepr)

#Bringing in Data
library(vroom)
train <- vroom("train.csv")  |>
  select(shot_made_flag, everything()) |>
  mutate(shot_made_flag = as.factor(shot_made_flag))
test <- vroom("test.csv") 

#Setting Up Parallel Computing
library(doParallel)
registerDoParallel(cores = 2)

### FEATURE ENGINEERING ###
#Making Original Recipe
kobe_recipe <- recipe(shot_made_flag ~ ., data = train) |>
  step_rm(c("shot_id", "game_event_id", "game_id", "game_date")) |>
  step_mutate(across(where(is.character), as.factor)) |>
  step_mutate(#New Variables
    distance_from_hoop = sqrt(loc_x^2 + loc_y^2),
    angle_from_hoop = atan2(loc_y, loc_x),
    total_seconds_remaining = (minutes_remaining * 60) + seconds_remaining) |>
  step_mutate(#Interactions
    shot_type_interaction = as.factor(paste(action_type, combined_shot_type, sep = "_")),
    time_period_interaction = as.factor(paste(period, total_seconds_remaining, sep = "_"))) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors())
prep <- prep(kobe_recipe)
baked <- bake(prep, new_data = train)
colnames(baked)

### WORK FLOWS ###
library(xgboost)
#XGBoost
#Defining Model
xgb_model <- boost_tree(
  mode = "classification",
  trees = 500, #500 Trees Originally
  tree_depth = tune(), #6 Tree Depth Originally
  learn_rate = tune()) |> #0.05 Learn Rate Originally
  set_engine("xgboost", early_stopping_rounds = 20)

#Creating a Workflow
xgb_wf <- workflow() |>
  add_recipe(kobe_recipe)|>
  add_model(xgb_model)

### CROSS VALIDATION ###
library(dials)
#Defining Grids of Values
xgb_grid <- grid_regular(tree_depth(range = c(3,10)),
                         learn_rate(range = c(0.01,0.1)),
                          levels = 3) #3 for Testing; 5 for Results

#Splitting Data
xgb_folds <- vfold_cv(train,
                       v = 5,
                       repeats = 1) #1 for Testing; 3 for Results

#Run Cross Validations
xgb_results <- xgb_wf |>
  tune_grid(resamples = xgb_folds,
            grid = xgb_grid,
            metrics = metric_set(mn_log_loss),
            control = control_grid(parallel_over = "resamples"))

#Find Best Tuning Parameters
xgb_best <- xgb_results |>
  select_best(metric = "mn_log_loss")

### SUBMISSION ###
#Finalizing Workflow with Cross Validation
final_xwf <- xgb_wf |>
  finalize_workflow(xgb_best) |>
  fit(data = train)

#Making Predictions
xgb_pred <- predict(final_xwf, new_data = test, type = "prob")

#Formatting Predictions for Kaggle
kaggle_xgb <- xgb_pred %>%
  transmute(
    shot_id = test$shot_id,
    shot_made_flag = .pred_1
  )

#Saving CSV File
vroom_write(x = kaggle_xgb, file = "./XGB_BATCH2.csv", delim = ",")

#Saving Cross Validation Results
saveRDS(xgb_results, file = "xgb_results.rds")

#End Parallel Computing
registerDoSEQ()
