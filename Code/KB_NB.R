### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(beepr)
library(vroom)
library(dials)
library(discrim)

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
  #Handle new factor levels, dummies, normalize, and numeric imputation
  step_zv(all_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())

### WORKFLOW ###
#Naive Bayes
#Defining Model
bayes_model <- naive_Bayes(Laplace = tune(),
                           smoothness = tune()) |>
  set_engine("naivebayes") |>
  set_mode("classification")

#Creating a Workflow
bayes_wf <- workflow() |>
  add_recipe(kobe_recipe) |>
  add_model(bayes_model)

### CROSS VALIDATION ###
#Defining Grids of Values
bayes_grid <- grid_regular(Laplace(range = c(0, 2)),
                           smoothness(range = c(0.01, 1)),
                           levels = 5) #3 for Testing; 5 for Results

#Splitting Data
bayes_folds <- vfold_cv(train,
                      v = 5,
                      repeats = 3) #1 for Testing; 3 for Results

#Run Cross Validations
bayes_results <- bayes_wf |>
  tune_grid(resamples = bayes_folds,
            grid = bayes_grid,
            metrics = metric_set(mn_log_loss))
beepr::beep()

#Find Best Tuning Parameters
bayes_best <- bayes_results |>
  select_best(metric = "mn_log_loss")

### FIT AND PREDICT ###
#Finalizing Workflow
final_bwf <- bayes_wf |>
  finalize_workflow(bayes_best) |>
  fit(data = train)

#Making Predictions
bayes_pred <- predict(final_bwf, new_data = test, type = "prob")

### SUBMISSION ###
#Formatting Predictions for Kaggle
kaggle_bayes <- bayes_pred |>
  transmute(
    shot_id = test$shot_id,
    shot_made_flag = .pred_1
  )

#Saving CSV File
vroom_write(kaggle_bayes, file = "./Bayes_Test.csv", delim = ",")
