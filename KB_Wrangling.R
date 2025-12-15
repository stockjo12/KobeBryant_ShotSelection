#Bringing in Libraries
library(tidyverse)
library(tidymodels)
library(vroom)

#Bringing in Data
setwd("~/Library/CloudStorage/OneDrive-BrighamYoungUniversity/STAT 348/Coding/Kobe Bryant Shot Selection")
data <- vroom("data.csv")

#Wrangling into Training and Test Datasets
test <- data |> #5000 Obs in Test
  filter(is.na(shot_made_flag)) |>
  select(c(-shot_made_flag, -team_id, -team_name))
train <- data |> 
  filter(shot_made_flag %in% c(0,1)) |>
  select(c(-team_id, -team_name))

#Making Training and Test Datasets
vroom_write(train, "train.csv")
vroom_write(test, "test.csv")
#Metric: Accuracy