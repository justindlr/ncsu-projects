library(MASS)
library(tidyverse)
library(caret)

#Reading in Data
read.csv("C:\\Users\\Justin\\Documents\\NC State\\Fall 2021\\ST 405\\Data\\high_diamond_ranked_10min.csv",header=T)->df

# Split the data into training and test set
set.seed(123)
training.samples <- df$blueWins %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df[training.samples, ]
test.data <- df[-training.samples, ]

# Creating the model 

final_model = glm(blueWins ~ 
                    blueDragons+ redDragons
                  + blueExperienceDiff + blueGoldDiff, family = binomial,
                  data = train.data)

summary(final_model)



# Make predictions
probabilities <- final_model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
# Model accuracy
mean(predicted.classes==test.data$blueWins)

# final model (uses full data)

act_final_model = glm(blueWins ~ 
                        blueDragons+ redDragons
                      + blueExperienceDiff + blueGoldDiff, family = binomial,
                      data = df)

summary(act_final_model)



# Make predictions
probabilities <- final_model2 %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
# Model accuracy
mean(predicted.classes==test.data$blueWins)

#stepwise

#define full model
full.model <- glm(blueWins ~., data = df, family = binomial)
step.model <- full.model %>% stepAIC(trace = FALSE)
summary(step.model)

# Make predictions
probabilities2 <- step.model %>% predict(test.data, type = "response")
predicted.classes2 <- ifelse(probabilities2 > 0.5, "1", "0")
# Model accuracy
mean(predicted.classes2==test.data$blueWins)


      