library(pwr)

#sink data to text file output
sink("/Users/Zach/Documents/Modeling_for_Perception/p_values.txt")
#import csv into r
p_data <- read.csv("/Users/Zach/Documents/Modeling_for_Perception/clean_alldata.csv")
#attach the data to the script so we can use the headers to call data sets
attach(p_data)
#assign our factors
subj <- factor(subject)
con <- factor(condition)
ang <- factor(desired_angle)
mag <- factor(desired_magnitude)

print("Anova Results for the Angle of Motion")
#compute anova for angle measures

model1 <- aov(angle_error ~ (con * ang * mag) + Error(subj / (con * ang * mag)))
atest <- summary(model1)
print(atest)

print("\n")
print("Anova Results for the Magnitude of Motion")
#compute anova for the speed measures

model2 <- aov(speed ~ (con * ang * mag) + Error(subj / (con * ang * mag)))
mtest <- summary(model2)
print(mtest)

detach(p_data)