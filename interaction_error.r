library(pwr)
library(ggplot2)

#sink("/Users/Zach/Documents/Modeling_for_Perception/interactions.txt")
p_data <- read.csv("/Users/Zach/Documents/Modeling_for_Perception/clean_alldata.csv")
attach(p_data)

se <- function(x) sqrt(var(x) / length(x))

#assign our factors
subj <- factor(subject)
con <- factor(condition)
ang <- factor(desired_angle)
mag <- factor(desired_magnitude)

means <- by(angle_error, list(con, ang), mean)
ses <- by(angle_error, list(con, ang), se)

write.csv(t(means), "/Users/Zach/Documents/Modeling_for_Perception/d1means.csv")
write.csv(t(ses), "/Users/Zach/Documents/Modeling_for_Perception/d1ses.csv")

means <- by(rad_mag_error, list(con, ang), mean)
ses <- by(rad_mag_error, list(con, ang), se)

write.csv(t(means), "/Users/Zach/Documents/Modeling_for_Perception/d2means.csv")
write.csv(t(ses), "/Users/Zach/Documents/Modeling_for_Perception/d2ses.csv")

means <- by(rad_mag_error, list(con, mag), mean)
ses <- by(rad_mag_error, list(con, mag), se)

write.csv(t(means), "/Users/Zach/Documents/Modeling_for_Perception/d3means.csv")
write.csv(t(ses), "/Users/Zach/Documents/Modeling_for_Perception/d3ses.csv")

means <- by(rad_mag_error, list(ang, mag), mean)
ses <- by(rad_mag_error, list(ang, mag), se)
write.csv(means, "/Users/Zach/Documents/Modeling_for_Perception/d4means.csv")
write.csv(ses, "/Users/Zach/Documents/Modeling_for_Perception/d4ses.csv")

means <- by(angle_error, list(ang, mag), mean)
ses <- by(angle_error, list(ang, mag), se)
write.csv(means, "/Users/Zach/Documents/Modeling_for_Perception/d5means.csv")
write.csv(ses, "/Users/Zach/Documents/Modeling_for_Perception/d5ses.csv")

conang <- interaction(con, ang)

pdf("/Users/Zach/Documents/Modeling_for_Perception/means_plot.pdf")
p <- ggplot(p_data, aes(x = ang, y = angle_error)) + geom_boxplot(aes(fill = conang))
print(p)
dev.off()

detach(p_data)