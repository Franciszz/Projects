library(viridis)
library(ggridges)
library(tidyverse)
library(gridExtra)





updown_data <- finance_data[,c('Äê·Ý','Çø¼äÕÇµø·ù')]
colnames(updown_data) <- c('year','percent')
updown_data <- na.omit(updown_data)
updown_data$year <- as.factor(updown_data$year)
updown_data$percent <- round(updown_data$percent/100,2)


updown_data1 <- updown_data[updown_data$percent<4,]
#
# 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 
# 1570 1642 2016 2298 2451 2454 2577 2745 3005 3451


updown_data2 <- updown_data[updown_data$percent>4 & updown_data$year %in% c(2009,2015,2016,2017),]

#### >4
# 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 
#  1   27    1    0    2    1    2   54   23   16



table1 <- table(updown_data1$year)
index1 <- paste(names(table1),paste0('(',table1,')'),sep='\n')
names(index1) <- names(table1)
(updown_pic1 <- ggplot(updown_data1, aes(x=percent, y=year, fill=..x..))+
    geom_density_ridges_gradient(scale=3, rel_min_height=0.01, gradient_lwd = 1.)+
  scale_x_continuous(expand = c(0.01, 0))+
  scale_y_discrete(expand = c(0.01,0),labels = index1)+
  theme_ridges(font_size = 13, grid = FALSE)+
  scale_fill_viridis(name="ÕÇµø·ù", option = "C" )+
  geom_vline(aes(xintercept=0),color='red')+
  theme(axis.title.y = element_blank(),axis.title.x = element_blank(),legend.title = element_text(family = 'Hei',size = 12))
)




table2 <- table(updown_data2$year)
table2 <- table2[table2>0]
index2 <- paste(names(table2),paste0('(',table2,')'),sep='\n')
names(index2) <- names(table2)
(updown_pic2 <- ggplot(updown_data2, aes(x=percent, y=year, fill=..x..))+
    geom_density_ridges_gradient(scale=3, rel_min_height=0.01, gradient_lwd = 1.)+
    scale_x_continuous(expand = c(0.01, 0))+
    scale_y_discrete(expand = c(0.01,0),labels = index2)+
    theme_ridges(font_size = 13, grid = FALSE)+
    scale_fill_viridis(name="ÕÇµø·ù", option = "C")+
    geom_vline(aes(xintercept=0),color='red')+
    theme(axis.title.y = element_blank(),axis.title.x = element_blank(),legend.title = element_text(family = 'Hei',size = 12))
)

ggsave(filename = 'updown_pic1.png',updown_pic1)
ggsave(filename = 'updown_pic2.png',updown_pic2)
